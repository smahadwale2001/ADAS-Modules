#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

config_path = "/home/sam/Project/Carla/ros_9.6/src/ros-bridge/cnn_models/models/yolov3/yolov3.cfg"
config_path = "/home/sam/Project/Carla/ros_9.6/src/ros-bridge/cnn_models/models/tiny_yolo/tiny_yolo.cfg"
# the YOLO net weights file
weights_path = "/home/sam/Project/Carla/ros_9.6/src/ros-bridge/cnn_models/models/yolov3/yolov3.weights"
weights_path = "/home/sam/Project/Carla/ros_9.6/src/ros-bridge/cnn_models/models/tiny_yolo/tiny_yolo.weights"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
labels = open("/home/sam/Project/Carla/ros_9.6/src/ros-bridge/cnn_models/models/tiny_yolo/coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

bridge = CvBridge()
flag = 1

def get_yolo_boxes(data,CONFIDENCE = 0.1,SCORE_THRESHOLD = 0.1,IOU_THRESHOLD = 0.5,font_scale = 1,thickness = 1):
	global net,colors,bridge,flag
	print(data.header)
	flag+=1
	if flag%5 != 0:
		return
	image = np.frombuffer(data.data, dtype=np.uint8).reshape(256,416, -1)[:,:,:3]
	image = np.array(image,dtype=np.uint8)
	#cv2.imwrite('/home/sam/record/rgb/'+str(data.header.stamp)+'.png',image)
	#image = bridge.imgmsg_to_cv2(data)[:,:,:3]
	#print(image.shape)
	#cv2.imshow('image',image)
	h, w = 256,416
	# create 4D blob
	blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,256), swapRB=False, crop=False)
	net.setInput(blob)
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
	layer_outputs = net.forward((ln))#['yolo_16', 'yolo_23']))
	boxes, confidences, class_ids = [], [], []
	# loop over each of the layer outputs
	for output in layer_outputs:
		for detection in output:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > CONFIDENCE:
				box = detection[:4] * np.array([w, h, w, h])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				class_ids.append(class_id)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
	#print(len(idxs))
	if len(idxs) > 0:
	# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			print(class_ids[i])
			if class_ids[i] == 0:
				text = 'Person  '+str(int(confidences[i]*100))
			elif class_ids[i] == 2 or class_ids[i] == 3 or class_ids[i] == 5 or class_ids[i] == 7:
				text = 'Vehicle  '+str(int(confidences[i]*100))
			elif class_ids[i] == 9:
				text = 'Traffic Light  '+str(int(confidences[i]*100))
			else:
				continue
			x, y = boxes[i][0], boxes[i][1]
			w, h = boxes[i][2], boxes[i][3]
			color = [int(c) for c in colors[class_ids[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
			(text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
			text_offset_x = x
			text_offset_y = y - 5
			box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
			overlay = image.copy()
			cv2.rectangle(overlay, box_coords[0], box_coords[1], color, thickness=cv2.FILLED)
			image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
	#cv2.imwrite('/home/sam/record/yolo/'+str(data.header.stamp)+'.png',image)
	cv2.imshow('YOLO',image)
	cv2.waitKey(10)


rospy.init_node('YOLO',anonymous=True)
sub = rospy.Subscriber('/carla/ego_vehicle/camera/rgb/front/image_color',Image,get_yolo_boxes)
rospy.spin()