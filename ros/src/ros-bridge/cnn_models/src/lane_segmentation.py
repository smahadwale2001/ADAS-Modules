#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tensorflow.keras.models import load_model
import time

UNET = load_model('/home/sam/Project/Carla/ros_9.6/src/ros-bridge/cnn_models/models/lane_seg/UNET_lane_acc_68_last.h5')
flag = 1

def get_lane_seg(data):
	global UNET,flag
	print(flag)
	flag+=1
	if flag%5 != 0:
		return
	t1 = time.time()
	image = np.frombuffer(data.data, dtype=np.uint8).reshape(256, 416, 4)
	print(time.time()-t1)
	image = np.array(image,dtype=np.uint8)[:,:,:3]
	cv2.imwrite('/home/sam/record/rgb/'+str(data.header.stamp)+'.png',image)
	image = cv2.resize(image,(288,192)).reshape(1,192,288,3)
	#cv2.imshow('lane_detection',image)
	#cv2.waitKey(10)
	#return
	result = np.argmax(UNET.predict(image),axis=3).reshape(192,288,1)*127
	result[:90] = 0
	result = np.array(result,dtype=np.uint8)
	#print(result.shape)
	#cv2.imwrite('/home/sam/record/semantic/'+str(data.header.stamp)+'.png',result)
	cv2.imshow('lane_detection',result)
	cv2.waitKey(10)

rospy.init_node('lane_seg',anonymous=True)
sub = rospy.Subscriber('/carla/ego_vehicle/camera/rgb/front/image_color',Image,get_lane_seg)
rospy.spin()