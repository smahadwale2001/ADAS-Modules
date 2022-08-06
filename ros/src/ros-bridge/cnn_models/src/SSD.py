#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
#from cv_bridge import CvBridge, CvBridgeError
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import mxnet as mx
import gluoncv as gcv

net = model_zoo.get_model('ssd_300_vgg16_atrous_coco', pretrained=True)
flag = 1

def get_ssd_boxes(data,CONFIDENCE = 0.1,SCORE_THRESHOLD = 0.1,IOU_THRESHOLD = 0.5,font_scale = 1,thickness = 1):
	global net,colors,bridge,flag
	flag+=1
	if flag%5 != 0:
		return
	print(flag)
	image = np.frombuffer(data.data, dtype=np.uint8).reshape(256, 416, -1)[:,:,:3]
	#image = np.array(image,dtype=np.uint8)
	#image = cv2.resize(image,(416,256))
	xrgb = mx.nd.array(image).astype('uint8')
	rgb_nd, xrgb = gcv.data.transforms.presets.ssd.transform_test(xrgb, short=300)
	class_IDs, scores, bounding_boxes = net(rgb_nd)
	l = scores.shape[0]
	print(bounding_boxes.shape)
	for i in range(0,l):
		if scores[0,i,0] > 0.5:
			x1,y1,x2,y2 = int(bounding_boxes[:,i,0]),int(bounding_boxes[:,i,1]),int(bounding_boxes[:,i,2]),int(bounding_boxes[:,i,3])
			cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
		else:
			break
	cv2.imshow('SSD',image)
	cv2.waitKey(20)


rospy.init_node('SSD',anonymous=True)
sub = rospy.Subscriber('/carla/ego_vehicle/camera/rgb/front/image_color',Image,get_ssd_boxes)
rospy.spin()