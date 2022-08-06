#!/usr/bin/env python

import glob
import os
import sys
import carla
import random
import time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class data_gen:
	def __init__(self,frame_skip=5):
		self.frame_skip = frame_skip
		self.sub = rospy.Subscriber("/carla/ego_vehicle/camera/rgb/front/image_color", Image, self.save_to_disk)
		self.path = '/home/sam/Project/Carla/Lane_Detection/custom_data/'
		self.bridge = CvBridge()
		self.file = open(self.path+'count.txt','a+')
		print(self.file)
		self.count = int(self.file.read())*frame_skip
		print(self.count)

	def save_to_disk(self,image):
		if self.count % self.frame_skip == 0:
			image = self.bridge.imgmsg_to_cv2(image)
			cv2.imwrite(self.path+str(int(self.count/self.frame_skip))+'.jpg',image)
			#cv2.imshow('Data',image)
			#cv2.waitKey(10)
			self.file.truncate(0)
			self.file.write(str(int(self.count/self.frame_skip)))
			print(int(self.count/self.frame_skip))
		self.count+=1

if __name__ == '__main__':
	rospy.init_node('data_generator',anonymous=True)
	gen = data_gen(15)
	rospy.spin()