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

flag = 0
path = '/home/sam/Project/Carla/ros/src/ros-bridge/sensor_data/dataset/lane_detection'

bridge = CvBridge()

def get_ipm(color_img_canny,scale=0.5):
    pts1 = np.float32([[0,300], [800,300],[0, 500], [800, 500]]) * scale
    pts2 = np.float32([[0, 0], [800, 0],[320, 500], [480, 500]]) * scale #280
    #pts1 = np.float32([[0,180], [400,180],[0, 300], [400, 300]])
    #pts2 = np.float32([[0, 0], [400, 0],[160, 300], [240, 300]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(color_img_canny, matrix, (400,250))
    return result

def getCannyPerspective(test_img):
    color_img = test_img
    #color_img = cv2.resize(color_img,(int(800*scale),int(500*scale)))
    color_img_canny = cv2.Canny(cv2.GaussianBlur(color_img,(1,1),cv2.BORDER_DEFAULT),50,50)
    result = get_ipm(color_img_canny)
    lines_p = cv2.HoughLinesP(result, 1, np.pi / 180, 100, None, 50, 10)
    result = cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
    if lines_p is None:
        return color_img_canny,result,lines_p 
    for i in range(0, len(lines_p)):
        l = lines_p[i][0]
        cv2.circle(result,(l[0],l[1]),3,(0,255,0),3)
        cv2.circle(result,(l[2],l[3]),3,(0,255,0),3)
        cv2.line(result, (l[0], l[1]), (l[2], l[3]), (255,255,0), 1, cv2.LINE_AA)
    return color_img_canny,result,lines_p

def show_ipm(image):
    image = bridge.imgmsg_to_cv2(image)
    #cv2.imwrite('/home/sam/data/img_sized.jpg',image)
    #ipm,result,lines_p = getCannyPerspective(image)
    #cv2.imshow('IPM_color',get_ipm(image))
    #cv2.imshow('Canny',result)
    #cv2.imshow('IPM_Canny',ipm)
    cv2.imshow('Image',image)
    cv2.waitKey(10)

if __name__ == '__main__':
    rospy.init_node('IPM_check',anonymous=True)
    sub = rospy.Subscriber("/carla/ego_vehicle/camera/rgb/front/image_color", Image, show_ipm)
    #sub = rospy.Subscriber("/carla/ego_vehicle/camera/semantic_segmentation/front/image_segmentation",Image,show_ipm)
    rospy.spin()
