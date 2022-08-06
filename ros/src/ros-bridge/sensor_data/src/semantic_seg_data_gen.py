#!/usr/bin/env python

import os
import carla
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import signal
 
def handler(signum, frame):
    print("\nExit , Ctrl-c was pressed.")
    exit(1)
 
signal.signal(signal.SIGINT, handler)

frame_skip = 10
flag = 0
path = '/home/sam/rgbd/'

color_image = []
sem_image = []
depth_image = []
flag = 0
bridge = CvBridge()
count = 0

def _color_img(image):
    global flag,color_image
    color_image = bridge.imgmsg_to_cv2(image)
    flag +=1

def _sem_img(s_image):
    global sem_image
    sem_image = bridge.imgmsg_to_cv2(s_image)

def _depth_img(d_image):
    global depth_image
    depth = np.array(np.frombuffer(d_image.data,dtype=np.uint8),dtype=np.uint32).reshape(288,480,4)
    depth_image = np.left_shift(depth[:,:,3],24) + np.left_shift(depth[:,:,2],16) + np.left_shift(depth[:,:,1],8) + depth[:,:,0]
    #depth_image = bridge.imgmsg_to_cv2(d_image)#,desired_encoding='32fc1')

def save_image():
    global count,color_image,sem_image,file
    print(count)
    cv2.imwrite(path+'images/'+str(count)+'.jpg',color_image)
    np.save(path+'labels/'+str(count),sem_image)
    np.save(path+'depth/'+str(count),depth_image)
    #cv2.imwrite(path+'labels/'+str(count)+'.jpg',sem_image)
    count+=1


rospy.init_node('IPM_check',anonymous=True)
sub = rospy.Subscriber("/carla/ego_vehicle/camera/rgb/front/image_color", Image, _color_img)
sub1 = rospy.Subscriber("/carla/ego_vehicle/camera/semantic_segmentation/front/image_segmentation",Image,_sem_img)
sub2 = rospy.Subscriber("/carla/ego_vehicle/camera/depth/front/image_depth",Image, _depth_img)
count = len(os.listdir(path+'images/')) + 1
while True:
    if flag == frame_skip:
        save_image()
        flag = 0
rospy.spin()