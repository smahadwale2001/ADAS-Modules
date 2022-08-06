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

flag = 0
bridge = CvBridge()
count = 0

frgb = []
fsem = []
fdep = []

lrgb = []
lsem = []
ldep = []

rrgb = []
rsem = []
rdep = []

rergb = []
resem = []
redep = []

trgb = []
tsem = []
tdep = []

def _front_rgb(data):
    global flag,frgb
    frgb = bridge.imgmsg_to_cv2(data)
    flag+=1

def _front_sem(data):
    global fsem
    fsem = bridge.imgmsg_to_cv2(data)

def _front_depth(data):
    global fdep
    test = np.array(np.frombuffer(data.data, dtype=np.uint8).reshape(256, 416, 4),dtype=np.int32)
    fdep = np.array(np.left_shift(test[:,:,3],24) + np.left_shift(test[:,:,2],16) + np.left_shift(test[:,:,1],8) + test[:,:,0],dtype=np.int32)

def _left_rgb(data):
    global lrgb
    lrgb = bridge.imgmsg_to_cv2(data)

def _left_sem(data):
    global lsem
    lsem = bridge.imgmsg_to_cv2(data)

def _left_depth(data):
    global ldep
    test = np.array(np.frombuffer(data.data, dtype=np.uint8).reshape(256, 416, 4),dtype=np.int32)
    ldep = np.array(np.left_shift(test[:,:,3],24) + np.left_shift(test[:,:,2],16) + np.left_shift(test[:,:,1],8) + test[:,:,0],dtype=np.int32)

def _right_rgb(data):
    global rrgb
    rrgb = bridge.imgmsg_to_cv2(data)

def _right_sem(data):
    global rsem
    rsem = bridge.imgmsg_to_cv2(data)

def _right_depth(data):
    global rdep
    test = np.array(np.frombuffer(data.data, dtype=np.uint8).reshape(256, 416, 4),dtype=np.int32)
    rdep = np.array(np.left_shift(test[:,:,3],24) + np.left_shift(test[:,:,2],16) + np.left_shift(test[:,:,1],8) + test[:,:,0],dtype=np.int32)

def _rear_rgb(data):
    global rergb
    rergb = bridge.imgmsg_to_cv2(data)

def _rear_sem(data):
    global resem
    resem = bridge.imgmsg_to_cv2(data)

def _rear_depth(data):
    global redep
    test = np.array(np.frombuffer(data.data, dtype=np.uint8).reshape(256, 416, 4),dtype=np.int32)
    redep = np.array(np.left_shift(test[:,:,3],24) + np.left_shift(test[:,:,2],16) + np.left_shift(test[:,:,1],8) + test[:,:,0],dtype=np.int32)

def _top_rgb(data):
    global trgb
    trgb = bridge.imgmsg_to_cv2(data)

def _top_sem(data):
    global tsem
    tsem = bridge.imgmsg_to_cv2(data)

def save_image():
    global frgb,fsem,lrgb,lsem,rrgb,rsem,rergb,resem,trgb,tsem,count
    count+=1
    np.save('/home/sam/bigdata/rgbd/front/rgb/'+str(count)+'.npy',frgb)
    np.save('/home/sam/bigdata/rgbd/left/rgb/'+str(count)+'.npy',lrgb)
    np.save('/home/sam/bigdata/rgbd/right/rgb/'+str(count)+'.npy',rrgb)
    np.save('/home/sam/bigdata/rgbd/rear/rgb/'+str(count)+'.npy',rergb)
    #np.save('/home/sam/bigdata/rgbd/top/rgb/'+strgbd/r(count)+'.npy',trgb)
    np.save('/home/sam/bigdata/rgbd/front/sem/'+str(count)+'.npy',fsem)
    np.save('/home/sam/bigdata/rgbd/left/sem/'+str(count)+'.npy',lsem)
    np.save('/home/sam/bigdata/rgbd/right/sem/'+str(count)+'.npy',rsem)
    np.save('/home/sam/bigdata/rgbd/rear/sem/'+str(count)+'.npy',resem)
    #np.save('/home/sam/bigdata/rgbd/top/sem/'+str(count)+'.npy',tsem)
    np.save('/home/sam/bigdata/rgbd/front/depth/'+str(count)+'.npy',fdep)
    np.save('/home/sam/bigdata/rgbd/left/depth/'+str(count)+'.npy',ldep)
    np.save('/home/sam/bigdata/rgbd/right/depth/'+str(count)+'.npy',rdep)
    np.save('/home/sam/bigdata/rgbd/rear/depth/'+str(count)+'.npy',redep)
    print(count)


rospy.init_node('IPM_check',anonymous=True)


front_rgb = '/carla/ego_vehicle/camera/rgb/front/image_color'
front_sem = '/carla/ego_vehicle/camera/semantic_segmentation/front/image_segmentation'
front_depth = '/carla/ego_vehicle/camera/depth/front/image_depth'

left_rgb = '/carla/ego_vehicle/camera/rgb/left/image_color'
left_sem = '/carla/ego_vehicle/camera/semantic_segmentation/left/image_segmentation'
left_depth = '/carla/ego_vehicle/camera/depth/left/image_depth'

right_rgb = '/carla/ego_vehicle/camera/rgb/right/image_color'
right_sem = '/carla/ego_vehicle/camera/semantic_segmentation/right/image_segmentation'
right_depth = '/carla/ego_vehicle/camera/depth/right/image_depth'

rear_rgb = '/carla/ego_vehicle/camera/rgb/rear/image_color'
rear_sem = '/carla/ego_vehicle/camera/semantic_segmentation/rear/image_segmentation'
rear_depth = '/carla/ego_vehicle/camera/depth/rear/image_depth'

top_rgb = '/carla/ego_vehicle/camera/rgb/top/image_color'
top_sem = '/carla/ego_vehicle/camera/semantic_segmentation/top/image_segmentation'

s0 = rospy.Subscriber(front_rgb,Image,_front_rgb)
s1 = rospy.Subscriber(front_sem,Image,_front_sem)
sfd = rospy.Subscriber(front_depth,Image,_front_depth)

s2 = rospy.Subscriber(left_rgb,Image,_left_rgb)
s3 = rospy.Subscriber(left_sem,Image,_left_sem)
sld = rospy.Subscriber(left_depth,Image,_left_depth)

s4 = rospy.Subscriber(right_rgb,Image,_right_rgb)
s5 = rospy.Subscriber(right_sem,Image,_right_sem)
srd = rospy.Subscriber(right_depth,Image,_right_depth)

s6 = rospy.Subscriber(rear_rgb,Image,_rear_rgb)
s7 = rospy.Subscriber(rear_sem,Image,_rear_sem)
sred = rospy.Subscriber(rear_depth,Image,_rear_depth)

#s8 = rospy.Subscriber(top_rgb,Image,_top_rgb)
#s9 = rospy.Subscriber(top_sem,Image,_top_sem)

count = len(os.listdir('/home/sam/bigdata/rgbd/front/rgb/'))
while True:
    if flag == frame_skip:
        save_image()
        flag = 0
rospy.spin()