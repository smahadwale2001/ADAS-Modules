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

class camera:
	def __init__(self,blueprint_library,vehicle,im_width=640,im_height=480,fov=110,topic='imageRGB',pos=[2.5,0,0.7],angle=[0,0,0],display='False',cam_type='rgb'):
		self.im_width = im_width
		self.im_height = im_height
		self.fov = fov
		self.topic = topic
		self.spawn_point = carla.Transform(carla.Location(x=pos[0], y=pos[1], z=pos[2]),carla.Rotation(roll=angle[0],pitch=angle[1],yaw=angle[2]))
		self.display = False
		if display == 'True':
			self.display = True
		self.pub = rospy.Publisher(self.topic, Image, queue_size=10)
		self.bridge = CvBridge()
		self.type = cam_type
		self.sensor_blueprint = 'sensor.camera.'+cam_type
		blueprint = blueprint_library.find(self.sensor_blueprint)
		blueprint.set_attribute('image_size_x', str(self.im_width))
		blueprint.set_attribute('image_size_y', str(self.im_height))
		blueprint.set_attribute('fov', str(self.fov))
		sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
		sensor.listen(lambda data: self.process_img(data))
		print(sensor)

	def process_img(self,image_raw):
		image_raw = np.array(image.raw_data,dtype=np.uint8).reshape(self.im_height,self.im_width,4)[:,:,:3]
		if self.display:
			cv2.imshow(self.topic,image_raw)
		self.pub.publish(bridge.cv2_to_imgmsg(image_raw, encoding="passthrough"))
		print(image_raw.shape)

if __name__ == '__main__':
	rospy.init_node('sensors', anonymous=True)
	time.sleep(5)
	# Host and Port for carla client
	params = rospy.get_param('carla')
	host = params['host']
	port = params['port']

	try:
		car  = rospy.get_param('/sensors/model')
	except:
		car = 'model3'

	# Define Carla client
	client = carla.Client(host, 2000)
	client.set_timeout(2.0)

	world = client.get_world()
	blueprint_library = world.get_blueprint_library()

	bp = blueprint_library.filter(car)[0]
	print(bp)

	spawn_point = random.choice(world.get_map().get_spawn_points())

	vehicle = world.spawn_actor(bp, spawn_point)
	vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
	# vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.
	
	# https://carla.readthedocs.io/en/latest/cameras_and_sensors
	# get the blueprint for this sensor
	cam_count = rospy.get_param('/sensors/cam_count')
	cam_lst = []
	for i in range(0,cam_count):
		cam_param = rospy.get_param('/sensors/camera'+str(i+1))
		try:
			print("Image Width : ",cam_param['im_width'])
		except:
			cam_param['im_width'] = 640
			print("Image Width : ",cam_param['im_width'])
		try:
			print("Image Height : ",cam_param['im_height'])
		except:
			cam_param['im_height'] = 480
			print("Image Height : ",cam_param['im_height'])
		try:
			print("FOV : ",cam_param['fov'])
		except:
			cam_param['fov'] = 110
			print("FOV : ",cam_param['fov'])
		try:
			print("ROS Publish Topic :",cam_param['topic'])
		except:
			cam_param['topic'] = 'imageRGB'+str(i+1)
			print("ROS Publish Topic :",cam_param['topic'])
		try:
			print("Position [X,Y,Z] : ",cam_param['pos'])
		except:
			cam_param['pos'] = [2.35,0,0.7]
			print("Position [X,Y,Z] : ",cam_param['pos'])
		try:
			print("Angles [roll,pitch,yaw] : ",cam_param['angle'])
		except:
			cam_param['angle'] = [0,0,0]
			print("Angles [roll,pitch,yaw] : ",cam_param['angle'])
		try:
			print("Output Display : ",cam_param['display'])
		except:
			cam_param['display'] = False
			print("Output Display : ",cam_param['display'])
		try:
			print("Camera Type : ",cam_param['type'])
		except:
			cam_param['type'] = 'rgb'
			print("Camera Type : ",cam_param['type'])
		cam_lst.append(camera(blueprint_library,vehicle,cam_param['im_width'],cam_param['im_height'],cam_param['fov'],cam_param['topic'],cam_param['pos'],cam_param['angle'],cam_param['display'],cam_param['type']))
	while not rospy.is_shutdown():
		continue
