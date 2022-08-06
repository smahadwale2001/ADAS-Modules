import sys
import dlib
import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time

t1 = 0
def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords



my_parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
my_parser.add_argument('--scale',metavar='scale',type=float,help='scaling of webcam')
my_parser.add_argument('--points',metavar='points',type=int,help='Number of points')

args = my_parser.parse_args()

scale = args.scale
points = args.points

if points==5:
	name="shape_predictor_5_face_landmarks.dat"
else:
	name="shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(name)

scale = 0.3
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
t_diff = 0
fps=0
counts=0
while True:
    ret_val, image = cam.read()
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    #image = cv2.resize(image,dim)
    img = cv2.cvtColor(cv2.resize(image,dim), cv2.COLOR_BGR2GRAY)
    rects = detector(img,1)
    for (i, rect) in enumerate(rects):
    	shape = predictor(img, rect)
    	shape = face_utils.shape_to_np(shape)
    	#(x, y, w, h) = face_utils.rect_to_bb(rect)
    	#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    	#cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    	for (x, y) in shape:
    		cv2.circle(image, (int(x/scale), int(y/scale)), 3, (0, 255, 0), -1)
    cv2.rectangle(image,(10,5),(100,30),(0,0,0),-1)
    cv2.putText(image, "FPS :"+str(fps), (20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    t2 = time.time()
    counts+=1
    if t2 - t1 >= 2:
    	fps = int(1/2*(counts))
    	counts=0
    	t1 = t2
    cv2.imshow('my webcam', image)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()