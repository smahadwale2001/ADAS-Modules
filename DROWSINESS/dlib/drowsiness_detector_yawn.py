import sys
import dlib
import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
#import winsound
from scipy.spatial import distance as dist

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

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	avg = (A + B) / (2.0 * C)
	return avg

def mouth_aspect_ratio(mouth,pt1,pt2):
	A = dist.euclidean(mouth[13], mouth[19])
	B = dist.euclidean(mouth[14], mouth[18])
	C = dist.euclidean(mouth[15], mouth[17])
	MAR = (A + B + C) / (dist.euclidean(pt1,pt2)*3.0)
	return MAR

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 30
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0

t1 = 0

my_parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
my_parser.add_argument('--scale',metavar='scale',type=float,help='scaling of webcam')
my_parser.add_argument('--points',metavar='points',type=int,help='Number of points')
my_parser.add_argument('--record',metavar='record',type=bool,help='Want to record a video or not')
my_parser.add_argument('--filename',metavar='filename',type=str,help='Filename for Recorded video')

args = my_parser.parse_args()

scale = args.scale
points = args.points
record = args.record
filename = args.filename

if points==5:
	name="shape_predictor_5_face_landmarks.dat"
else:
	name="shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(name)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print(lStart,lEnd)
print(rStart,rEnd)

scale = 0.3
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
t_diff = 0
fps=0
counts=0
ear = 0
first_frame = 0
m_counter = 0
global result

if record:
	result = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MP4V'),15,(640,480))

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
    	# For Eyes
    	leftEye = shape[lStart:lEnd]
    	rightEye = shape[rStart:rEnd]
    	mouth_pts = shape[mStart:mEnd]
    	leftEAR = eye_aspect_ratio(leftEye)
    	rightEAR = eye_aspect_ratio(rightEye)
    	#(x, y, w, h) = face_utils.rect_to_bb(rect)
    	#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    	#cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    	#Avg Aspect ration for both eyes
    	ear = (leftEAR + rightEAR) / 2.0
    	mar = mouth_aspect_ratio(mouth_pts,shape[60],shape[64])
    	#c=0
    	for (x, y) in shape:
    		cv2.circle(image, (int(x/scale), int(y/scale)), 3, (0, 255, 0), -1)
    		#cv2.putText(image,str(c),(int(x/scale),int(y/scale)),cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,0,0),1)
    		#c+=1
    cv2.rectangle(image,(10,5),(300,150),(0,0,0),-1)
    cv2.putText(image, "FPS :"+str(fps), (20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, "EAR :"+str(round(ear,2)) + " MAR : "+str(round(mar,2)), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    t2 = time.time()
    counts+=1
    if first_frame < 10:
    	if EYE_AR_THRESH < ear:
    		EYE_AR_THRESH = ear
    	first_frame+=1
    if first_frame == 10:
    	EYE_AR_THRESH *= 0.88
    	first_frame+=1
    if ear < EYE_AR_THRESH and first_frame > 10:
    	COUNTER += 1
    	# if the eyes were closed for a sufficient number of
    	# then sound the alarm
    	if COUNTER >= EYE_AR_CONSEC_FRAMES:
    		cv2.putText(image, "DROWSINESS ALERT!", (20, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    		#winsound.Beep(500,200)
    else:
    	COUNTER = 0
    if mar > 0.3:
    	m_counter+=1
    else:
    	m_counter = 0
    if m_counter > 30:
    	cv2.putText(image, "YAWN", (20, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    	#winsound.Beep(500,200)
    if t2 - t1 >= 2:
    	fps = int(1/2*(counts))
    	counts=0
    	t1 = t2
    #print(image.shape)
    cv2.imshow('my webcam', image)
    if record:
    	result.write(image)
    if cv2.waitKey(1) == 27:
        break
cam.release()
if record:
	result.release()
cv2.destroyAllWindows()