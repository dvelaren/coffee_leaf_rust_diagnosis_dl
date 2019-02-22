# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import datetime
import glob

# construct the argument parser and parse the arguments

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#predictor = dlib.shape_predictor("predictor.dat")

file = open("re_plants/training.xml","w")

file.write("<?xml version='1.0' encoding='ISO-8859-1'?>\n")
file.write("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n")
file.write("<dataset>\n")
file.write("<name>Training plants</name>\n")
file.write("<comment>Rust Images.\n")
file.write("   This images are from rust Dataset\n")
file.write("</comment>\n")
file.write("<images>\n")

files = glob.glob("./re_plants/*")
#print(files)
print(len(files))
contFrame = 0
band = True
font = cv2.FONT_HERSHEY_SIMPLEX
for i in files:

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(i)

	#ret, image = cap.read()
	
	if image is None:
		band = False
	
	if image is not(None):
		image = imutils.resize(image, width=700)
		cv2.imshow("Output", image)
		if contFrame % 1 == 0:
			
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# detect faces in the grayscale image
			
			rects = []
			fromCenter = False
			# Select multiple rectangles
			rects = cv2.selectROIs('Output', image, fromCenter)
			#rects = cv2.selectROI("Output", image, False, fromCenter)

			if len(rects) > 0:
				filename = str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+str(contFrame))+".jpg" 
				cv2.imwrite("re_plants/"+str(filename),image)
				file.write("  <image file='"+str(filename)+"'>\n")
				
			

			# loop over the face detections
			for rect in rects:
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				x,y,w,h = rect
				
				#x,y,w,h = rect
				file.write("    <box top='"+str(y)+"' left='"+str(x)+"' width='"+str(w)+"' height='"+str(h)+"'>\n")
				# show the face number
					 
				
				#cv2.waitKey(10000)
				file.write("    </box>\n")

			if len(rects) > 0:

				file.write("  </image>\n")
		 
		# show the output image with the face detections + facial landmarks
		
		contFrame = contFrame + 1 
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

file.write("</images>\n")
file.write("</dataset>\n")