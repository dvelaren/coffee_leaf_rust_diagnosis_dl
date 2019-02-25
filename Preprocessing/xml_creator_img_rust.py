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
parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--fp", help="Folder with images")
args = parser.parse_args()

file = open(str(args.fp)+"/training.xml","w")

# write all the xml header format
file.write("<?xml version='1.0' encoding='ISO-8859-1'?>\n")
file.write("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n")
file.write("<dataset>\n")
file.write("<name>Training plants</name>\n")
file.write("<comment>Rust Images.\n")
file.write("   This images are from rust Dataset\n")
file.write("</comment>\n")
file.write("<images>\n")

files = glob.glob("./"+str(args.fp)+"/*")
contFrame = 0
band = True
font = cv2.FONT_HERSHEY_SIMPLEX
for i in files:

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(i)

	
	if image is None:
		band = False
	
	if image is not(None):
		image = imutils.resize(image, width=700)
		cv2.imshow("Output", image)
		if contFrame % 1 == 0:
			
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			
			rects = []
			fromCenter = False
			# Select multiple rectangles
			rects = cv2.selectROIs('Output', image, fromCenter)
			
			if len(rects) > 0:
				filename = str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+str(contFrame))+".jpg" 
				cv2.imwrite(str(args.fp)+"/"+str(filename),image)
				file.write("  <image file='"+str(filename)+"'>\n")
				
			# loop over the plants selection
			for rect in rects:
				# extract the shapes
				x,y,w,h = rect
				file.write("    <box top='"+str(y)+"' left='"+str(x)+"' width='"+str(w)+"' height='"+str(h)+"'>\n")
				file.write("    </box>\n")

			if len(rects) > 0:
				file.write("  </image>\n")
		 
		contFrame = contFrame + 1 
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

file.write("</images>\n")
file.write("</dataset>\n")