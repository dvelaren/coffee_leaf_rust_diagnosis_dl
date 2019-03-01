import cv2
import os
import argparse
from multiprocessing import Pool
import multiprocessing
import shutil
import numpy as np
import imutils
import dlib
import json


parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--fp", help="Folder where is the four target folders")
args = parser.parse_args()


class preprocessData:

	global file, folderPath, detector
	def __init__(self,folderPath):
		
		self.folderPath = folderPath
		self.detector = dlib.simple_object_detector("detectors/detector_plants_v5_C150.svm")
	
	def findTypes4Clean(self,folder):
		"""
		Search in the given folder for the four types and classify them into a differents lists
		"""
		json = []
		rgb = []
		re = []
		rgn = []
		# this list has the values which didn't match with any type
		other = []
		
		# search and append to the list all files in rgb folder
		folderRgb = folder + "rgb_data"
		for root, dirs, files in os.walk(folderRgb):
			for file in files:
				line = str(os.path.join(root, file))
				rgb.append(line)


		# search and append to the list all files in re folder
		folderRe = folder + "re_data"
		for root, dirs, files in os.walk(folderRe):
			for file in files:
				line = str(os.path.join(root, file))
				re.append(line)


		# search and append to the list all files in rgn folder
		folderRgn = folder + "rgn_data"
		for root, dirs, files in os.walk(folderRgn):
			for file in files:
				line = str(os.path.join(root, file))
				rgn.append(line)


		# search and append to the list all files in json folder
		folderJson = folder + "json_data"
		for root, dirs, files in os.walk(folderJson):
			for file in files:
				line = str(os.path.join(root, file))
				json.append(line)
			
		
		return json,rgb,re,rgn, other

					
	
	def getGraphicHistogram(self,frame,concatenate=True):

		"""
		Get a graphic with the histogram information of a frame
		"""

		# Separate the three channels
		bgr_planes = cv2.split(frame)
		# Define a Size and Range of the table
		histSize = 256
		histRange = (0, 256) 
		accumulate = False

		# calculate the histogram for each channel
		b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
		g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
		r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

		hist_w = 512
		hist_h = 400


		bin_w = int(np.round( hist_w/histSize ))
		# create a new black frame with the frame shapes and with 3 channels 
		histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

		# Normalize the result to [ 0, hist_h ]
		cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
		cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
		cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

		# Draw for each channel
		for i in range(1, histSize):
			cv2.line(histImage, ( bin_w*(i-1), hist_h - int(np.round(b_hist[i-1])) ),
					( bin_w*(i), hist_h - int(np.round(b_hist[i])) ),
					( 255, 0, 0), thickness=2)
			cv2.line(histImage, ( bin_w*(i-1), hist_h - int(np.round(g_hist[i-1])) ),
					( bin_w*(i), hist_h - int(np.round(g_hist[i])) ),
					( 0, 255, 0), thickness=2)
			cv2.line(histImage, ( bin_w*(i-1), hist_h - int(np.round(r_hist[i-1])) ),
					( bin_w*(i), hist_h - int(np.round(r_hist[i])) ),
					( 0, 0, 255), thickness=2)

		# resize the image 
		histImage= cv2.resize(histImage,(200,100))

		# If we don't want to paste it into the Source frame
		if not(concatenate):
			return histImage

		x_offset= int(frame.shape[1]/2)
		y_offset=0

		# Put the histogram frame into the source frame
		frame[y_offset:y_offset+histImage.shape[0], x_offset:x_offset+histImage.shape[1]] = histImage

		return frame


	def saveVideoAnalyse(self,type):

		"""
		Save a video with information of the histogram graphic
		"""
		# Extrat the first frame to get the image shapes
		frame = cv2.imread(rgn[0])
		frame = np.concatenate((frame, frame), axis=1)

		# Create a videowriter to save the video
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		out = cv2.VideoWriter("video.avi", fourcc, 10,(frame.shape[1], frame.shape[0]), True)

		# For each image into the list
		for img in rgn:
			frame =cv2.imread(img)
			
			# Detects the plants into the fram
			dets = self.detector(frame)
			
			# Draw the rect into the image			
			for k, d in enumerate(dets):
				cv2.rectangle(img_output, (d.left(),d.top() ), (d.left() + d.width(), d.top() + d.height()), (0, 255, 0), 2)

			cv2.imshow("dects",img_output)
			cv2.waitKey(1)


			src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			dst = cv2.equalizeHist(src)

			img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

			# equalize the histogram of the Y channel
			img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

			# convert the YUV image back to RGB format
			img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
			
			# Get the histogram graphic
			frame = self.getGraphicHistogram(frame)
			img_output = self.getGraphicHistogram(img_output)

			frame = np.concatenate((frame, img_output), axis=1)

			out.write(frame)			
			cv2.waitKey(1)
	


	def cleanValue(self,pixel):
		"""
			Clean a pixel if this has a negative value
		"""
		if pixel < 0 :
			return 0

		return pixel

	def analysePlant(self, img, debug=False):
		"""
			Analyse if the detect of the plant is a false positive
		"""

		# Convert the img to Gray
		grayscaled = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
		# Get the thresh of the image with a value of 50
		retval, threshold = cv2.threshold(img.copy(), 50, 255, cv2.THRESH_BINARY)

		# Get the thresh of the image with the adaptative method
		th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 1)

		# Get the histogram graphic
		gh = self.getGraphicHistogram(img.copy(),False)
		version = int(cv2.__version__[:1])
		# Find the contours
		if version >= 4:
			contours, _ = cv2.findContours(th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		else:
			_, contours, _ = cv2.findContours(th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# detect if there is a plant to re-confirm 
		dets = self.detector(img)
		detsFrame = img.copy()
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(detsFrame,str(len(dets)),(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)

		# Draw the new detections
		if debug:			
			for k, d in enumerate(dets):
					x = self.cleanValue(d.left())
					y = self.cleanValue(d.top())
					w = self.cleanValue(d.width())
					h = self.cleanValue(d.height())

					cv2.rectangle(detsFrame, (d.left(),d.top() ), (d.left() + d.width(), d.top() + d.height()), (0, 255, 0), 2)
				
		gray = cv2.bilateralFilter(grayscaled, 11, 17, 17)
		edged = cv2.Canny(gray, 30, 200)

        
		# Find the new contourns
		if version >= 4:
			cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		else:
			_, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
		
		screenCnt = []
		
		# Iterate the contours and removed each one which is smaller than 400	
		for c in contours:
			peri = cv2.arcLength(c, True)
			if peri > 400:
				screenCnt.append(c)
			
			
		amountConts = len(screenCnt)
		cantDets = len(dets)
		valueResponse = True

		# If the image doesn't have contourns and a new detection, removed it
		if amountConts == 0 and cantDets == 0:
			valueResponse = False
		elif amountConts > 7:
			valueResponse = False

		# If debug is activated show the screens 
		if debug:
			frame = img.copy()
			cv2.putText(frame,str(len(screenCnt)),(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
			cv2.drawContours(frame, screenCnt, -1, (0, 255, 0), 3)
			cv2.imshow('original',img)
			cv2.imshow('detsFrame',detsFrame)
			cv2.imshow('cnts2',frame)
			cv2.imshow('threshold',threshold)
			cv2.imshow('Adaptitive threshold',th)
			cv2.imshow('graphicH',gh)

		
		
			cv2.waitKey(35000)
	
		return valueResponse


	def cleanRe(self,frame):

		"""
			Clean Re image
		"""

		# Get the frame shapes
		x = frame.shape[0]
		y = frame.shape[1]
		ch = frame.shape[2]
		
		# detect the plants into the image
		dets = self.detector(frame)

		# Create a new black frame
		newFrame = np.zeros((x,y,ch),np.uint8)
		
		# For each plant into the image
		for k, d in enumerate(dets):
			x = self.cleanValue(d.left())
			y = self.cleanValue(d.top())
			w = self.cleanValue(d.width())
			h = self.cleanValue(d.height())

			# Extract the image of the plant
			crop_img = frame[y:y+h, x:x+w]

			# Analyse if it's a plant
			decision = self.analysePlant(crop_img)

			if decision:
				x_offset= x
				y_offset= y

				# Concatenate the plant image into black image
				newFrame[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img

		return newFrame


	def cleanRgn(self,frame):

		"""
			Clean Re image
		"""

		# Get the frame shapes
		x = frame.shape[0]
		y = frame.shape[1]
		ch = frame.shape[2]

		# detect the plants into the image
		dets = self.detector(frame)

		# Create a new black frame
		newFrame = np.zeros((x,y,ch),np.uint8)
	
		# For each plant into the image
		for k, d in enumerate(dets):
			x = self.cleanValue(d.left())
			y = self.cleanValue(d.top())
			w = self.cleanValue(d.width())
			h = self.cleanValue(d.height())

			# Extract the image of the plant
			crop_img = frame[y:y+h, x:x+w]

			# Analyse if it's a plant
			decision = self.analysePlant(crop_img)
			if decision:
				x_offset= x
				y_offset= y

				# Concatenate the plant image into black image
				newFrame[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img


		return newFrame

	def analyseRgbPlant(self, img, debug=False):

		"""
			Analyse rgb image
		"""

		# Establish a orange interval for hsv image
		ORANGE_MIN = np.array([0, 50, 50],np.uint8)
		ORANGE_MAX = np.array([40 , 255, 255],np.uint8)

		# Establish a red interval for hsv image
		RED_MIN = np.array([0, 100, 100],np.uint8)
		RED_MAX = np.array([160, 100, 100],np.uint8)

		# Establish a green interval for hsv image
		GREEN_MIN = np.array([100, 0, 100],np.uint8)
		GREEN_MAX = np.array([100, 255, 100],np.uint8)

		# Transform the image to hsv
		hsv_img = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2HSV)
		frame_threshed = cv2.inRange(hsv_img, RED_MIN, RED_MAX)

		mask = cv2.inRange(hsv_img, (36, 10, 10), (76, 255,255))

		maskOrange = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)

		# slice the green
		imask = mask>0
		# create a black image
		green = np.zeros_like(img, np.uint8)
		# overwrite the black image with the mask
		green[imask] = img[imask]

		#slice the orange
		imask2 = maskOrange>0

		# Use the green-black image to overwrite the orange
		orange = green

		# overwrite the green-black image with the new mask
		orange[imask2] = img[imask2]

		valueResponse = False

		return valueResponse,orange

	def cleanRgb(self, frame):

		"""
			Clean rgb images
		"""
		decision,img = self.analyseRgbPlant(frame,False)
		
		return img

	def cleanJson(self, json_path):
		
		try:
			with open(json_path) as f:
				
				if os.stat(json_path).st_size == 0:
					return {}

				data = json.load(f)
				
				# validate if all the keys exists
				if 'ph' not in data or 'soil_temperature' not in data or 'soil_moisture' not in data or 'illuminance' not in data or 'env_temperature' not in data or 'env_humidity' not in data:
					return {}

				# validate the ranges of the values
				band = True
				ph = data['ph']
				if ph < 4.5 or ph > 9.5:
					band = False

				s_temp = data['soil_temperature']
				if s_temp > 40:
					band = False

				s_moist = data['soil_moisture']
				if s_moist < 10 or s_moist > 95:
					band = False

				illuminance = data['illuminance']
				if illuminance < 40 or illuminance > 15000:
					band = False

				env_temp = data['env_temperature']
				if env_temp < 10 or env_temp > 40:
					band = False

				env_humi = data['env_humidity']
				if env_humi < 20 or env_humi > 99:
					band = False
				document = {}
				# If all the ranges are good save them
				if band:
				
					document = {
						'ph' : ph,
						's_temperature' : s_temp,
						's_moisture': s_moist,
						'illuminance' : illuminance,
						'env_temperature' : env_temp,
						'env_humidity' : env_humi
						} 
					
				else:		
			
					document = {}

				return document
				
		
		except:
			 return {}

		
	def cleanFiles(self,folder):
		"""
			Clean all the types and overwrite them
		"""

		# get all the lists types
		jsonf,rgb,re,rgn, idk = self.findTypes4Clean(folder)
		
		print("Cleaning re data...")
		#loop for clean all the re images
		for e in re:
			frame =cv2.imread(e)
			# get the New clean Frame			
			cleanFrame = self.cleanRe(frame)
			# overwrite the new frame
			cv2.imwrite(e,cleanFrame)
			
		print("Cleaning rgn data...")
		# loop for clean all the rgn images
		for n in rgn:
			frame =cv2.imread(n)
			# get the New clean Frame
			cleanFrame = self.cleanRgn(frame)
			# overwrite the new frame
			cv2.imwrite(n,cleanFrame)
		
		print("Cleaning rgb data...")
		# loop for clean all the rgn images
		for b in rgb:
			frame = cv2.imread(b)
			# get the New clean Frame
			cleanFrame = self.cleanRgb(frame)
			# overwrite the new frame
			cv2.imwrite(b,cleanFrame)
			
		print("Cleaning json data...")
		for j in jsonf:
			# Get the clean document
			document = self.cleanJson(j)
			
			# If there aren't data into the dictionary delete the file
			if len(document) == 0:
				#Remove file
				os.remove(j)
			else:
				# Save the file
				try:
					with open(j,'w') as outfile:
						json.dump(document, outfile, indent=4)
				except:
					None
			

p = preprocessData(args.fp)
p.cleanFiles(args.fp)





