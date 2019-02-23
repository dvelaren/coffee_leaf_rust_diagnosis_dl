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
parser.add_argument("-v", "--verbose", help="show depuration", action="store_true")
parser.add_argument("-fp", "--fp", help="Folder where is the four target folders")

args = parser.parse_args()


class preprocessData:

	global file, folderPath, detector, tempImage
	def __init__(self,folderPath):
		
		self.folderPath = folderPath
		self.detector = dlib.simple_object_detector("detectors/detector_plants_v5_C150.svm")
		self.tempImage = None


	def findTypes4Clean(self,folder):
		json = []
		rgb = []
		re = []
		rgn = []
		# this list has the values which didn't match with any type
		other = []
		#find the files in the whole path
		cont = 0
		# search in rgb files
		folder1 = folder + "rgb_data"
		for root, dirs, files in os.walk(folder1):
			for file in files:
				line = str(os.path.join(root, file))
				#print(line)
				cont+=1
				rgb.append(line)


		# search in re files
		folder2 = folder + "re_data"
		for root, dirs, files in os.walk(folder2):
			for file in files:
				line = str(os.path.join(root, file))
				#print(line)
				cont+=1
				re.append(line)


		# search in rgn files
		folder3 = folder + "rgn_data"
		for root, dirs, files in os.walk(folder3):
			for file in files:
				line = str(os.path.join(root, file))
				#print(line)
				cont+=1
				rgn.append(line)


		# search in json files
		folder4 = folder + "json_data"
		for root, dirs, files in os.walk(folder4):
			for file in files:
				line = str(os.path.join(root, file))
				#print(line)
				cont+=1
				json.append(line)
			

		#print(json)
		
		return json,rgb,re,rgn, other

					
	
	def getGrapichHistogram(self,frame,concatenate=True):
		bgr_planes = cv2.split(frame)
		histSize = 256
		histRange = (0, 256) # the upper boundary is exclusive
		accumulate = False
		b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
		g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
		r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
		hist_w = 512
		hist_h = 400
		bin_w = int(np.round( hist_w/histSize ))
		histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
		cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
		cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
		cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
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

		histImage= cv2.resize(histImage,(200,100))
		if not(concatenate):
			return histImage

		x_offset= int(frame.shape[1]/2)
		y_offset=0
		frame[y_offset:y_offset+histImage.shape[0], x_offset:x_offset+histImage.shape[1]] = histImage

		return frame


	def saveVideoAnalyse(self,rgn):
		frame = cv2.imread(rgn[0])
		frame = np.concatenate((frame, frame), axis=1)
		#fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		#out = cv2.VideoWriter("video.avi", fourcc, 10,(frame.shape[1], frame.shape[0]), True)

		for img in rgn:
			frame =cv2.imread(img)
			
			dets = self.detector(frame)
			print("Number of plants detected: {}".format(len(dets)))
			for k, d in enumerate(dets):
				cv2.rectangle(img_output, (d.left(),d.top() ), (d.left() + d.width(), d.top() + d.height()), (0, 255, 0), 2)
				print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
					k, d.left(), d.top(), d.right(), d.bottom()))

			cv2.imshow("dects",img_output)
			cv2.waitKey(5000)

			src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			dst = cv2.equalizeHist(src)

			img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

			# equalize the histogram of the Y channel
			img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

			# convert the YUV image back to RGB format
			img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
			
			frame = self.getGrapichHistogram(frame)
			img_output = self.getGrapichHistogram(img_output)

			frame = np.concatenate((frame, img_output), axis=1)


			cv2.imshow("rgnImage",frame)

			#out.write(frame)

			#cv2.imshow('calcHist Demo', histImage)
			#cv2.imshow("hist",img_output)

			cv2.waitKey(1)

			# if cv2.waitKey(1) & 0xFF == ord('q'):
			# 	break
	
	def Hist_and_Backproj(self,val):

		hsv = cv2.cvtColor(self.tempImage, cv2.COLOR_BGR2HSV)
		ch = (0, 0)
		hue = np.empty(hsv.shape, hsv.dtype)
		cv2.mixChannels([hsv], [hue], ch)
	
		bins = val
		histSize = max(bins, 2)
		ranges = [0, 180] # hue_range
		
		
		hist = cv2.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
		cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		
		
		backproj = cv2.calcBackProject([hue], [0], hist, ranges, scale=1)
		
		
		cv2.imshow('BackProj', backproj)
		
		
		w = 400
		h = 400
		bin_w = int(round(w / histSize))
		histImg = np.zeros((h, w, 3), dtype=np.uint8)
		for i in range(bins):
			cv2.rectangle(histImg, (i*bin_w, h), ( (i+1)*bin_w, h - int(np.round( hist[i]*h/255.0 )) ), (0, 0, 255), cv2.FILLED)
		cv2.imshow('Histogram', histImg)
		

	def cleanValue(self,pixel):
		if pixel < 0 :
			return 0

		return pixel

	def analysePlant(self, img, debug=False):

		grayscaled = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
		retval, threshold = cv2.threshold(img.copy(), 50, 255, cv2.THRESH_BINARY)
		th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 1)
		gh = self.getGrapichHistogram(img.copy(),False)
		im2, contours, hierarchy = cv2.findContours(th.copy() ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		self.tempImage = img

		#window_image = 'original'
		#cv2.namedWindow(window_image)
		bins = 25
		#cv2.createTrackbar('* Hue  bins: ', window_image, bins, 180, self.Hist_and_Backproj )
		#self.Hist_and_Backproj(bins)

		dets = self.detector(img)
		detsFrame = img.copy()
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(detsFrame,str(len(dets)),(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)

		# To show the new detections
		if debug:			
			for k, d in enumerate(dets):
					x = self.cleanValue(d.left())
					y = self.cleanValue(d.top())
					w = self.cleanValue(d.width())
					h = self.cleanValue(d.height())

					cv2.rectangle(detsFrame, (d.left(),d.top() ), (d.left() + d.width(), d.top() + d.height()), (0, 255, 0), 2)
					print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
						k, d.left(), d.top(), d.right(), d.bottom()))

		gray = cv2.bilateralFilter(grayscaled, 11, 17, 17)
		edged = cv2.Canny(gray, 30, 200)

		im2, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
		#cnts = imutils.grab_contours(cnts)
		#cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
		screenCnt = []
		# This image is temporal for find the back projection value
		
		for c in contours:
			peri = cv2.arcLength(c, True)
			if peri > 400:
				screenCnt.append(c)
				#   print("entro")
			
		#cv2.drawContours(frame, cnts, -1, (0,255,0), 3)
		#cv2.drawContours(img, contours, -1, (0,255,0), 3)
		cantContornos = len(screenCnt)
		cantDets = len(dets)
		valueResponse = True

		if cantContornos == 0 and cantDets == 0:
			valueResponse = False
		elif cantContornos > 7:
			valueResponse = False

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
		# print("---------------------------------")
		# print("Response"+str(valueResponse))
		# print("contornos"+str(cantContornos))
		# print("cantDets"+str(cantDets))
		# print("---------------------------------")
		return valueResponse


	def cleanRe(self,frame):

		x = frame.shape[0]
		y = frame.shape[1]
		ch = frame.shape[2]
		
		dets = self.detector(frame)
		newFrame = np.zeros((x,y,ch),np.uint8)
		#print("Number of plants detected: {}".format(len(dets)))
		for k, d in enumerate(dets):
			x = self.cleanValue(d.left())
			y = self.cleanValue(d.top())
			w = self.cleanValue(d.width())
			h = self.cleanValue(d.height())

			crop_img = frame[y:y+h, x:x+w]

			decision = self.analysePlant(crop_img)
			if decision:
				x_offset= x
				y_offset= y

				newFrame[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img

			cv2.rectangle(frame.copy(), (d.left(),d.top() ), (d.left() + d.width(), d.top() + d.height()), (0, 255, 0), 2)
			#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
				#k, d.left(), d.top(), d.right(), d.bottom()))

		#Save in the image path the new photo
		#cv2.write(img,newFrame)

		
		#cv2.imshow("dects",frame)

		#cv2.waitKey(10000)
		return newFrame


	def cleanRgn(self,frame):

		x = frame.shape[0]
		y = frame.shape[1]
		ch = frame.shape[2]
		
		dets = self.detector(frame)
		newFrame = np.zeros((x,y,ch),np.uint8)
		#print("Number of plants detected: {}".format(len(dets)))
		for k, d in enumerate(dets):
			x = self.cleanValue(d.left())
			y = self.cleanValue(d.top())
			w = self.cleanValue(d.width())
			h = self.cleanValue(d.height())

			crop_img = frame[y:y+h, x:x+w]

			decision = self.analysePlant(crop_img)
			if decision:
				x_offset= x
				y_offset= y

				newFrame[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img

			cv2.rectangle(frame.copy(), (d.left(),d.top() ), (d.left() + d.width(), d.top() + d.height()), (0, 255, 0), 2)
			#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
			#	k, d.left(), d.top(), d.right(), d.bottom()))

		#Save in the image path the new photo
		#cv2.write(img,newFrame)

		
		#cv2.imshow("dects",frame)

		#cv2.waitKey(10000)
		return newFrame

	def analyseRgbPlant(self, img, debug=False):

		ORANGE_MIN = np.array([0, 50, 50],np.uint8)
		ORANGE_MAX = np.array([40 , 255, 255],np.uint8)

		RED_MIN = np.array([0, 100, 100],np.uint8)
		RED_MAX = np.array([160, 100, 100],np.uint8)

		GREEN_MIN = np.array([100, 0, 100],np.uint8)
		GREEN_MAX = np.array([100, 255, 100],np.uint8)

		hsv_img = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2HSV)
		frame_threshed = cv2.inRange(hsv_img, RED_MIN, RED_MAX)

		mask = cv2.inRange(hsv_img, (36, 10, 10), (76, 255,255))

		maskOrange = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
		#mask = cv2.inRange(hsv_img, (36, 25, 25), (70, 255,255))

		## slice the green
		imask = mask>0
		green = np.zeros_like(img, np.uint8)
		green[imask] = img[imask]

		imask2 = maskOrange>0
		orange = green
		orange[imask2] = img[imask2]

		#cv2.imshow("thresh",hsv_img)
		#cv2.imshow("green",green)
		#cv2.imshow("orange",orange)
		#cv2.imshow("img",img)		
				

		valueResponse = False

		return valueResponse,orange

	def cleanRgb(self, frame):
		decision,img = self.analyseRgbPlant(frame,False)
		
		return img

	def cleanJson(self, json_data):
		#print(json_data)
		with open(json_data) as f:
			
			if os.stat(json_data).st_size == 0:
				return 0,0,0,0,0,0,False
			data = json.load(f)
			#print(data)
			
			# validate if all the keys exists

			if 'ph' not in data or 'soil_temperature' not in data or 'soil_moisture' not in data or 'illuminance' not in data or 'env_temperature' not in data or 'env_humidity' not in data:
				return 0,0,0,0,0,0,False

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

		return ph, s_temp, s_moist, illuminance, env_temp, env_humi, band


	def cleanFiles(self,folder):

		jsonf,rgb,re,rgn, idk = self.findTypes4Clean(folder)
		# clean the rgn images
		#loop for clean all the re images
		# for e in re:
		# 	frame =cv2.imread(e)
		# 	#print(frame.shape[2])
		# 	print(e)
		# 	cleanFrame = self.cleanRe(frame)
		# 	cv2.imwrite(e,cleanFrame)
		# 	#cv2.imshow("newFrame",cleanFrame)
		# 	#cv2.waitKey(1)

		# # loop for clean all the re images
		# for n in rgn:
		# 	frame =cv2.imread(n)
		# 	#print(frame.shape[2])
		# 	cleanFrame = self.cleanRgn(frame)
		# 	cv2.imwrite(n,cleanFrame)
		# #	cv2.imshow("newFrame",cleanFrame)
		# 	print(n)

		# for b in rgb:
		# 	frame = cv2.imread(b)
		# 	cleanFrame = self.cleanRgb(frame)
		# 	cv2.imwrite(b,cleanFrame)
		# 	#cv2.imshow("newFrame",cleanFrame)
		# 	print(b)
			

		contTotal = 0
		contGood = 0
		contBad = 0
		for j in jsonf:
			#print(j)
			ph, s_temp, s_moist, illuminance, env_temp, env_humi,Json = self.cleanJson(j)
			#print(Json)
			if Json:
				
				data = {
					'ph' : ph,
					's_temperature' : s_temp,
					's_moisture': s_moist,
					'illuminance' : illuminance,
					'env_temperature' : env_temp,
					'env_humidity' : env_humi
					} 

				#print(data)
				
				with open(j, 'w') as outfile:
					json.dump(data, outfile, indent=4)

				contGood+=1
			if not(Json):
				os.remove(j)
				# print(ph)
				# print(s_temp )
				# print(s_moist)
				# print(illuminance)
				# print(env_temp)
				# print(env_humi)
				# print("------------")

				contBad+=1

			contTotal+=1
			
			#print(Json)

		# print("contT "+str(contTotal))
		# print("contG "+str(contGood))
		# print("contB "+str(contBad))

	

p = preprocessData(args.fp)
p.cleanFiles(args.fp)





