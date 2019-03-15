import cv2
import os
import argparse
import json
import glob
import shutil
from time import time
import collections
import math
import random
import csv
from sklearn.metrics import f1_score
from distutils.dir_util import copy_tree
import sys
import numpy as np
from tensorflow.keras import models as km
sys.path.append('..')
from Model.json_data_structurer import JsonDataStructurer
from Model.rgb_data_structurer import RgbDataStructurer
from Model.re_data_structurer import ReDataStructurer
from Model.rgn_data_structurer import RgnDataStructurer
from Preprocessing.preProcessing import preprocessData
import warnings
warnings.filterwarnings('ignore')

# Creates a argumenter parser to handle the script parameters
parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--fp", help="Folder where is the test set data")
parser.add_argument("-jpfn", "--jpfn", help="Directory where is the json pickle file")
parser.add_argument("-dp", "--dp", help="--Directory where is the predictor (The predictor is on GitHub, into Preprocessing/detectors and the final predictor that must be used is detector_plants_v5_C150.svm ")
parser.add_argument("-smd", "--smd", help="Directory where is the folder which contains the submodels")
args = parser.parse_args()


class modelEvaluator:

	global folder_path, json_data_scaler, json_sub_model, rgb_sub_model, re_sub_model, rgn_sub_model
	global preprocessData, json_data_scaler, json_data_structurer, rgb_data_structurer, re_data_structurer
	def __init__(self,folder_path):
			self.folder_path = folder_path
			self.json_data_structurer = JsonDataStructurer()
			self.rgb_data_structurer = RgbDataStructurer()
			self.re_data_structurer = ReDataStructurer()
			self.rgn_data_structurer = RgnDataStructurer()
			self.json_data_scaler = self.json_data_structurer.load_json_data_scaler(args.jpfn)
			self.preprocess_data = preprocessData('',args.dp)
			print("Loading models ...")
			self.json_sub_model = km.load_model(args.smd + '/' + 'json_sub_model.h5')
			self.rgb_sub_model = km.load_model(args.smd + '/' + 'rgb_sub_model.h5')
			self.re_sub_model = km.load_model(args.smd + '/' + 're_sub_model.h5')
			self.rgn_sub_model = km.load_model(args.smd + '/' + 'rgn_sub_model.h5')
			self.json_sub_model_weight = 0.143480102 # 0.191215455 (not squared f1-score).
			self.rgb_sub_model_weight = 0.304496943 # 0.278559916 (not squared f1-score).
			self.re_sub_model_weight = 0.26099692 # 0.257896141 (not squared f1-score).
			self.rgn_sub_model_weight = 0.291026035 # 0.272328488 (not squared f1-score).



	def get_lot_dirs_by_label_directory(self, data_path):
		"""
		Checks the label directories and returns a list for each one containing the paths to their respective lot
		directories.
		"""
		lot_dirs_by_label_directory = dict()
		for root, dirs, files in os.walk(data_path):
			
			# Iterates over the first layer of directories.
			for directory in dirs:
				directory_path = os.path.join(root, directory)
				lot_dirs = list()
				# Iterates over each element within the directory path.
				for lot_directory in os.listdir(directory_path):
					lot_directory_path = os.path.join(directory_path, lot_directory)
					# Appends the current element to the list if it is a directory.
					lot_dirs.append(lot_directory_path) if os.path.isdir(lot_directory_path) else None
				# Creates an entry on the dictionary for every label directory.
				lot_dirs_by_label_directory[directory] = lot_dirs
			# Only the first layer of directories is required. It is not necessary to keep walking through the tree.
			break
		# Orders the dictionary by the label directory and returns it.
		return collections.OrderedDict(sorted(lot_dirs_by_label_directory.items()))

	def findTypes(self,folder):

		"""
			Search in the given folder for the four types and classify them into a differents lists
		"""

		json = ""
		rgb = []
		re = ""
		rgn = ""
		# this list has the values which didn't match with any type
		other = []
		#find the files in the whole path
		for root, dirs, files in os.walk(folder):
			for file in files:
				line = str(os.path.join(root, file))
				
				if line.find("rgb") != -1:
					rgb.append(line)

				elif line.find("re.") != -1:
					re = line

				elif line.find("rgn.") != -1:
					rgn = line

				elif line.find(".json") != -1:
					json = line

				else:
					other.append(line)
		
		return json,rgb,re,rgn, other

	def get_json_predict(self, data_path):

		"""
			Gets a json value as a dictionary and returns the model prediction label
		"""

		# Cleans the json file
		document = self.preprocess_data.cleanJson(data_path)

		# Tranforms the json values to an array
		json_array = self.json_data_structurer.get_json_values_as_array(document)
		
		json_array = json_array.reshape(1, -1)
		json_data_scaled = self.json_data_structurer.scale_feature_data(json_array, self.json_data_scaler)

		# Predicts what is the label
		json_results = self.json_sub_model.predict(json_data_scaled)

		
		# Finds the max value index
		prediction_index = np.argmax(json_results)
		
		
		# Converts the first output prediction to the true value prediction
		if prediction_index == 1:
			prediction_index = 2

		elif prediction_index == 2:
			prediction_index = 3

		elif prediction_index == 3:
			prediction_index = 4
		
		return int(prediction_index)


	def get_rgb_predict(self, list_data_path):

		"""
			Gets the rgb prediction
		"""

		# Creates a rgb results list for each rgb image
		rgb_results = []
		for rgb in list_data_path:

			# loads the image
			rgb_frame = cv2.imread(rgb)

			# Cleans the frame, extracting the color which is not green
			rgb_cleaned = self.preprocess_data.cleanRgb(rgb_frame)

			#==========================================================================
			# During the process of cleaning the training set data we saved the images and
			# to give warranty that the testing set data takes the same process we'll save the 
			# image and reload it
			#==========================================================================

			# Save the image in a temp file
			cv2.imwrite('temp.jpg',rgb_cleaned)

			# Reloads the image
			rgb_cleaned = cv2.imread('temp.jpg')
			#==========================================================================

			# Resizes the image
			rgb_resized = self.rgb_data_structurer.resize_image(rgb_cleaned)

			# Scales the given feature data and returns the result
			rgb_scaled = self.rgb_data_structurer.scale_feature_data(rgb_resized)

			# Reshapes the array to the it first state
			rgb_expand = np.expand_dims(rgb_scaled, 0)

			# Gets the prediction 
			rgb_result = self.rgb_sub_model.predict(rgb_expand)
			
			# Finds the max value index
			prediction_index = np.argmax(rgb_result)

			# Appends the result to the general list
			rgb_results.append(prediction_index)
		
		# Gets the average 
		final_prediction = sum(rgb_results)/ len(rgb_results)

		# Rounds the average
		final_prediction = self.round_number(final_prediction)

		return int(final_prediction)

	def get_re_predict(self, data_path):

		"""
			Gets the re prediction of an image
		"""

		# Loads the image
		re_frame = cv2.imread(data_path)
		# Cleans the frame, search the plants into the image 
		re_cleaned = self.preprocess_data.cleanRe(re_frame)


		#==========================================================================
		# During the process of cleaning the training set data we saved the images and
		# to give warranty that the testing set data takes the same process we save the 
		# image and reload it
		#==========================================================================
		# Save the image in a temp file
		cv2.imwrite('temp.jpg',re_cleaned)

		# Reloads the image
		re_cleaned = cv2.imread('temp.jpg')
		#==========================================================================

		# Resizes the image
		re_resized = self.re_data_structurer.resize_image(re_cleaned)

		# Scales the given feature data and returns the result
		re_scaled = self.re_data_structurer.scale_feature_data(re_resized)

		# Reshapes the array to the it first state
		re_expand = np.expand_dims(re_scaled, 0)

		# Gets the prediction 
		re_result = self.re_sub_model.predict(re_expand)

		# Finds the max value index
		prediction_index = np.argmax(re_result)

		# Converts the first output prediction to the true value prediction
		if prediction_index == 1:
			prediction_index = 2

		elif prediction_index == 2:
			prediction_index = 3

		elif prediction_index == 3:
			prediction_index = 4


		return int(prediction_index)

	def get_rgn_predict(self, data_path):
		"""
			Gets the rgn prediction of an image
		"""

		# Loads the image
		rgn_frame = cv2.imread(data_path)
		# Cleans the frame, search the plants into the image 
		rgn_cleaned = self.preprocess_data.cleanRgn(rgn_frame)

		#==========================================================================
		# During the process of cleaning the training set data we saved the images and
		# to give warranty that the testing set data takes the same process we'll save the 
		# image and reload it
		#==========================================================================
		# Save the image in a temp file
		cv2.imwrite('temp.jpg',rgn_cleaned)

		# Reloads the image
		rgn_cleaned = cv2.imread('temp.jpg')
		
		# Resizes the image
		rgn_resized = self.rgn_data_structurer.resize_image(rgn_cleaned)

		# Scales the given feature data and returns the result
		rgn_scaled = self.rgn_data_structurer.scale_feature_data(rgn_resized)

		# Reshapes the array to the it first state
		rgn_expand = np.expand_dims(rgn_scaled, 0)

		# Gets the prediction 
		rgn_result = self.rgn_sub_model.predict(rgn_expand)

		# Finds the max value index
		prediction_index = np.argmax(rgn_result)

		# Converts the first output prediction to the true value prediction
		if prediction_index == 1:
			prediction_index = 2

		elif prediction_index == 2:
			prediction_index = 3

		elif prediction_index == 3:
			prediction_index = 4

		return int(prediction_index)
    
	def round_number(self, number):
		int_number = int(number)
		if (number - 0.5) < int_number:
			rounded_number = int_number
		else:
			rounded_number = int_number + 1
		return rounded_number

	def run(self):
		"""
			Main program
		"""

		# Gets an ordered dictionary of the lots folders
		lot_dirs_g = self.get_lot_dirs_by_label_directory(data_path=args.fp)
		
		# List where is going to be all the final results
		list_of_results = []

		# Lists where is going to be the results of each type
		list_json_results = []

		list_rgn_results = []

		list_re_results = []

		list_rgb_results = []

		print("Cleaning and predicting the files ...")
		# Iterates over the all label directories 
		for label_directory, lot_dirs in lot_dirs_g.items():
			
			# Iterates over the all lot directories 
			for lot_directory in lot_dirs:	 
				
				# Divide by type 
				json,rgb,re,rgn,idk = self.findTypes(lot_directory)

				# Gets the json predict
				json_prediction = self.get_json_predict(json)

				# Gets the rgb predict
				rgb_prediction = self.get_rgb_predict(rgb)

				# Gets the re predict
				re_prediction = self.get_re_predict(re)

				# Gets the rgn predict
				rgn_prediction = self.get_rgn_predict(rgn)

				# Inserts the predictions into a list
				list_prediction = [json_prediction, rgb_prediction, re_prediction, rgn_prediction]
                
				# Appends to each list the prediction type value
				list_json_results.append([int(label_directory) , int(json_prediction)])

				list_rgn_results.append([int(label_directory) , int(rgn_prediction)])

				list_re_results.append([int(label_directory) , int(re_prediction)])

				list_rgb_results.append([int(label_directory) , int(rgb_prediction)])

				# Gets the weighted average of the four predictions according to each sub_model's weight.
				final_prediction = (json_prediction * self.json_sub_model_weight)
				final_prediction += (rgn_prediction * self.rgn_sub_model_weight)
				final_prediction += (re_prediction * self.re_sub_model_weight)
				final_prediction += (rgb_prediction * self.rgb_sub_model_weight)

				# Rounds the average
				final_prediction = self.round_number(final_prediction)
			
				# Splits the lot directory
				list_lot_directory = lot_directory.split("/")

				# Gets only the lot number of the folder
				number_lot_directory = (list_lot_directory[len(list_lot_directory)-1]).split('_')[-1]

				# Sets the id lot 
				id_lot = str(label_directory) + number_lot_directory

				# Appends to the results list
				list_of_results.append([str(id_lot),int(label_directory),int(final_prediction)])


		print("Results: ")
		#==========================================================================
		# Gets the f1 score for json predictions
		#==========================================================================
		# Gets only the visual inspection data and put them into a list
		labels_json = [i[0] for i in list_json_results]

		# Gets only the prediction data and put them into a list
		predictions_json = [i[1] for i in list_json_results]
		
		# Gets the f1 score
		f1_score_result_json = f1_score(labels_json, predictions_json, average='weighted')  

		print("json sub-model f1-score: "+str(f1_score_result_json))
		#==========================================================================

		#==========================================================================
		# Gets the f1 score for rgn predictions
		#==========================================================================
		
		# Gets only the visual inspection data and put them into a list
		labels_rgn = [i[0] for i in list_rgn_results]

		# Gets only the prediction data and put them into a list
		predictions_rgn = [i[1] for i in list_rgn_results]
		
		# Gets the f1 score
		f1_score_result_rgn = f1_score(labels_rgn, predictions_rgn, average='weighted')  

		print("rgn sub-model f1-score: "+str(f1_score_result_rgn))

		#==========================================================================

		#==========================================================================
		# Gets the f1 score for re predictions
		#==========================================================================
		
		# Gets only the visual inspection data and put them into a list
		labels_re = [i[0] for i in list_re_results]

		# Gets only the prediction data and put them into a list
		predictions_re = [i[1] for i in list_re_results]
		
		# Gets the f1 score
		f1_score_result_re = f1_score(labels_re, predictions_re, average='weighted')  

		print("re sub-model f1-score: "+str(f1_score_result_re))

		#==========================================================================

		#==========================================================================
		# Gets the f1 score for rgb predictions
		#==========================================================================
		
		# Gets only the visual inspection data and put them into a list
		labels_rgb = [i[0] for i in list_rgb_results]

		# Gets only the prediction data and put them into a list
		predictions_rgb = [i[1] for i in list_rgb_results]
		
		# Gets the f1 score
		f1_score_result_rgb = f1_score(labels_rgb, predictions_rgb, average='weighted')  

		print("rgb sub-model f1-score: "+str(f1_score_result_rgb))

		#==========================================================================

		#==========================================================================
		# Gets the f1 score for the final predictions
		#==========================================================================

		# Gets only the visual inspection data and put them into a list
		labels = [i[1] for i in list_of_results]

		# Gets only the prediction data and put them into a list
		predictions = [i[2] for i in list_of_results]
		
		# Gets the f1 score
		f1_score_result = f1_score(labels, predictions, average='weighted')  
		
		print("general averaged f1-score: "+str(f1_score_result))

		#==========================================================================
		random.shuffle(list_of_results)
		# Split the folder path
		csv_folder_path =  '/'.join((args.fp).split('/')[:-2])

		# Inserts into the first line of the list the header
		list_of_results.insert(0,["ID","Visual inspection","Technological integration"] )

		# Saves the list of results into a csv file
		with open(csv_folder_path + '/' +'results.csv', 'w') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerows(list_of_results)

		# Closes the csv file
		csvFile.close()

		# Removes the temporal image
		os.remove('temp.jpg')

		print("The model evaluation was successful and the result can be found in: "+ str(csv_folder_path + '/' +'results.csv'))

me = modelEvaluator(args.fp)
me.run()

