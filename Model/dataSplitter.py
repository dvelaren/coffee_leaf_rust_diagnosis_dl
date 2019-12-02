import os
import argparse
import json
import glob
import shutil
from time import time
import collections
import math
import os
import random
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--fp", help="Folder where is the data")
parser.add_argument("-dn", "--dn", help="The name of the new folder where is going to be the data")

args = parser.parse_args()


class dataStructuring:

	def __init__(self,folder_path,destination_path):
			
			self.folder_path = folder_path
			self.destination_path = destination_path


	def split_data_into_train_test(self, data_path, destination_path):
		"""
		Splits the content inside the given data path into train and test sets and creates two new directories with
		the result.
		"""
		print("Splitting data into train and test...")
		train_set_path = os.path.join(destination_path, "train_set")
		test_set_path = os.path.join(destination_path, "test_set")
		self.create_data_split_directory(destination_path=destination_path, train_set_path=train_set_path,
										 test_set_path=test_set_path)

		lot_dirs_by_label_directory = self.get_lot_dirs_by_label_directory(data_path=data_path)

		return lot_dirs_by_label_directory


	def create_data_split_directory(self, destination_path, train_set_path, test_set_path):
		"""Creates the directories for the data split."""
		if os.path.exists(destination_path):
			shutil.rmtree(destination_path)
		os.mkdir(destination_path)
		initial_time = time()
		# Waits for some time before creating the directories inside the one that was just created.
		while (time() - initial_time) < 1:
			None
		os.mkdir(train_set_path)
		os.mkdir(test_set_path)

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


	def elements_numbers_for_type(self, data_path):
		"""
			Gets the amount of each data type
		"""
		cont_rgb = 0
		cont_re = 0
		cont_rgn = 0
		cont_json = 0

		for root, dirs, files in os.walk(data_path):
			# Iterates over all the files into de lot folder
			for file in files:

				# Finds the type of the file and sum it to its count
				if file.find(".json") != -1:
					cont_json += 1

				if file.find("re.JPG") != -1:
					cont_re += 1

				if file.find("rgn.JPG") != -1:
					cont_rgn += 1

				if file.find("plant") != -1:
					cont_rgb += 1

			
		return cont_json, cont_rgb, cont_rgn, cont_re


	def get_stop_criterion(self,list_conts):
		"""
			Gets the stop criterion of the label and its index
		"""
		min_value = min(list_conts)
		min_index_value = list_conts.index(min(list_conts))
		type_selection = ""

		if min_index_value == 0:
			type_selection = "json"
		
		if min_index_value == 1:
			type_selection = "rgb"

		if min_index_value == 2:
			type_selection = "rgn"

		if min_index_value == 3:
			type_selection = "re"

		stop_criterion = math.ceil(min_value * 0.25)

		return(stop_criterion,type_selection)


	def valid_for_test(self, lot_directory):
		"""
			Gets the amount of each data type
		"""
		cont_rgb = 0
		cont_re = 0
		cont_rgn = 0
		cont_json = 0

		for root, dirs, files in os.walk(lot_directory):
			# Iterates over all the files into de lot folder
			for file in files:

				# Finds the type of the file and sum it to its count
				if file.find(".json") != -1:
					cont_json += 1

				if file.find("re.JPG") != -1:
					cont_re += 1

				if file.find("rgn.JPG") != -1:
					cont_rgn += 1

				if file.find("plant") != -1:
					cont_rgb += 1

		# If It has all the type files return true
		if cont_json >= 1 and cont_rgb >= 1 and cont_rgn >= 1 and cont_re >= 1 :
			return True

		return False

	def validate_stop_criterion(self, lot_directory, type_selection):

		"""
			Gets the amount of each data type
		"""
		cont_rgb = 0
		cont_re = 0
		cont_rgn = 0
		cont_json = 0

		for root, dirs, files in os.walk(lot_directory):
			# Iterates over all the files into de lot folder
			for file in files:

				# Finds the type of the file and sum it to its count
				if file.find(".json") != -1:
					cont_json += 1

				if file.find("re.JPG") != -1:
					cont_re += 1

				if file.find("rgn.JPG") != -1:
					cont_rgn += 1

				if file.find("plant") != -1:
					cont_rgb += 1


		if type_selection == "json":
			return cont_json
		
		if type_selection == "rgb":
			return cont_rgb

		if type_selection == "rgn":
			return cont_rgn

		if type_selection == "re":
			return cont_re
		
	def copy_tree(self,list_set,list_set_path,label_directory):

		"""
			Copies the lot directory in destination directory
		"""
		train_lot_dirs_destination = os.path.join(list_set_path,label_directory)
		for list_index in list_set:
			lot_directory_path_list = list_index.split('/')
			# Extracts the name of the lot directory.
			lot_directory_name = lot_directory_path_list[len(lot_directory_path_list) - 1]
			lot_directory_destination = os.path.join(train_lot_dirs_destination, lot_directory_name)
			# Copies the entire lot directory to the corresponding destination and preserves its content.
			shutil.copytree(list_index, lot_directory_destination)
			

	def run(self):

		# Split the folder by '/'
		folder_path_list = self.folder_path.split('/')

		# Removes the last element of the list
		folder_path_list = folder_path_list[:len(folder_path_list)-2]

		# Create a new folder path without the last element, and concatenate the new name 
		new_folder_path = '/'.join(folder_path_list) + '/'+ args.dn
		self.destination_path = new_folder_path 
		
		lot_dirs_g = self.split_data_into_train_test(self.folder_path,self.destination_path)
		
		for label_directory, lot_dirs in lot_dirs_g.items():
			# shuffle the list for each label
			random.shuffle(lot_dirs)

			cont_label_json = 0
			cont_label_rgb = 0
			cont_label_rgn = 0
			cont_label_re = 0
			
			# Iterates over the lot directories and count the amount of each type
			for lot_directory in lot_dirs:
			
				cont_json, cont_rgb, cont_rgn, cont_re = self.elements_numbers_for_type(lot_directory)
				cont_label_json += cont_json
				cont_label_rgb += cont_rgb
				cont_label_rgn += cont_rgn
				cont_label_re += cont_re

			list_conts = [cont_label_json, cont_label_rgb, cont_label_rgn, cont_label_re]

			# Gets the stop criterion and its type
			stop_criterion,type_selection = self.get_stop_criterion(list_conts)
		
			train_set = []
			test_set = []
			cont = 0
			# Iterates over all the lot dirs
			for lot_directory in lot_dirs:

				# Verifies if the dir is valid because has the 4 types
				if self.valid_for_test(lot_directory):
					# Verifies if don't pass the stop croterion
					if cont + self.validate_stop_criterion(lot_directory,type_selection) <= stop_criterion:
						test_set.append(lot_directory)
						# Acumulates the cont with the value of the type selection
						cont += self.validate_stop_criterion(lot_directory,type_selection)

					else:
						train_set.append(lot_directory)

				else:
					train_set.append(lot_directory)
	
			train_set_path = os.path.join(self.destination_path, "train_set")	
			test_set_path = os.path.join(self.destination_path, "test_set")

			# Copies train and test lists into the folders
			self.copy_tree(train_set,train_set_path,label_directory)
			self.copy_tree(test_set,test_set_path,label_directory)
			
		print("Splitting data successfully...")

ds = dataStructuring(args.fp,args.dn)
ds.run()