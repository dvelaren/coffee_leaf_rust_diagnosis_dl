import os
import argparse
import shutil
import json
import glob


parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--fp", help="Folder where is the data")
args = parser.parse_args()


class preprocessData:

	global file, folderPath
	def __init__(self,folderPath):
		
		self.folderPath = folderPath
		
	def findRgbFolder(self,list_elements):
		"""
			Finds if there is a rgb folder
		"""

		for l in list_elements:
			if l.find("rgb_images") != -1:
				return True

		return False

	def amountOfFiles(self, data_path):
		"""
		 	Gets the total number of files into the folder path
		"""
		cont = 0
		for root, dirs, files in os.walk(data_path):
			for file in files:
				cont +=1

		return cont
		
	def findJson(self, elements):

		"""
			Finds which element is the json file
		"""
		jsonFile = ""
		for e in elements:
			if e.find(".json") != -1:
				jsonFile = e


		return jsonFile

	def findTypes(self,folder):

		"""
			Search in the given folder for the four types and classify them into a differents lists
		"""
		print("Cleaning folder...")
		lot_dirs = list()
		for root, dirs, files in os.walk(folder):
			
			# Iterates over the first layer of directories.
			for directory in dirs:
				directory_path = os.path.join(root, directory)	
				# Iterates over each element within the directory path.
				for lot_directory in os.listdir(directory_path):
					lot_directory_path = os.path.join(directory_path, lot_directory)
					# Appends the current element to the list if it is a directory.
					lot_dirs.append(lot_directory_path) if os.path.isdir(lot_directory_path) else None
			# Only the first layer of directories is required. It is not necessary to keep walking through the tree.
			break

		# Iterates over all lot folders
		for lot in lot_dirs:
			# Gets the elements of the lot folder
			elements = glob.glob(lot+"/*")
			
			# If there is not a rgb folder inside, remove the lot folder
			if self.findRgbFolder(elements):
				
				# Gets the json file 
				jsonFile = self.findJson(elements)

				# Cleans the json
				document = self.cleanJson(jsonFile)
				
				# Checks if Json is ok
				if len(document) == 0:

					if jsonFile != "":
						os.remove(jsonFile)

					# Gets how many files are into the lot folder
					amountOfFiles = self.amountOfFiles(lot)

					# If there are not files
					if amountOfFiles == 0:
						shutil.rmtree(lot)

			else: 
				shutil.rmtree(lot)

		print("Folder cleaned")
				
			
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
						'soil_temperature' : s_temp,
						'soil_moisture': s_moist,
						'illuminance' : illuminance,
						'env_temperature' : env_temp,
						'env_humidity' : env_humi
						} 
					
				else:		
			
					document = {}

				return document
				
		
		except:
			 return {}

			
p = preprocessData(args.fp)
p.findTypes(args.fp)





