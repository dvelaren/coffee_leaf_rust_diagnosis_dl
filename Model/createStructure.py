import os
import argparse
import shutil
import json


parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--fp", help="folder where is the data, where are the coffee leaf rust development stage folders")
args = parser.parse_args()


class createStructure:

	global file, folderPath, detector, globalPath

	def __init__(self,folderPath):	
		self.folderPath = folderPath
		self.globalPath = folderPath
		
	def createSubFolders(self):
		"""
			create the four folders into the file path
		"""

		types = ["json","rgb","re","rgn"]
		file_path = self.folderPath[:-1]
		file_path = file_path[:file_path.rindex('/')]+'/'
		self.globalPath = file_path
		for i in types:
			path = str(file_path)+str(i)+"_data"
			if not os.path.exists(path):
				os.mkdir(path)
				
	def findTypes(self,folder):

		"""
			Search in the given folder for the four types and classify them into a differents lists
		"""

		json = []
		rgb = []
		re = []
		rgn = []
		# this list has the values which didn't match with any type
		other = []
		#find the files in the whole path
		for root, dirs, files in os.walk(folder):
			for file in files:
				line = str(os.path.join(root, file))
				
				if line.find("rgb") != -1:
					rgb.append(line)

				elif line.find("re.") != -1:
					re.append(line)

				elif line.find("rgn.") != -1:
					rgn.append(line)

				elif line.find(".json") != -1:
					json.append(line)

				else:
					other.append(line)
		
		return json,rgb,re,rgn, other


	def sendData(self,list_type,typestr):
	
		"""
			Send the new data to the new folder for eact type
		"""

		print("Creating {} folder...".format(typestr))
		cont = 0
		changeCount = 0

		for i in list_type:
			# Replace whole file path with the part of the file path where the lot is 
			cfpath = i.replace(self.folderPath,'')
			# Find the last '/'
			line_split = cfpath.rindex('/')
			# Use the last '/' to erase the namefile
			cfpath = cfpath[:line_split+1]

			# Update the path with the folder and the rest of the path
			save_file = self.globalPath+typestr+"/"+cfpath
			

			# Erase a '/' to find the new last index of the '/'
			save_file = save_file[:len(save_file)-1]
			
			# Find the last '/'
			line_split = save_file.rindex('/')

			# Erase the folder lot*
			save_file = save_file[:line_split+1]

			# If the type is rgb its nececesarry to delete other folder of the path
			if typestr == "rgb_data":
				last_index = i.rindex('/')
				place = i[last_index-1:last_index]
				
				save_file = save_file[:save_file.find("lot")]
				save_file = save_file[:-2]

				# Save the new path with the new place of the rgb specific illness 
				save_file = save_file +place+"/"

			# Create the folder
			if not os.path.exists(save_file):
				os.makedirs(save_file)

			# Copy the file
			shutil.copy(i, save_file)

			file = i[i.rindex('/'):]
			ext = ""

			if typestr == "json_data":
				ext = ".json"
			else:
				ext = ".JPG"

			# rename the file
			os.rename(save_file+file, save_file+str(cont)+ext)
			cont+=1
		


	def extractFiles(self):
		

		# Create the four new subfolders
		self.createSubFolders()

		json,rgb,re,rgn, idk = self.findTypes(self.folderPath)

		types = [json,rgb,re,rgn]
		string_types = ["json_data","rgb_data","re_data","rgn_data"]
		

		for st,t in zip(string_types,types):
			self.sendData(t,st)

		# # Rename all the rgb files
		p.renameRgbFiles()
		p.renameRgbFiles(True)

		# # Rename all the re files
		p.renameReFiles()
		p.renameReFiles(True)

		# # Rename all the rgn files
		p.renameRgnFiles()
		p.renameRgnFiles(True)

		# # Rename all the json files
		p.renameJsonFiles()
		p.renameJsonFiles(True)



	def renameRgbFiles(self,firsTime=False):
		"""
			This function is created for rename all the Rgb files
		"""

		cont = 0
		number = 0
		contentro= 0
		path = self.globalPath+'rgb_data/'
		for root, dirs, files in os.walk(path):
			for file in files:
				line = str(os.path.join(root, file))
				
				if line.find("rgb_") != -1:
					
					findNumber = line.rindex("data")
					if number != line[findNumber+5:findNumber+6]:
						cont = 0
						contentro+=1

					number = line[findNumber+5:findNumber+6]

					line2 = line[:line.rindex('/')]
					if firsTime:
						newfile = line2+"/"+str(cont)+".JPG"
					else:
						newfile = line2+"/"+"0"+str(cont)+".JPG"
					
					os.rename(line, newfile )
					cont+=1

	def renameReFiles(self,firsTime=False):
		"""
			This function is created for rename all the Re files
		"""

		cont = 0
		number = 0
		contentro= 0
		path = self.globalPath+'re_data/'
		for root, dirs, files in os.walk(path):
			for file in files:
				line = str(os.path.join(root, file))
				
				if line.find("re_") != -1:
					
					findNumber = line.rindex("data")
					
					if number != line[findNumber+5:findNumber+6]:
						cont = 0
						contentro+=1

					number = line[findNumber+5:findNumber+6]

					line2 = line[:line.rindex('/')]
					if firsTime:
						newfile = line2+"/"+str(cont)+".JPG"
					else:
						newfile = line2+"/"+"0"+str(cont)+".JPG"

					os.rename(line, newfile )
					cont+=1


	def renameRgnFiles(self,firsTime=False):
		"""
			This function is created for rename all the Rgn files
		"""

		cont = 0
		number = 0
		contentro= 0
		path = self.globalPath+'rgn_data/'
		for root, dirs, files in os.walk(path):
			for file in files:
				line = str(os.path.join(root, file))
				
				if line.find("rgn_") != -1:
					
					findNumber = line.rindex("data")
		
					if number != line[findNumber+5:findNumber+6]:
						cont = 0
						contentro+=1

					number = line[findNumber+5:findNumber+6]

					line2 = line[:line.rindex('/')]
					if firsTime:
						newfile = line2+"/"+str(cont)+".JPG"
					else:
						newfile = line2+"/"+"0"+str(cont)+".JPG"
					
					os.rename(line, newfile )
					cont+=1

	def renameJsonFiles(self,firsTime=False):

		"""
			This function is created for rename all the Rgb files
		"""

		cont = 0
		number = 0
		contentro= 0
		path = self.globalPath+'json_data/'
		for root, dirs, files in os.walk(path):
			for file in files:
				line = str(os.path.join(root, file))
				
				if line.find("json_") != -1:
					
					findNumber = line.rindex("data")
					
					if number != line[findNumber+5:findNumber+6]:
						cont = 0
						contentro+=1

					number = line[findNumber+5:findNumber+6]

					line2 = line[:line.rindex('/')]
					if firsTime:
						newfile = line2+"/"+str(cont)+".json"
					else:
						newfile = line2+"/"+"0"+str(cont)+".json"
					
					os.rename(line, newfile )
					cont+=1


p = createStructure(args.fp)
p.extractFiles()
