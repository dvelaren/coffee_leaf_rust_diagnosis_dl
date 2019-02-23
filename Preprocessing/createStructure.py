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
parser.add_argument("-v", "--verbose", help="Mostrar informacion de depuracion", action="store_true")
parser.add_argument("-fp", "--fp", help="Carpeta a procesar")

args = parser.parse_args()


class createStructure:
	global file, folderPath, detector, tempImage, globalPath
	def __init__(self,folderPath):
		
		self.folderPath = folderPath
		self.globalPath = folderPath
		#self.absolutePath = absolutePath
		self.tempImage = None


	def createSubFolders(self):
		types = ["json","rgb","re","rgn"]
		file_path = self.folderPath[:-1]
		file_path = file_path[:file_path.rindex('/')]+'/'
		self.globalPath = file_path
		for i in types:
			path = str(file_path)+str(i)+"_data"
			if not os.path.exists(path):
				os.mkdir(path)
				
	def findTypes(self,folder):

		json = []
		rgb = []
		re = []
		rgn = []
		# this list has the values which didn't match with any type
		other = []
		cont=0
		#find the files in the whole path
		for root, dirs, files in os.walk(folder):
			for file in files:
				line = str(os.path.join(root, file))
				cont+=1
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

		print("cont4: "+str(cont))

		#print(json)
		
		return json,rgb,re,rgn, other


	def sendData(self,list_type,typestr):
	
		cont = 0
		changeCount = 0


		for i in list_type:

			#Erase the begginig of the path
			#cfpath = i.replace("../model_preprocessing/data","")
			cfpath = i[i.find('data')+len('data'):]
			#Find the last '/'
			line_split = cfpath.rindex('/')
			#use the last '/' to erase the namefile
			cfpath = cfpath[:line_split+1]

			#update the path with the folder and the rest of the path
			save_file = self.globalPath+typestr+"/"+cfpath
			

			#Erase a '/' to find the new last index of the '/'
			save_file = save_file[:len(save_file)-1]
			
			#print("sin/"+save_file)
			#Find the last '/'
			line_split = save_file.rindex('/')

			#Erase the folder lot*
			save_file = save_file[:line_split+1]
			#print(save_file)
			#print("sin/pal"+save_file)

			# Extract the number of the illness to restart the counter
			# if changeCount != i[28] and typestr != "rgb_data":
			# 	cont=0

			#Update the new number of the illness
			#changeCount = i[28]
			#print(typestr+str(changeCount))

			# If the type is rgb its nececesarry to delete other folder of the path
			if typestr == "rgb_data":
				last_index = i.rindex('/')
				place = i[last_index-1:last_index]
				
				#print(place)
				#print(i)

				save_file = save_file[:save_file.find("lot")]
				save_file = save_file[:-2]
				#Save the new path with the new place of the rgb specific illness 
				save_file = save_file +place+"/"
				
				
				#print(save_file)

			#Create the folder
			if not os.path.exists(save_file):
				os.makedirs(save_file)

			#copy the file
			shutil.copy(i, save_file)

			file = i[i.rindex('/'):]
			ext = ""

			if typestr == "json_data":
				ext = ".json"
			else:
				ext = ".jpg"

			#print(typestr+str(cont))
			#print(save_file+str(cont)+ext)
			#print(save_file)
			os.rename(save_file+file, save_file+str(cont)+ext)
			cont+=1
		


	def extractFiles(self):
		#path = os.listdir(self.folderPath)

		# Create the four new subfolders
		self.createSubFolders()

		json,rgb,re,rgn, idk = self.findTypes(self.folderPath)

		types = [json,rgb,re,rgn]
		string_types = ["json_data","rgb_data","re_data","rgn_data"]
		
		# #Print the total values of the set
		# #print(len(json)+len(rgb)+len(re)+len(rgn)+len(idk))
		# #Print the values which doesn't match with no type
		# #print(idk)

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
		cont = 0
		number = 0
		contentro= 0
		path = self.globalPath+'rgb_data/'
		for root, dirs, files in os.walk(path):
			for file in files:
				line = str(os.path.join(root, file))
				
				if line.find("rgb_") != -1:
					
					findNumber = line.rindex("data")
					
					#print(line[findNumber+5:findNumber+6])
					if number != line[findNumber+5:findNumber+6]:
						cont = 0
						contentro+=1

					number = line[findNumber+5:findNumber+6]

					line2 = line[:line.rindex('/')]
					if firsTime:
						newfile = line2+"/"+str(cont)+".jpg"
					else:
						newfile = line2+"/"+"0"+str(cont)+".jpg"
					#print(line)
					#print(newfile)
					os.rename(line, newfile )
					cont+=1

	def renameReFiles(self,firsTime=False):
		cont = 0
		number = 0
		contentro= 0
		path = self.globalPath+'re_data/'
		for root, dirs, files in os.walk(path):
			for file in files:
				line = str(os.path.join(root, file))
				
				if line.find("re_") != -1:
					
					findNumber = line.rindex("data")
					
					#print(line[findNumber+5:findNumber+6])
					if number != line[findNumber+5:findNumber+6]:
						cont = 0
						contentro+=1

					number = line[findNumber+5:findNumber+6]

					line2 = line[:line.rindex('/')]
					if firsTime:
						newfile = line2+"/"+str(cont)+".jpg"
					else:
						newfile = line2+"/"+"0"+str(cont)+".jpg"
					#print(line)
					#print(newfile)
					os.rename(line, newfile )
					cont+=1


	def renameRgnFiles(self,firsTime=False):
		cont = 0
		number = 0
		contentro= 0
		path = self.globalPath+'rgn_data/'
		for root, dirs, files in os.walk(path):
			for file in files:
				line = str(os.path.join(root, file))
				
				if line.find("rgn_") != -1:
					
					findNumber = line.rindex("data")
					
					#print(line[findNumber+5:findNumber+6])
					if number != line[findNumber+5:findNumber+6]:
						cont = 0
						contentro+=1

					number = line[findNumber+5:findNumber+6]

					line2 = line[:line.rindex('/')]
					if firsTime:
						newfile = line2+"/"+str(cont)+".jpg"
					else:
						newfile = line2+"/"+"0"+str(cont)+".jpg"
					#print(line)
					#print(newfile)
					os.rename(line, newfile )
					cont+=1

	def renameJsonFiles(self,firsTime=False):
		cont = 0
		number = 0
		contentro= 0
		path = self.globalPath+'json_data/'
		for root, dirs, files in os.walk(path):
			for file in files:
				line = str(os.path.join(root, file))
				
				if line.find("json_") != -1:
					
					findNumber = line.rindex("data")
					
					#print(line[findNumber+5:findNumber+6])
					if number != line[findNumber+5:findNumber+6]:
						cont = 0
						contentro+=1

					number = line[findNumber+5:findNumber+6]

					line2 = line[:line.rindex('/')]
					if firsTime:
						newfile = line2+"/"+str(cont)+".json"
					else:
						newfile = line2+"/"+"0"+str(cont)+".json"
					#print(line)
					#print(newfile)
					os.rename(line, newfile )
					cont+=1

#4880



p = createStructure(args.fp)
p.extractFiles()
