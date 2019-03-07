# Prepocessing code

This part of project is focused on tries to make a cleaning of the diferent types of data(re,rgn,rgb and json) using image processing and machine learning algorithms, with the final intention to pass them to a deep learning model and get better accuracy and behavior.


# Sources Files:

1. createStructure (It is into Model folder):

This code creates the basic structure for the folder types. 

Usage:

 - python3.6 createStructure.py -fp --folder where is the data, where are the coffee leaf rust development stage folders 

 Output: This code create 4 folders name "(type)_data" into -fp folder path 

Functions table:

                                     createStructure.py
|Function       |     Input parameters    |                 Return values                       |
|---------------|--------------------------|-----------------------------------------------------|
|findTypes   |     folder(Str)       | re(list),rgn(list),rgb(list),json(list),other(list)		 |
|sendData    |   list_type(list) ,typestr |                                                   |


2. rustCode:

This code is focused on cleaning of the all types of files. 

Usage:

 - python3.6 rustCode.py -fp --folder where is the four target folders 

 Output: This code overwrite the files of the four target folders replacing the old files with new cleaning files.

Functions table:


                                     rustCode.py
|Function       |     Input parameters    |                 Return values                       |
|---------------|--------------------------|-----------------------------------------------------|
|findTypes4Clean   |     folder(Str)       | re(list),rgn(list),rgb(list),json(list)		   	 |
|getGraphicHistogram | frame(numpy.ndarray), concatenate(Bool)(Optional) | frame(numpy.ndarray)  |
|saveVideoAnalyse   |     list(Str)       |                                                    |
|analysePlant   | img(numpy.ndarray), debug(Bool)(default=False)|    valueResponse(Bool)         |
|cleanRe   |     frame(numpy.ndarray)       | frame(numpy.ndarray)		   	 					 |
|cleanRgn   |     frame(numpy.ndarray)       | frame(numpy.ndarray)		   	 					 |
|cleanRgb   |     frame(numpy.ndarray)       | frame(numpy.ndarray)		   	 					 |
|cleanJson  | json_path(str)              | document{}(dictionary))   	 		                 |
|cleanFiles | folder_path(str)		  	  |                                                      |

The explanation of each function can be find inside of the source code.


3. xml_creator_img_rust:

This code creates a xml file with all bounding boxes of plants into an image. This xml will be the input of the dlib plants classifier

Usage:

- python3.6 xml_creator_img_rust.py -fp --Folder with images

The output is the xml file which can be opened with an ethernet navigator.



4. cleanStructure (It is into Model folder):

This code is the first part of the pipeline and it cleans all the data folder about json files problems and empty rgb folders into the lots

Usage:

- python3.6 cleanStructure.py -fp --Folder where the data

Output: data folder without bad json files and empty rgb folder. 


Functions table:


                                     cleanStructure.py
|Function       |     Input parameters    |                 Return values                        |
|---------------|--------------------------|-----------------------------------------------------|
|findTypes   |     folder(Str)       | 	   	 													 |
|cleanJson  | json_path(str)              | document{}(dictionary))   	 		                 |
|findJson  | elements[](str)              | jsonFile(str)   	 		                 |
|amountOfFiles  | data_path(str)              | cont(int)   	 		                 |
|findRgbFolder  | elements[](str)              | band(bool)   	 		                 |

The explanation of each function can be find inside of the source code.


5. dataStructuring(It is into Model folder)
		

This code splits the data into train_set and test_set through a stop criterion where test is representated by the 25% of the minimum of the amount of the any type (json,rgb,re,rgn)

Usage:

- python3.6 dataStructuring.py -fp --Folder where is the data -fd --The name of the new folder where is going to be the data


Output: New folder which contains two folders, train_set and test_set 


Functions table:


                                     dataStructuring.py
|Function       |     Input parameters    |                 Return values                        |
|---------------|--------------------------|-----------------------------------------------------|
|run -> main method   |          | 	   	 													 |
|validate_stop_criterion  | lot_directory(str), type_selection(str)              | cont_type(int)   	 		                 |
|valid_for_test  | lot_directory(str)              | band(bool)   	 		                 |
|get_stop_criterion  | list_conts[](str)              | stop_criterion(int),type_selection(str)   	 		                 |
|elements_numbers_for_type  | data_path(str)              | cont_json(int), cont_rgb(int), cont_rgn(int), cont_re(int)   	 		                 |

The explanation of each function can be find inside of the source code.




