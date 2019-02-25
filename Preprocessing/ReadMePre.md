# Prepocessing code

This part of project is focused on tries to make a cleaning of the diferent types of data(re,rgn,rgb and json) using image processing and machine learning algorithms, with the final intention to pass them to a deep learning model and get better accuracy and behavior.

# Sources Files:

1. createStructure:

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
|saveVideoAnalyse   |     list[](Str)       |                                                    |
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






		

 	



