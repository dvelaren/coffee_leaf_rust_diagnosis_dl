import os
import sys

from json_data_structurer import JsonDataStructurer
from re_data_structurer import ReDataStructurer
from rgb_data_structurer import RgbDataStructurer
from rgn_data_structurer import RgnDataStructurer

FILE_TYPES = ["json", "rgb", "rgn", "re"]
json_data_structurer = JsonDataStructurer()
rgb_data_structurer = RgbDataStructurer()
rgn_data_structurer = RgnDataStructurer()
re_data_structurer = ReDataStructurer()

arguments = sys.argv
if len(arguments) > 2:
    file_type = arguments[1]
    file_type_data_path = arguments[2]
    if file_type in FILE_TYPES:
        if os.path.exists(file_type_data_path):
            file_type_data_path_list = file_type_data_path.split('/')
            # Extracts the path to the penultimate directory of the given data path.
            destination_path = '/'.join(file_type_data_path_list[:len(file_type_data_path_list) - 2])
            if file_type == FILE_TYPES[0]:
                json_data_structurer.structure_data(data_path=file_type_data_path, destination_path=destination_path)
            elif file_type == FILE_TYPES[1]:
                rgb_data_structurer.structure_data(data_path=file_type_data_path, destination_path=destination_path)
            elif file_type == FILE_TYPES[2]:
                rgn_data_structurer.structure_data(data_path=file_type_data_path, destination_path=destination_path)
            else:
                re_data_structurer.structure_data(data_path=file_type_data_path, destination_path=destination_path)
        else:
            print("Try again. The given path to the data does not exist.")
    else:
        print("Try again. The given file type is not allowed. File types: {}.".format(FILE_TYPES))
else:
    print("Try again. Indicate the file type and the path to the respective data.")
