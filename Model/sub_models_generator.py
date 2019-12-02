import os
import sys

from json_sub_model_generator import JsonSubModelGenerator
from rgb_sub_model_generator import RgbSubModelGenerator
from rgn_sub_model_generator import RgnSubModelGenerator
from re_sub_model_generator import ReSubModelGenerator

SUB_MODEL_TYPES = ["json", "rgb", "rgn", "re"]

arguments = sys.argv
if len(arguments) > 3:
    sub_model_type = arguments[1]
    input_data_path = arguments[2]
    sub_model_destination_path = arguments[3]
    if sub_model_type in SUB_MODEL_TYPES:
        if os.path.exists(input_data_path):
            if os.path.exists(sub_model_destination_path):
                if sub_model_type == SUB_MODEL_TYPES[0]:
                    json_sub_model_generator = JsonSubModelGenerator(data_path=input_data_path,
                                                                     destination_path=sub_model_destination_path)
                    json_sub_model_generator.generate_sub_model()
                elif sub_model_type == SUB_MODEL_TYPES[1]:
                    rgb_sub_model_generator = RgbSubModelGenerator(data_path=input_data_path,
                                                                   destination_path=sub_model_destination_path)
                    rgb_sub_model_generator.generate_sub_model()
                elif sub_model_type == SUB_MODEL_TYPES[2]:
                    rgn_sub_model_generator = RgnSubModelGenerator(data_path=input_data_path,
                                                                   destination_path=sub_model_destination_path)
                    rgn_sub_model_generator.generate_sub_model()
                else:
                    re_sub_model_generator = ReSubModelGenerator(data_path=input_data_path,
                                                                 destination_path=sub_model_destination_path)
                    re_sub_model_generator.generate_sub_model()
            else:
                print("Try again. The given destination path for the generated sub-model does not exist.")
        else:
            print("Try again. The given path to the input data does not exist.")
    else:
        print("Try again. The given sub-model type is not allowed. Sub-model types: {}.".format(SUB_MODEL_TYPES))
else:
    print("Try again. Indicate the sub-model type, the path to the respective input data and the destination path for "
          + "the sub-model.")
