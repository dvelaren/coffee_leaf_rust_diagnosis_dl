from train_test_sets_splitter import TrainTestSetsSplitter
import sys
import os

train_test_splitter = TrainTestSetsSplitter()

arguments = sys.argv
if len(arguments) > 1:
    data_path = arguments[1]
    destination_name = arguments[2]
    if os.path.exists(data_path):
        data_path_list = data_path.split('/')
        # Extracts the path to the penultimate directory of the given data path.
        partial_destination_path = '/'.join(data_path_list[0:len(data_path_list) - 2])
        # Determines the new destination path for storing the result of the split.
        destination_path = os.path.join(partial_destination_path, destination_name)
        train_test_splitter.split_data_into_train_test(data_path=data_path, destination_path=destination_path)
    else:
        print("Try again. The given path to the data does not exist.")
else:
    print("Try again. Indicate the path to the data to be split and the directory's name of the destination.")
