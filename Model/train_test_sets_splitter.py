from time import time
import os
import shutil
import collections
import math
import random


class TrainTestSetsSplitter:
    TRAIN_PERCENTAGE = 0.75

    def __init__(self):
        pass

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
        elements_for_selection, lot_dirs_by_label_directory = self \
            .get_minimum_size_for_selection(lot_dirs_by_label_directory=lot_dirs_by_label_directory)
        lot_dirs_for_copy = self.shuffle_and_select_lot_dirs(elements_for_selection=elements_for_selection,
                                                             lot_dirs_by_label_directory=lot_dirs_by_label_directory)
        train_lot_dirs, test_lot_dirs = self \
            .split_lot_dirs_into_train_test(elements_in_lot_dirs_for_copy=elements_for_selection,
                                            lot_dirs_for_copy=lot_dirs_for_copy)
        self.create_train_test_sets(train_set_path=train_set_path, test_set_path=test_set_path,
                                    train_lot_dirs=train_lot_dirs, test_lot_dirs=test_lot_dirs)
        print("The data was successfully split and the result can be found in {}.".format(destination_path))

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
        for root, dirs, files in os.walk(data_path):
            lot_dirs_by_label_directory = dict()
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

    def get_minimum_size_for_selection(self, lot_dirs_by_label_directory):
        """
        Finds the minimum non-zero size among each of the lists of lot directories by label and removes entries, which
        contain no elements."""
        label_dirs_for_deletion = list()
        minimum_size_for_selection = math.inf
        '''
        Iterates over every label directory on the dictionary in order to find the minimum non-zero size among their
        corresponding lists of lot directories.
        '''
        for label_directory, lot_dirs in lot_dirs_by_label_directory.items():
            lot_dirs_length = len(lot_dirs)
            if lot_dirs_length is 0:
                # The current label directory contains no elements and should be deleted from the dictionary.
                label_dirs_for_deletion.append(label_directory)
            elif lot_dirs_length < minimum_size_for_selection:
                minimum_size_for_selection = lot_dirs_length
        # Deletes every label directory of the dictionary, which contains no elements.
        for label_directory_for_deletion in label_dirs_for_deletion:
            del lot_dirs_by_label_directory[label_directory_for_deletion]
        return minimum_size_for_selection, lot_dirs_by_label_directory

    def shuffle_and_select_lot_dirs(self, elements_for_selection, lot_dirs_by_label_directory):
        """Shuffles the list of lot directories by label and selects a portion of them."""
        selected_lot_dirs = dict()
        for label_directory, lot_dirs in lot_dirs_by_label_directory.items():
            random.shuffle(lot_dirs)
            # Selects the first m elements from the shuffled list.
            selected_lot_dirs[label_directory] = lot_dirs[:elements_for_selection]
        return collections.OrderedDict(sorted(selected_lot_dirs.items()))

    def split_lot_dirs_into_train_test(self, elements_in_lot_dirs_for_copy, lot_dirs_for_copy):
        """Splits the list of lot directories by label randomly into two groups for creating the train and test sets."""
        train_lot_dirs = dict()
        test_lot_dirs = dict()
        # Calculates the number of elements for train according to the defined train percentage.
        elements_for_train = int(elements_in_lot_dirs_for_copy * self.TRAIN_PERCENTAGE)
        for label_directory, lot_dirs in lot_dirs_for_copy.items():
            random.shuffle(lot_dirs)
            # Selects the first n elements from the shuffled list for the train set.
            train_lot_dirs[label_directory] = lot_dirs[:elements_for_train]
            # Selects the remaining elements from the shuffled list for the test set.
            test_lot_dirs[label_directory] = lot_dirs[elements_for_train:]
        return collections.OrderedDict(sorted(train_lot_dirs.items())), \
               collections.OrderedDict(sorted(test_lot_dirs.items()))

    def create_train_test_sets(self, train_set_path, test_set_path, train_lot_dirs, test_lot_dirs):
        """
        Copies the selected lot directories to their respective destinations in order to create the train and test sets
        while preserving the original structure of the data.
        """
        # Copies the train lot directories to the train set.
        for label_directory, lot_dirs in train_lot_dirs.items():
            train_lot_dirs_destination = os.path.join(train_set_path, label_directory)
            # Creates the respective label directory on the train set path.
            os.mkdir(train_lot_dirs_destination)
            for lot_directory in lot_dirs:
                lot_directory_path_list = lot_directory.split('/')
                # Extracts the name of the lot directory.
                lot_directory_name = lot_directory_path_list[len(lot_directory_path_list) - 1]
                lot_directory_destination = os.path.join(train_lot_dirs_destination, lot_directory_name)
                # Copies the entire lot directory to the corresponding destination and preserves its content.
                shutil.copytree(lot_directory, lot_directory_destination)
        # Copies the test lot directories to the test set.
        for label_directory, lot_dirs in test_lot_dirs.items():
            test_lot_dirs_destination = os.path.join(test_set_path, label_directory)
            # Creates the respective label directory on the test set path.
            os.mkdir(test_lot_dirs_destination)
            for lot_directory in lot_dirs:
                lot_directory_path_list = lot_directory.split('/')
                # Extracts the name of the lot directory.
                lot_directory_name = lot_directory_path_list[len(lot_directory_path_list) - 1]
                lot_directory_destination = os.path.join(test_lot_dirs_destination, lot_directory_name)
                # Copies the entire lot directory to the corresponding destination and preserves its content.
                shutil.copytree(lot_directory, lot_directory_destination)
