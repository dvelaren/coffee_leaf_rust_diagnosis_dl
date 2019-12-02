import os

import cv2
import imutils
import numpy as np
from six.moves import cPickle as pickle


class RgnDataStructurer:
    JPG_EXTENSION = ".JPG"
    FRAME_HEIGHT = 128
    FRAME_WIDTH = 96

    def __init__(self):
        pass

    def structure_data(self, data_path, destination_path):
        """Structures the rgn data into feature and label data and saves the result into a single file."""
        print("Structuring rgn data...")
        rgn_data_file_path = os.path.join(destination_path, "rgn_data.pickle")
        # Removes the rgn data file, if it already exists.
        os.remove(rgn_data_file_path) if os.path.exists(rgn_data_file_path) else None
        num_rgn_files = self.count_num_jpg_files(data_path=data_path)
        # Creates the containers for the feature and label data.
        rgn_feature_data = np.ndarray(shape=(num_rgn_files, self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.float32)
        rgn_label_data = np.ndarray(shape=num_rgn_files, dtype=np.int32)
        rgn_feature_data_index = 0
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if self.JPG_EXTENSION in file:
                    root_list = root.split('/')
                    # Extracts the root's name, which is the label for all the files contained there.
                    label = int(root_list[len(root_list) - 1])
                    image = self.get_image_from_path(image_file_path=os.path.join(root, file))
                    image_as_nd_array = self.resize_image(image=image)
                    # Inserts the features array and the label in their respective container.
                    rgn_feature_data[rgn_feature_data_index, :, :, :] = image_as_nd_array
                    rgn_label_data[rgn_feature_data_index] = label
                    # Updates the position in the containers for the next image.
                    rgn_feature_data_index += 1
        rgn_feature_data, rgn_label_data = self.permute_feature_label_data(feature_data=rgn_feature_data,
                                                                           label_data=rgn_label_data)
        rgn_feature_data = self.scale_feature_data(feature_data=rgn_feature_data)
        self.create_structured_rgn_data_file(rgn_data_file_path=rgn_data_file_path, feature_data=rgn_feature_data,
                                             label_data=rgn_label_data)
        print("The rgn data was successfully structured and the result can be found in {}.".format(rgn_data_file_path))

    def count_num_jpg_files(self, data_path):
        """Returns the number of jpg files inside the given data path."""
        num_files = 0
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if self.JPG_EXTENSION in file:
                    num_files += 1
        return num_files

    def get_image_from_path(self, image_file_path):
        """Loads and retrieves the object in the given image file path."""
        image = np.full(shape=(self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), fill_value=0, dtype=np.float32)
        try:
            image = cv2.imread(image_file_path)
        except:
            None
        return image

    def resize_image(self, image):
        """Resizes the image according to the global frame height and returns it."""
        return imutils.resize(image, height=self.FRAME_HEIGHT)

    def permute_feature_label_data(self, feature_data, label_data):
        """Generates a random order and permutes the feature and label data accordingly."""
        permutation = np.random.permutation(label_data.shape[0])
        # Reorganizes the given feature data and its labels in the permutation order.
        permuted_feature_data = feature_data[permutation, :, :, :]
        permuted_label_data = label_data[permutation]
        return permuted_feature_data, permuted_label_data

    def scale_feature_data(self, feature_data):
        """Scales the given feature data and returns the result."""
        return (feature_data / 127.5) - 1.0

    def create_structured_rgn_data_file(self, rgn_data_file_path, feature_data, label_data):
        """Creates a single file containing the structured rgn data."""
        try:
            f = open(rgn_data_file_path, "wb")
            structured_rgn_data = {
                "feature_data": feature_data,
                "label_data": label_data
            }
            pickle.dump(structured_rgn_data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except:
            print("Unable to save the structured rgn data to {}.".format(rgn_data_file_path))
            raise
