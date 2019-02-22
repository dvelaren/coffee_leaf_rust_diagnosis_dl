import json
import os

import numpy as np
from six.moves import cPickle as pickle
from sklearn import preprocessing


class JsonDataStructurer:
    JSON_EXTENSION = ".json"

    def __init__(self):
        pass

    def structure_data(self, data_path, destination_path):
        """TODO"""
        print("Structuring json data...")
        json_data_file_path = os.path.join(destination_path, "json_data.pickle")
        # Removes the json data file, if it already exists.
        os.remove(json_data_file_path) if os.path.exists(json_data_file_path) else None
        num_json_files = self.count_num_json_files(data_path=data_path)
        # Creates the containers for the feature and label data.
        json_feature_data = np.ndarray(shape=(num_json_files, 6), dtype=np.float32)
        json_label_data = np.ndarray(shape=num_json_files, dtype=np.int32)
        json_feature_data_index = 0
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if self.JSON_EXTENSION in file:
                    root_list = root.split('/')
                    # Extracts the root's name, which is the label for all the files contained there.
                    label = int(root_list[len(root_list) - 1])
                    document = self.get_json_from_path(json_file_path=os.path.join(root, file))
                    json_values_as_array = self.get_json_values_as_array(document=document)
                    # Inserts the features array and the label in their respective container.
                    json_feature_data[json_feature_data_index, :] = json_values_as_array
                    json_label_data[json_feature_data_index] = label
                    # Updates the position in the containers for the next document.
                    json_feature_data_index += 1
        json_feature_data, json_label_data = self.permute_feature_label_data(feature_data=json_feature_data,
                                                                             label_data=json_label_data)
        json_data_scaler = self.generate_json_data_scaler(feature_data=json_feature_data)
        json_feature_data = self.scale_feature_data(feature_data=json_feature_data, json_data_scaler=json_data_scaler)
        self.create_structured_json_data_file(json_data_file_path=json_data_file_path, feature_data=json_feature_data,
                                              label_data=json_label_data, json_data_scaler=json_data_scaler)
        print("The json data was successfully structured and the result can be found in {}."
              .format(json_data_file_path))

    def count_num_json_files(self, data_path):
        """Returns the number of json files inside the given data path."""
        num_files = 0
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if self.JSON_EXTENSION in file:
                    num_files += 1
        return num_files

    def get_json_from_path(self, json_file_path):
        """Loads and retrieves the object in the given json file path."""
        document = dict()
        try:
            with open(json_file_path, 'r') as json_file:
                document = json.load(fp=json_file)
        except:
            # The path to the json file does not exist.
            None
        return document

    def get_json_values_as_array(self, document):
        """Returns an array appending the document's values in a specific order."""
        json_values_as_array = list()
        try:
            # Appends the values in the specified order.
            json_values_as_array.append(document["ph"])
            json_values_as_array.append(document["soil_temperature"])
            json_values_as_array.append(document["soil_moisture"])
            json_values_as_array.append(document["illuminance"])
            json_values_as_array.append(document["env_temperature"])
            json_values_as_array.append(document["env_humidity"])
        except:
            # There is a missing key in the document and an empty result is returned.
            json_values_as_array = [0] * 6
        return np.array(json_values_as_array)

    def permute_feature_label_data(self, feature_data, label_data):
        """Generates a random order and permutes the feature and label data accordingly."""
        permutation = np.random.permutation(label_data.shape[0])
        # Reorganizes the given feature data and its labels in the permutation order.
        permuted_feature_data = feature_data[permutation, :]
        permuted_label_data = label_data[permutation]
        return permuted_feature_data, permuted_label_data

    def load_json_data_scaler(self, json_data_file_path):
        """Loads the json data scaler, if it already exists. Otherwise, it returns 'None'."""
        try:
            with open(json_data_file_path, "rb") as f:
                json_data_file = pickle.load(f)
                json_data_scaler = json_data_file["json_data_scaler"]
                del json_data_file
        except:
            json_data_scaler = None
        return json_data_scaler

    def generate_json_data_scaler(self, feature_data):
        """Generates a scaler using the given feature data."""
        return preprocessing.StandardScaler().fit(feature_data)

    def scale_feature_data(self, feature_data, json_data_scaler):
        """Scales the given feature data using the given json data scaler and returns the result."""
        return json_data_scaler.transform(feature_data)

    def create_structured_json_data_file(self, json_data_file_path, feature_data, label_data, json_data_scaler):
        """Creates a single file containing the structured json data as well as its corresponding scaler."""
        try:
            f = open(json_data_file_path, "wb")
            structured_json_data = {
                "feature_data": feature_data,
                "label_data": label_data,
                "json_data_scaler": json_data_scaler
            }
            pickle.dump(structured_json_data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except:
            print("Unable to save the structured json data to {}.".format(json_data_file_path))
            raise
