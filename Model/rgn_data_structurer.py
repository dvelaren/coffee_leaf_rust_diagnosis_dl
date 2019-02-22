import os


class RgnDataStructurer:
    JPG_EXTENSION = ".JPG"

    def __init__(self):
        pass

    def structure_data(self, data_path, destination_path):
        """TODO"""
        print("Structuring rgn data...")
        rgn_data_file_path = os.path.join(destination_path, "rgn_data.pickle")
        print("The rgn data was successfully structured and the result can be found in {}.".format(rgn_data_file_path))

    def count_num_jpg_files(self, data_path):
        """Returns the number of jpg files inside the given data path."""
        num_files = 0
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if self.JPG_EXTENSION in file:
                    num_files += 1
        return num_files
