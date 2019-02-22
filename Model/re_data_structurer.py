import os


class ReDataStructurer:
    JPG_EXTENSION = ".JPG"

    def __init__(self):
        pass

    def structure_data(self, data_path, destination_path):
        """TODO"""
        print("Structuring re data...")
        re_data_file_path = os.path.join(destination_path, "re_data.pickle")
        print("The re data was successfully structured and the result can be found in {}.".format(re_data_file_path))

    def count_num_jpg_files(self, data_path):
        """Returns the number of jpg files inside the given data path."""
        num_files = 0
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if self.JPG_EXTENSION in file:
                    num_files += 1
        return num_files
