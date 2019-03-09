import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from six.moves import cPickle as pickle


class ReSubModelGenerator:
    def __init__(self, data_path, destination_path):
        self.data_path = data_path
        self.destination_path = destination_path

    def generate_sub_model(self):
        pass
