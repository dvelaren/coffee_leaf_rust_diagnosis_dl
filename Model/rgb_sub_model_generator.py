import os

import numpy as np
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from six.moves import cPickle as pickle
from sklearn.model_selection import GridSearchCV


class RgbSubModelGenerator:
    FRAME_HEIGHT = 96
    FRAME_WIDTH = 128

    def __init__(self, data_path, destination_path):
        self.data_path = data_path
        self.destination_path = destination_path
        self.feature_data = None
        self.label_data = None

    def generate_sub_model(self):
        """Generates a RGB sub-model and saves it in order to use it for the posterior assessment."""
        print("Generating rgb sub-model...")
        rgb_sub_model_file_path = os.path.join(self.destination_path, "rgb_sub_model.h5")
        # Removes the rgb sub-model file, if it already exists.
        os.remove(rgb_sub_model_file_path) if os.path.exists(rgb_sub_model_file_path) else None
        self.load_data()
        class_weight = self.get_class_weight()
        estimator = KerasClassifier(build_fn=self.create_model, verbose=0)
        param_grid = self.get_param_grid()
        best_estimator = self.find_best_estimator(estimator=estimator, param_grid=param_grid, class_weight=class_weight)
        # Saves the best estimator on the given path for using it at evaluation-time.
        best_estimator.model.save(rgb_sub_model_file_path)
        print("The rgb sub-model was successfully generated and the result can be found in {}."
              .format(rgb_sub_model_file_path))

    def load_data(self):
        """Loads the feature and label data from the corresponding pickle file."""
        with open(self.data_path, "rb") as f:
            structured_rgb_data = pickle.load(f)
            self.feature_data = structured_rgb_data["feature_data"]
            self.label_data = structured_rgb_data["label_data"]
            del structured_rgb_data

    def get_class_weight(self):
        """
        Calculates and retrieves the weights for each class on the label data for taking the imbalance into account.
        """
        class_weight = dict()
        unique, counts = np.unique(self.label_data, return_counts=True)
        class_occurrences = dict(zip(unique, counts))
        max_occurrences = 0
        for label, occurrences in class_occurrences.items():
            max_occurrences = occurrences if occurrences > max_occurrences else max_occurrences
        for label, occurrences in class_occurrences.items():
            class_weight[label] = float(max_occurrences / occurrences)
        return class_weight

    def create_model(self, kernel_size=(3, 3), kernel_initializer="normal", activation="relu", pool_size=(2, 2),
                     rate=0.0, optimizer="adam"):
        """Creates and compiles the model using the given hyperparameters."""
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                         input_shape=(self.FRAME_HEIGHT, self.FRAME_WIDTH, 3)))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(Conv2D(filters=64, kernel_size=kernel_size, kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(rate=rate))
        model.add(Flatten())
        model.add(Dense(units=128, kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(Dropout(rate=rate))
        model.add(Dense(units=5, kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation="softmax"))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def get_param_grid(self):
        """Creates the hyperparameters grid for trying different combinations when training the model."""
        param_grid = dict()
        batch_size = [16, 32, 64, 128]
        epochs = [20]
        kernel_size = [(3, 3), (5, 5)]
        kernel_initializer = ["uniform", "lecun_uniform", "normal", "zero", "glorot_normal", "glorot_uniform",
                              "he_normal", "he_uniform"]
        activation = ["softmax", "softplus", "softsign", "relu", "elu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
        pool_size = [(2, 2)]
        rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # optimizer = ["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"]
        param_grid["batch_size"] = batch_size
        param_grid["epochs"] = epochs
        param_grid["kernel_size"] = kernel_size
        # param_grid["kernel_initializer"] = kernel_initializer
        # param_grid["activation"] = activation
        # param_grid["pool_size"] = pool_size
        # param_grid["rate"] = rate
        # param_grid["optimizer"] = optimizer
        return param_grid

    def find_best_estimator(self, estimator, param_grid, class_weight):
        """
        Executes every combination of the given hyperparameters grid on the given estimator and returns the estimator
        that achieved the best performance.
        """
        grid_search_cv = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring="f1_weighted", n_jobs=1)
        grid_search_cv_result = grid_search_cv.fit(X=self.feature_data, y=self.label_data, class_weight=class_weight)
        print("Best estimator's score: {}".format(str(grid_search_cv_result.best_score_)))
        print("Hyperparameters used: {}".format(grid_search_cv_result.best_params_))
        return grid_search_cv_result.best_estimator_
