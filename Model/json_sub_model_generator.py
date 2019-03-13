import os

import numpy as np
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from six.moves import cPickle as pickle
from sklearn.model_selection import GridSearchCV


class JsonSubModelGenerator:
    def __init__(self, data_path, destination_path):
        self.data_path = data_path
        self.destination_path = destination_path
        self.feature_data = None
        self.label_data = None

    def generate_sub_model(self):
        """Generates a JSON sub-model and saves it in order to use it for the posterior assessment."""
        print("Generating json sub-model...")
        json_sub_model_file_path = os.path.join(self.destination_path, "json_sub_model.h5")
        # Removes the json sub-model file, if it already exists.
        os.remove(json_sub_model_file_path) if os.path.exists(json_sub_model_file_path) else None
        self.load_data()
        class_weight = self.get_class_weight()
        estimator = KerasClassifier(build_fn=self.create_model, verbose=0)
        param_grid = self.get_param_grid()
        best_estimator = self.find_best_estimator(estimator=estimator, param_grid=param_grid, class_weight=class_weight)
        # Saves the best estimator on the given path for using it at evaluation-time.
        best_estimator.model.save(json_sub_model_file_path)
        print("The json sub-model was successfully generated and the result can be found in {}."
              .format(json_sub_model_file_path))

    def load_data(self):
        """Loads the feature and label data from the corresponding pickle file and formats it as necessary."""
        with open(self.data_path, "rb") as f:
            structured_json_data = pickle.load(f)
            self.feature_data = structured_json_data["feature_data"]
            raw_label_data = structured_json_data["label_data"]
            unique_labels = np.unique(raw_label_data).tolist()
            label_data_list = list()
            '''
            Converts all labels to their corresponding positions on the unique list so that the label data has a step
            of 1.
            '''
            for label in raw_label_data:
                label_data_list.append(unique_labels.index(label))
            self.label_data = np.array(label_data_list)
            del structured_json_data

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

    def create_model(self, kernel_initializer="normal", activation="relu", rate=0.0, optimizer="adam"):
        """Creates and compiles the model using the given hyperparameters."""
        model = Sequential()
        model.add(Dense(units=12, kernel_initializer=kernel_initializer, input_shape=(6,)))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(Dropout(rate=rate))
        model.add(Dense(units=8, kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(Dropout(rate=rate))
        model.add(Dense(units=4, kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation="softmax"))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def get_param_grid(self):
        """Creates the hyperparameters grid for trying different combinations when training the model."""
        param_grid = dict()
        batch_size = [32]
        epochs = [1]
        kernel_initializer = ["normal", "glorot_uniform"]
        activation = ["relu", "elu"]
        rate = [0.0, 0.1, 0.2, 0.3, 0.4]
        optimizer = ["adam"]
        param_grid["batch_size"] = batch_size
        param_grid["epochs"] = epochs
        param_grid["kernel_initializer"] = kernel_initializer
        param_grid["activation"] = activation
        param_grid["rate"] = rate
        param_grid["optimizer"] = optimizer
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
