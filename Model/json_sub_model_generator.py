import json
import os

import numpy as np
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from six.moves import cPickle as pickle
from sklearn.model_selection import GridSearchCV, cross_validate
from tensorflow.keras.backend import clear_session


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
        json_sub_model_prop_file_path = os.path.join(self.destination_path, "json_sub_model_prop.json")
        # Removes the json sub-model files, if they already exist.
        os.remove(json_sub_model_file_path) if os.path.exists(json_sub_model_file_path) else None
        os.remove(json_sub_model_prop_file_path) if os.path.exists(json_sub_model_prop_file_path) else None
        self.load_data()
        class_weight = self.get_class_weight()
        '''
        estimator = KerasClassifier(build_fn=self.create_model, verbose=0)
        param_grid = self.get_param_grid()
        best_estimator = self.find_best_estimator(estimator=estimator, param_grid=param_grid, class_weight=class_weight)
        '''
        self.find_best_estimator_2(class_weight=class_weight, json_sub_model_file_path=json_sub_model_file_path,
                                   json_sub_model_prop_file_path=json_sub_model_prop_file_path)
        # Saves the best estimator on the given path for using it at evaluation-time.
        # best_estimator.model.save(json_sub_model_file_path)
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

    def create_model(self, kernel_initializer="glorot_uniform", activation="relu", rate=0.0, optimizer="adam"):
        """Creates and compiles the model using the given hyperparameters."""
        model = Sequential()
        model.add(Dense(units=16, kernel_initializer=kernel_initializer, input_shape=(6,)))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(Dense(units=64, kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(Dropout(rate=rate))
        model.add(Dense(units=32, kernel_initializer=kernel_initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation=activation))
        model.add(Dropout(rate=rate))
        model.add(Dense(units=4, kernel_initializer=kernel_initializer))
        model.add(Activation(activation="softmax"))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def get_param_grid(self):
        """Creates the hyperparameters grid for trying different combinations when training the model."""
        param_grid = dict()
        batch_size = [16, 32, 64]
        epochs = [5, 10, 15]
        kernel_initializer = ["glorot_uniform", "normal"]
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

    def find_best_estimator_2(self, class_weight, json_sub_model_file_path, json_sub_model_prop_file_path):
        """
        Creates an estimator for every combination of pre-set hyperparameters and evaluate them with cross-validation in
        order to find and save the estimator that achieved the best performance.
        """
        best_score = 0.0
        best_hyperparameters = dict()
        best_estimator = None
        batch_sizes = [16, 32, 64]
        epoch_list = [5, 10, 15]
        kernel_initializers = ["glorot_uniform", "normal"]
        activations = ["relu", "elu"]
        rates = [0.0, 0.1, 0.2, 0.3, 0.4]
        optimizers = ["adam"]
        total_estimators = len(batch_sizes) * len(epoch_list) * len(kernel_initializers) * len(activations) * \
                           len(rates) * len(optimizers)
        tried_estimators = 0
        for batch_size in batch_sizes:
            for epochs in epoch_list:
                for kernel_initializer in kernel_initializers:
                    for activation in activations:
                        for rate in rates:
                            for optimizer in optimizers:
                                # Creates an estimator using one combination of hyperparameters.
                                estimator = KerasClassifier(build_fn=self.create_model,
                                                            kernel_initializer=kernel_initializer,
                                                            activation=activation, rate=rate, optimizer=optimizer,
                                                            verbose=0)
                                fit_params = {"batch_size": batch_size, "epochs": epochs, "class_weight": class_weight}
                                # Fits the estimator using the feature and label data and returns the result.
                                cv_results = cross_validate(estimator=estimator, X=self.feature_data, y=self.label_data,
                                                            scoring="f1_weighted", fit_params=fit_params,
                                                            return_train_score=False, return_estimator=True)
                                # Calculates the mean performance for the current estimator.
                                test_score = np.mean(cv_results["test_score"])
                                if test_score > best_score:
                                    # Updates the best results, if necessary.
                                    best_score = test_score
                                    best_hyperparameters = {"batch_size": batch_size, "epochs": epochs,
                                                            "kernel_initializer": kernel_initializer,
                                                            "activation": activation, "rate": rate,
                                                            "optimizer": optimizer}
                                    best_estimator_index = np.argmax(cv_results["test_score"])
                                    # Selects the best estimator from the cross-validation results.
                                    best_estimator = cv_results["estimator"][best_estimator_index]
                                    # Saves the best estimator (and its properties) for using it at evaluation-time.
                                    if os.path.exists(json_sub_model_file_path):
                                        os.remove(json_sub_model_file_path)
                                    if os.path.exists(json_sub_model_prop_file_path):
                                        os.remove(json_sub_model_prop_file_path)
                                    try:
                                        properties = best_hyperparameters.copy()
                                        # Adds the score to the properties of the estimator.
                                        properties["score"] = best_score
                                        with open(json_sub_model_prop_file_path, "w") as outfile:
                                            json.dump(obj=properties, fp=outfile, indent=4)
                                    except:
                                        None
                                    best_estimator.model.save(json_sub_model_file_path)
                                # Frees GPU-memory.
                                clear_session()
                                tried_estimators += 1
                                print("Current best estimator's score: {}".format(str(best_score)))
                                print("Tried estimators = {}/{}.".format(tried_estimators, total_estimators))
        print("Best estimator's score: {}".format(str(best_score)))
        print("Hyperparameters used: {}".format(best_hyperparameters))
