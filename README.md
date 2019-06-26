
# Coffee Leaf Rust Diagnosis - Deep Learning
This project uses the data collected on the [Coffee Leaf Rust Diagnosis project] and applies Deep Learning based on it in order to generate a model that predicts the Coffee Leaf Rust development stage. The data preprocessing, the model generation and the final evaluation occurs in an Academic Data Center.

## Table of contents
  * [Dependencies.](#dependencies)
  * [Data preparation.](#data-preparation)
  * [Components.](#components)
  * [Usage.](#usage)

## Dependencies
- [Python] 3.6.0 - Programming language.
- [Scikit-learn] 0.20.2 - Data preprocessing, cross-validation and evaluation.
- [TensorFlow] 1.12 - Model generation and persistance.
- [Keras] 2.2.4 - High level API on top of TensorFlow.

## Data preparation
The collected data is manually divided into five directories, each of them indicating a different development stage of the disease (from 0 to 4). Images and documents with no valuable content are eliminated in order to keep only the data that is significant for the model generation.

The whole data is shuffled and randomly divided into approx. 75 % for the train set and 25 % for the test set, taking into account that the directories within the train set must have at least one element per data source (JSON, RGB, RGN and RE).

The content within the train set is divided according to the file type. The documents are preprocessed so that outliers are removed and only the significant keys for the model generation remain. For its part, the images are preprocessed in order to set the pixels that do not correspond to a plant to black ("background removal"). Then, the files are loaded into memory as well as a reference to the respective labels. The resulting programming objects are normalized, structured as dictionaries and saved as _Pickle_ files (Python object serialization).

The generated _Pickle_ files are then loded into memory for the model generation. The model is composed by four sub-models, i.e. JSON sub-model, RGB sub-model, RGN sub-model and RE sub-model. The first sub-model is trained using a Multilayer Perceptron, whereas three Convolutional Neural Networks are trained to generate the remaining ones. Besides, Grid search with cross-validation is applied in general in order to find the estimators that achieve the best performance (here _F1-score_). Those estimators are saved as _.h5_ files because they are going to be used for evaluating the performance of the composite model at the end.

## Components
The tasks of every implemented component in order to achieve the processing of the collected data and evaluate the prediction of the disease's development stage is explained below.

Directory | Component (.py) | Description
| --- | --- | ---
Model | _cleanStructure_ | Eliminates the irrelevant JSON files (those with missing values and outliers) as well as the lot data directories that ended up with no content following that elimination.
Model | _dataSplitter_ | Shuffles the lot data directories within each label directory, counts the elements per file type, calculates the number of files of a specific type that should be included in the test set and, according to this number, copies a fraction of the lot data directories to another location for evaluation purposes.
Preprocessing | _createStructure_ | Divides the content of the lot data directories within the train set according to the file type.
Preprocessing | _preProcessing_ | Cleans the files by locating the plants and removing the background, in the case of images, and eliminating unnecessary keys, in the case of documents.
Model | _data_structuring_ | Produces one file per directory containing the respective data for the generation of each sub-model.
Model | _sub_models_generator_ | Generates one sub-model for each file type and saves the generated sub-models (best estimators) as _h5_ files.
Evaluation | _model_evaluator_ | Integrates the four sub-models and evaluates the composite model i.e. diagnosing the Coffee Leaf Rust development stage through it, creating a comparative table with the results and calculating the model’s performance.

## Usage
1. Make sure that the test set, the JSON _Picke_ file and the four sub-models are already created before running the composite model evaluation.
2.  Activate the virtual environment containing the necessary dependencies and execute the ``model_evaluator.py`` script by typing on the terminal:

    ```sh
    $ python3 model_evaluator.py -fp path/to/test_set/ -jpfn path/to/json_data.pickle -dp ../Preprocessing/detectors/detector_plants_v5_C150.svm smd path/to/sub_models/
    ```
    <br>
   This command loads all sub-models, iterates over the test data dividing it by the file type, gets each sub-model's prediction and, finally, calculates the Coffee Leaf Rust development stage as the rounded weighted average of them. At the end of the execution, you'll see the achieved _F1-score_ printed in the terminal and a CSV file (_results.csv_) comparing the true labels with the predictions will be also generated.

[Coffee Leaf Rust Diagnosis project]: <https://github.com/ibalejandro/coffee_leaf_rust_diagnosis/tree/develop>
[Python]: <https://www.python.org/>
[Scikit-learn]: <https://scikit-learn.org/stable/>
[TensorFlow]: <https://www.tensorflow.org/>
[Keras]: <https://keras.io/>
