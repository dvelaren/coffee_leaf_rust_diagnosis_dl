# Prepocessing code

This part of project is focused on the evaluation and prediction of a test set using models trained on tensorflow keras.


# Sources Files:

1. model_evaluator:

This code gets the predictions results of the trained models and gets the final stat. 

Usage:
 

 - python3.6 model_evaluator.py 

 arguments:
 
 fp --Folder where are the test set data

 jpfn --Directory where is the json pickle file

 dp --Directory where is the predictor (The predictor is on GitHub, into Preprocessing/detectors and the final predictor that must be used is detector_plants_v5_C150.svm )

 smd --Directory where is the folder which contains the submodels

 Example: python3.6 model_evaluator.py -fp ../../../test_predictions/ -jpfn ../../../results/json_data.pickle -dp ../Preprocessing/detectors/detector_plants_v5_C150.svm smd ../../../sub_models/

 
 Output: This code generates a csv file which will contains the results of the investigation and shows the f1 score in the standar output

Functions table:

                                     createStructure.py
|Function       |     Input parameters    |                 Return values                        |
|---------------|--------------------------|-----------------------------------------------------|
|findTypes      |     folder(Str)       | re(list),rgn(list),rgb(list),json(list),other(list)    |
|get_json_predict       |   data_path(str)  |        prediction_index(int)                       |
|get_rgb_predict       |   list_data_path(list)  |        prediction_index(int)                       |
|get_re_predict       |   data_path(str)  |        prediction_index(int)                       |
|get_rgn_predict       |   data_path(str)  |        prediction_index(int)                       |
|round_number       |   number(float)  |        rounded_number(int)                       |
|run(Main Function)       |    |  |

