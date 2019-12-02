import cv2
import os
import argparse
from multiprocessing import Pool
import multiprocessing
import shutil
import numpy as np
import imutils
import dlib
import json

from preProcessing import preprocessData
parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--fp", help="Folder where is the four target folders")
parser.add_argument("-dp", "--dp", help="detector file")
args = parser.parse_args()

p = preprocessData(args.fp, args.dp)
p.cleanFiles(args.fp)





