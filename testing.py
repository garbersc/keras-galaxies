import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
import load_data
import realtime_augmentation as ra
import time
import sys
import json
from custom_for_keras import input_generator
from datetime import datetime, timedelta
import csv
import os
import cPickle as pickle
import matplotlib.pyplot as plt
from termcolor import colored

from custom_keras_model_and_fit_capsels import kaggle_winsol

starting_time = time.time()

copy_to_ram_beforehand = False

debug = True

get_winsol_weights = False

BATCH_SIZE = 256  # keep in mind

NUM_INPUT_FEATURES = 3

TRAIN_LOSS_SF_PATH = "trainingNmbrs_keras_modular_includeFlip_and_37relu.txt"
# TARGET_PATH = "predictions/final/try_convnet.csv"
WEIGHTS_PATH = "analysis/final/try_convent_keras_modular_includeFlip_and_37relu.h5"

input_sizes = [(69, 69), (69, 69)]
PART_SIZE = 45

N_INPUT_VARIATION = 2

DO_VALID = True  # disable this to not bother with the validation set evaluation
DO_VALID_CORR = False  # not implemented yet
DO_VALID_SCATTER = True

VALID_CORR_OUTPUT_FILTER = np.zeros((37))
VALID_CORR_OUTPUT_FILTER[2] = 1  # star or artifact
VALID_CORR_OUTPUT_FILTER[3] = 1  # edge on yes
VALID_CORR_OUTPUT_FILTER[4] = 1  # edge on no
VALID_CORR_OUTPUT_FILTER[5] = 1  # bar feature yes
VALID_CORR_OUTPUT_FILTER[7] = 1  # spiral arms yes
VALID_CORR_OUTPUT_FILTER[14] = 1  # anything odd? no
VALID_CORR_OUTPUT_FILTER[18] = 1  # ring
VALID_CORR_OUTPUT_FILTER[19] = 1  # lence
VALID_CORR_OUTPUT_FILTER[20] = 1  # disturbed
VALID_CORR_OUTPUT_FILTER[21] = 1  # irregular
VALID_CORR_OUTPUT_FILTER[22] = 1  # other
VALID_CORR_OUTPUT_FILTER[23] = 1  # merger
VALID_CORR_OUTPUT_FILTER[24] = 1  # dust lane

N_Corr_Filter_Images = np.sum(VALID_CORR_OUTPUT_FILTER)

DO_VALIDSTUFF_ON_TRAIN = True

DO_TEST = False  # disable this to not generate predictions on the testset
DO_PRINT_FILTERS = False

IMAGE_OUTPUT_PATH = "images_wColorbar_newYear2_realValid"

output_names = ["smooth", "featureOrdisk", "NoGalaxy", "EdgeOnYes", "EdgeOnNo", "BarYes", "BarNo", "SpiralYes", "SpiralNo", "BulgeNo", "BulgeJust", "BulgeObvious", "BulgDominant", "OddYes", "OddNo", "RoundCompletly", "RoundBetween", "RoundCigar",
                "Ring", "Lense", "Disturbed", "Irregular", "Other", "Merger", "DustLane", "BulgeRound", "BlulgeBoxy", "BulgeNo2", "SpiralTight", "SpiralMedium", "SpiralLoose", "Spiral1Arm", "Spiral2Arm", "Spiral3Arm", "Spiral4Arm", "SpiralMoreArms", "SpiralCantTell"]

target_filename = os.path.basename(WEIGHTS_PATH).replace(".h5", ".npy.gz")
target_path_valid = os.path.join(
    "predictions/final/augmented/valid", target_filename)
target_path_test = os.path.join(
    "predictions/final/augmented/test", target_filename)

if copy_to_ram_beforehand:
    ra.myLoadFrom_RAM = True
    import copy_data_to_shm

y_train = np.load("data/solutions_train.npy")
ra.y_train = y_train

# split training data into training + a small validation set
ra.num_train = y_train.shape[0]

# integer division, is defining validation size
ra.num_valid = ra.num_train // 10
ra.num_train -= ra.num_valid


# training num check for EV usage
if ra.num_train != 55420:
    print "num_train = %s not %s" % (ra.num_train, 55420)

ra.y_valid = ra.y_train[ra.num_train:]
ra.y_train = ra.y_train[:ra.num_train]

load_data.num_train = y_train.shape[0]
load_data.train_ids = np.load("data/train_ids.npy")

ra.load_data.num_train = load_data.num_train
ra.load_data.train_ids = load_data.train_ids

ra.valid_ids = load_data.train_ids[ra.num_train:]
ra.train_ids = load_data.train_ids[:ra.num_train]


train_ids = load_data.train_ids
test_ids = load_data.test_ids

num_train = ra.num_train
num_test = len(test_ids)

num_valid = ra.num_valid

y_valid = ra.y_valid
y_train = ra.y_train

valid_ids = ra.valid_ids
train_ids = ra.train_ids

train_indices = np.arange(num_train)
valid_indices = np.arange(num_train, num_train + num_valid)
test_indices = np.arange(num_test)

N_TRAIN = num_train
N_VALID = num_valid


print("validation sample contains %s images. \n" %
      (ra.num_valid))

print 'initiate winsol class'
winsol = kaggle_winsol(BATCH_SIZE=BATCH_SIZE,
                       NUM_INPUT_FEATURES=NUM_INPUT_FEATURES,
                       PART_SIZE=PART_SIZE,
                       input_sizes=input_sizes,
                       LOSS_PATH=TRAIN_LOSS_SF_PATH,
                       WEIGHTS_PATH=WEIGHTS_PATH)

print "Build model"

if debug:
    print("input size: %s x %s x %s x %s" %
          (input_sizes[0][0],
           input_sizes[0][1],
           NUM_INPUT_FEATURES,
           BATCH_SIZE))

winsol.init_models()

# print 'output cuda_0'

# print np.shape(winsol.get_layer_output(layer='cuda_0'))

# print 'output maxout_2'

# max_out = winsol.get_layer_output(layer='maxout_2')
# print np.shape(max_out)


print 'end of testing.py'
