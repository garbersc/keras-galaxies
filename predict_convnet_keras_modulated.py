# at some point try
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

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

if debug:
    winsol.print_summary()

print "Load model weights"
winsol.load_weights(path=WEIGHTS_PATH)
winsol.WEIGHTS_PATH = ((WEIGHTS_PATH.split('.', 1)[0] + '_next.h5'))

if get_winsol_weights:
    print "import weights from run with original kaggle winner solution"
    winsol.load_weights()

print "Set up data loading"

ds_transforms = [
    ra.build_ds_transform(3.0, target_size=input_sizes[0]),
    ra.build_ds_transform(
        3.0, target_size=input_sizes[1])
    + ra.build_augmentation_transform(rotation=45)
]

num_input_representations = len(ds_transforms)

augmentation_params = {
    'zoom_range': (1.0 / 1.3, 1.3),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': True,
}


def create_valid_gen():
    data_gen_valid = ra.realtime_fixed_augmented_data_gen(
        valid_indices,
        'train',
        ds_transforms=ds_transforms,
        chunk_size=N_VALID,
        target_sizes=input_sizes)
    return data_gen_valid


print "Preprocess validation data upfront"
start_time = time.time()

xs_valid = [[] for _ in xrange(num_input_representations)]

for data, length in create_valid_gen():
    for x_valid_list, x_chunk in zip(xs_valid, data):
        x_valid_list.append(x_chunk[:length])

xs_valid = [np.vstack(x_valid) for x_valid in xs_valid]
# move the colour dimension up
xs_valid = [x_valid.transpose(0, 3, 1, 2) for x_valid in xs_valid]

validation_data = (
    [xs_valid[0], xs_valid[1]], y_valid)

t_val = (time.time() - start_time)
print "  took %.2f seconds" % (t_val)


if debug:
    print("Free GPU Mem before first step %s MiB " %
          (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024. / 1024.))


def save_exit():
    # print "\nsaving..."
    # winsol.save()
    print "Done!"
    print ' run for %s' % timedelta(seconds=(time.time() - start_time))
    exit()
    sys.exit(0)


try:
    print ''

    if DO_VALID:
        evalHist = winsol.evaluate([xs_valid[0], xs_valid[1]], y_valid=y_valid)

    print ''
    predictions = winsol.predict(
        [xs_valid[0], xs_valid[1]])

    print "Write predictions to %s" % target_path_valid
    load_data.save_gz(target_path_valid, predictions)

except KeyboardInterrupt:
    print "\ngot keyboard interuption"
    save_exit()
except ValueError:
    print "\ngot value error, could be the end of the generator in the fit"
    save_exit()


rmse_valid = evalHist['rmse'][-1]
rmse_augmented = np.sqrt(np.mean((y_valid - predictions)**2))
print "  MSE (last iteration):\t%.6f" % rmse_valid
print "  MSE (augmented):\t%.6f  RMSE/mean: %s " % (rmse_augmented,
                                                    rmse_augmented / np.mean(
                                                        y_valid))
print "  MSE output wise (augmented):"
for i in xrange(0, VALID_CORR_OUTPUT_FILTER.shape[0]):
    oneMSE = np.sqrt(np.mean((y_valid.T[i] - predictions.T[i])**2))
    if oneMSE / np.mean(y_valid.T[i]) > 1.2 * rmse_augmented / np.mean(
            y_valid):
        print colored("    output %s (%s): \t%.6f  RMSE/mean: %s " % (
            output_names[i], i, oneMSE, oneMSE / np.mean(y_valid.T[i])), 'red')
    elif oneMSE / np.mean(y_valid.T[i]) < 0.8 * rmse_augmented / np.mean(
            y_valid):
        print colored("    output %s (%s): \t%.6f  RMSE/mean: %s " % (
            output_names[i], i, oneMSE, oneMSE / np.mean(y_valid.T[i])),
            'green')
    else:
        print ("    output %s (%s): \t%.6f  RMSE/mean: %s " %
               (output_names[i], i, oneMSE, oneMSE / np.mean(y_valid.T[i])))


if DO_VALID_SCATTER:
    print 'do scatter plots'
    if not os.path.isdir(IMAGE_OUTPUT_PATH):
        os.mkdir(IMAGE_OUTPUT_PATH)
    # plt.gray()
    os.chdir(IMAGE_OUTPUT_PATH)
    if not os.path.isdir("ValidScatter"):
        os.mkdir("ValidScatter")
    os.chdir("ValidScatter")

    for i in xrange(0, VALID_CORR_OUTPUT_FILTER.shape[0]):
        y = predictions.T[i]
        x = y_valid.T[i]
        fig, ax = plt.subplots()
        fit = np.polyfit(x, y, deg=1)
        ax.plot(x, fit[0] * x + fit[1], color='red')
        ax.scatter(x, y)
        plt.ylabel('prediction')
        plt.xlabel('target')
        plt.title("valid %s" % (output_names[i]))
        oneMSE = np.sqrt(np.mean((y_valid.T[i] - predictions.T[i])**2))
        plt.text(60, .025, 'RMSE: %s , RMSE/mean: %s ' %
                 (oneMSE, oneMSE / np.mean(y_valid.T[i])))
        plt.savefig("validScatter_%s_%s.jpg" % (i, output_names[i]))
        plt.close()

    os.chdir("../..")

save_exit()
