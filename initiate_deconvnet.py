import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
import load_data
import realtime_augmentation as ra
import time
import sys
import json
from datetime import timedelta
import os
import matplotlib.pyplot as plt
from termcolor import colored
import functools
from ellipse_fit import get_ellipse_kaggle_par
from custom_keras_model_and_fit_capsels import kaggle_winsol
from deconvnet import deconvnet

starting_time = time.time()

copy_to_ram_beforehand = False

debug = True

get_winsol_weights = False

BATCH_SIZE = 16  # keep in mind

NUM_INPUT_FEATURES = 3

included_flipped = True

USE_BLENDED_PREDICTIONS = False
PRED_BLENDED_PATH = ''
if debug:
    print os.path.isfile(PRED_BLENDED_PATH)


TRAIN_LOSS_SF_PATH = 'try_ellipseOnly_2param.txt'
# TRAIN_LOSS_SF_PATH = "trainingNmbrs_keras_modular_includeFlip_and_37relu.txt"
# TARGET_PATH = "predictions/final/try_convnet.csv"
WEIGHTS_PATH = 'analysis/final/try_goodWeights.h5'
TXT_OUTPUT_PATH = '_'
# IMAGE_OUTPUT_PATH = "img_ellipse_fit"

# NUM_ELLIPSE_PARAMS = 2
ELLIPSE_FIT = WEIGHTS_PATH.find('ellipse') >= 0
if ELLIPSE_FIT:
    postfix = '_ellipse'


DONT_LOAD_WEIGHTS = False

input_sizes = [(69, 69), (69, 69)]
PART_SIZE = 45

postfix = ''

N_INPUT_VARIATION = 2

# set to True if the prediction and evaluation should be done when the
# prediction file already exists
REPREDICT_EVERYTIME = False

TEST = False  # disable this to not generate predictions on the testset


output_names = ["smooth", "featureOrdisk", "NoGalaxy", "EdgeOnYes", "EdgeOnNo", "BarYes", "BarNo", "SpiralYes", "SpiralNo", "BulgeNo", "BulgeJust", "BulgeObvious", "BulgDominant", "OddYes", "OddNo", "RoundCompletly", "RoundBetween", "RoundCigar",
                "Ring", "Lense", "Disturbed", "Irregular", "Other", "Merger", "DustLane", "BulgeRound", "BlulgeBoxy", "BulgeNo2", "SpiralTight", "SpiralMedium", "SpiralLoose", "Spiral1Arm", "Spiral2Arm", "Spiral3Arm", "Spiral4Arm", "SpiralMoreArms", "SpiralCantTell"]

question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9),
                   slice(9, 13), slice(13, 15), slice(15, 18), slice(18, 25),
                   slice(25, 28), slice(28, 31), slice(31, 37)]

question_requierement = [None] * len(question_slices)
question_requierement[1] = question_slices[0].start + 1
question_requierement[2] = question_slices[1].start + 1
question_requierement[3] = question_slices[1].start + 1
question_requierement[4] = question_slices[1].start + 1
question_requierement[6] = question_slices[0].start
question_requierement[9] = question_slices[4].start
question_requierement[10] = question_slices[4].start

print 'Question requirements: %s' % question_requierement

spiral_or_ellipse_cat = [[(0, 1), (1, 1), (3, 0)], [(0, 1), (1, 0)]]

target_filename = os.path.basename(WEIGHTS_PATH).replace(".h5", ".npy.gz")
if get_winsol_weights:
    target_filename = os.path.basename(WEIGHTS_PATH).replace(".pkl", ".npy.gz")
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
ra.num_valid = ra.num_train // 100
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

print 'initiate deconvnet class'
winsol = deconvnet(BATCH_SIZE=BATCH_SIZE,
                   NUM_INPUT_FEATURES=NUM_INPUT_FEATURES,
                   PART_SIZE=PART_SIZE,
                   input_sizes=input_sizes,
                   LOSS_PATH=TRAIN_LOSS_SF_PATH,
                   WEIGHTS_PATH=WEIGHTS_PATH,
                   include_flip=included_flipped)

layer_formats = winsol.layer_formats
layer_names = layer_formats.keys()

print "Build model"

if debug:
    print("input size: %s x %s x %s x %s" %
          (input_sizes[0][0],
           input_sizes[0][1],
           NUM_INPUT_FEATURES,
           BATCH_SIZE))

winsol.init_models()

if debug:
    winsol.print_summary(postfix=postfix)

print "Load model weights"
winsol.load_weights(path=WEIGHTS_PATH, postfix=postfix)
winsol.WEIGHTS_PATH = ((WEIGHTS_PATH.split('.', 1)[0] + '_next.h5'))


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
validation_data = (
    [np.asarray(xs_valid[0]), np.asarray(xs_valid[1])], validation_data[1])

t_val = (time.time() - start_time)
print "  took %.2f seconds" % (t_val)


if debug:
    print("Free GPU Mem before first step %s MiB " %
          (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024. / 1024.))


def save_exit():
    # winsol.save()
    print "Done!"
    print ' run for %s' % timedelta(seconds=(time.time() - start_time))
    exit()


print 'Done'
# sys.exit(0)
