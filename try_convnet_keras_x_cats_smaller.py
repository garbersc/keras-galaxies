# import pygpu.gpuarray.GpuContext as sbcuda
import numpy as np
import load_data
import realtime_augmentation as ra
import time
import sys
import os
import signal
import json
import os.path
from custom_for_keras import input_generator
from datetime import datetime, timedelta
from custom_keras_model_x_cat import kaggle_x_cat
from keras.optimizers import Adam
from make_class_weights import create_class_weight

start_time = time.time()

copy_to_ram_beforehand = False

debug = True
predict = False  # not implemented
continueAnalysis = True
saveAtEveryValidation = True

# FIXME reloading existing classweights seems not to work
use_class_weights = False

import_conv_weights = False

# only relevant if not continued and not gets winsol weights, see http://arxiv.org/abs/1511.06422 for
# describtion
# for this to work, the batch size has to be something like 128, 256, 512,
# ... reason not found
DO_LSUV_INIT = False

BATCH_SIZE = 256  # keep in mind

NUM_INPUT_FEATURES = 3

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
EPOCHS = 3
VALIDATE_EVERY = 5  # 20 # 12 # 6 # 6 # 6 # 5 #

INCLUDE_FLIP = True

TRAIN_LOSS_SF_PATH = "trainingNmbrs_10cat_smaller.txt"
# TARGET_PATH = "predictions/final/try_convnet.csv"
WEIGHTS_PATH = "analysis/final/try_10cat_smaller_next_next.h5"

CONV_WEIGHT_PATH = ''  # 'analysis/final/try_3cat_geometry_corr_geopics_next.h5'


LEARNING_RATE_SCHEDULE = {
    0: 0.001,
    5: 0.0005,
    100: 0.00005,
    200: 0.000005,
    # 40: 0.01,
    # 80: 0.005,
    # 120: 0.0005
    # 500: 0.04,
    # 0: 0.01,
    # 1800: 0.004,
    # 2300: 0.0004,
    # 0: 0.08,
    # 50: 0.04,
    # 2000: 0.008,
    # 3200: 0.0008,
    # 4600: 0.0004,
}
if continueAnalysis:
    LEARNING_RATE_SCHEDULE = {
        0: 0.0001,
        100: 0.00005,
        200: 0.00001,
        # 80: 0.005
        # 0: 0.0001,
        # 500: 0.002,
        # 800: 0.0004,
        # 3200: 0.0002,
        # 4600: 0.0001,
    }

optimizer = Adam(lr=LEARNING_RATE_SCHEDULE[0])

input_sizes = [(69, 69), (69, 69)]
PART_SIZE = 45

N_INPUT_VARIATION = 2


GEN_BUFFER_SIZE = 2

if copy_to_ram_beforehand:
    ra.myLoadFrom_RAM = True
    import copy_data_to_shm

y_train = np.load("data/solutions_train_10cat.npy")
# if not os.path.isfile('data/solution_certainties_train_10cat.npy'):
#    print 'generate 10 category solutions'
#    import solutions_to_10cat
#y_train = np.load('data/solution_certainties_train_10cat_alt_2.npy')
# y_train = np.concatenate((y_train, np.zeros((np.shape(y_train)[0], 30 - 3))),
#                          axis=1)

red_num = 0
ra.y_train = y_train

# split training data into training + a small validation set
ra.num_train = y_train.shape[0] - red_num

# integer division, is defining validation size
ra.num_valid = ra.num_train // 10
ra.num_valid -= ra.num_valid % BATCH_SIZE
ra.num_train -= ra.num_valid

ra.y_valid = ra.y_train[ra.num_train:ra.num_train + ra.num_valid]
ra.y_train = ra.y_train[:ra.num_train]

load_data.num_train = y_train.shape[0] - red_num
load_data.train_ids = np.load("data/train_ids.npy")

ra.load_data.num_train = load_data.num_train
ra.load_data.train_ids = load_data.train_ids

ra.valid_ids = load_data.train_ids[ra.num_train:ra.num_train + ra.num_valid]
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

if debug:
    print np.shape(y_valid)
    print y_valid[0]
    print np.shape(y_train)
print("The training sample contains %s , the validation sample contains %s images. \n" %
      (ra.num_train,  ra.num_valid))

# maybe put into class
with open(TRAIN_LOSS_SF_PATH, 'a')as f:
    if continueAnalysis:

        f.write('#continuing from ')
        f.write(WEIGHTS_PATH)
    # f.write("#wRandFlip \n")
    f.write("#The training is running for %s epochs, each with %s images. The validation sample contains %s images. \n" % (
        EPOCHS, N_TRAIN, ra.num_valid))
    f.write("#validation is done every %s epochs\n" % VALIDATE_EVERY)
    f.write("the learning rate schedule is ")
    json.dump(LEARNING_RATE_SCHEDULE, f)
    f.write('\n')


class_weights = None
if use_class_weights:
    class_weight_path = 'classweights.json'
    if os.path.isfile(class_weight_path):
        print 'loading category weights from %s' % class_weight_path
        with open(class_weight_path, 'r') as f:
            class_weights = json.load(f)
    else:
        print 'generating category weights...'
        class_weights = create_class_weight(
            y_train, savefile=class_weight_path)
        print 'saved category weights to %s' % class_weight_path

print 'initiate winsol class'
winsol = kaggle_x_cat(BATCH_SIZE=BATCH_SIZE,
                      NUM_INPUT_FEATURES=NUM_INPUT_FEATURES,
                      PART_SIZE=PART_SIZE,
                      input_sizes=input_sizes,
                      LEARNING_RATE_SCHEDULE=LEARNING_RATE_SCHEDULE,
                      MOMENTUM=MOMENTUM,
                      LOSS_PATH=TRAIN_LOSS_SF_PATH,
                      WEIGHTS_PATH=WEIGHTS_PATH, include_flip=INCLUDE_FLIP,
                      debug=debug)

print "Build model"

if debug:
    print("input size: %s x %s x %s x %s" %
          (input_sizes[0][0],
           input_sizes[0][1],
           NUM_INPUT_FEATURES,
           BATCH_SIZE))


print '\nidentify the used convolution layers\n'

used_conv_layers = {}
used_conv_layers = {'conv_1': [0, 1, 2, 12, 15, 30, 33, 48, 56, 57, 58, 59, 60, 61, 62, 63], 'conv_0': [0, 1, 2, 3, 4, 5, 6, 8, 17, 18, 23, 25, 26, 28, 30, 31], 'conv_3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 31, 35, 36, 54, 58, 59, 67, 73, 74, 77, 79, 83, 94, 101, 115, 118], 'conv_2': [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 38, 41, 50, 51, 52, 60, 62, 64, 68, 69, 72, 78, 80, 81, 82, 85, 86, 88, 91, 92, 96, 98, 99, 109, 112, 115, 118, 123]}


print
print 'convolution layers that will be used:'
print used_conv_layers
print


print 'building smaller model'
conv_filters_n = tuple(len(used_conv_layers['conv_%s' % i]) for i in range(4))
print conv_filters_n
winsol.init_models(final_units=10, optimizer=optimizer,
                   # loss='mean_squared_error',
                   conv_filters_n=conv_filters_n)


if debug:
    print winsol.models['model_norm'].get_output_shape_at(0)

if debug:
    winsol.print_summary()


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


def create_data_gen():
    augmented_data_gen = ra.realtime_augmented_data_gen(
        num_chunks=N_TRAIN / BATCH_SIZE * (EPOCHS + 1),
        chunk_size=BATCH_SIZE,
        augmentation_params=augmentation_params,
        ds_transforms=ds_transforms,
        target_sizes=input_sizes)

    post_augmented_data_gen = ra.post_augment_brightness_gen(
        augmented_data_gen, std=0.5)

    train_gen = load_data.buffered_gen_mp(
        post_augmented_data_gen, buffer_size=GEN_BUFFER_SIZE)

    input_gen = input_generator(train_gen)

    return input_gen


#  # may need doubling the generator,can be done with
# itertools.tee(iterable, n=2)
input_gen = create_data_gen()


def create_valid_gen():
    data_gen_valid = ra.realtime_fixed_augmented_data_gen(
        valid_indices,
        'train',
        ds_transforms=ds_transforms,
        chunk_size=N_VALID,
        target_sizes=input_sizes)
    # load_data.buffered_gen_mp(data_gen_valid, buffer_size=GEN_BUFFER_SIZE)
    return data_gen_valid


print "Preprocess validation data upfront"
start_time_val1 = time.time()

xs_valid = [[] for _ in xrange(num_input_representations)]

for data, length in create_valid_gen():
    for x_valid_list, x_chunk in zip(xs_valid, data):
        x_valid_list.append(x_chunk[:length])

xs_valid = [np.vstack(x_valid) for x_valid in xs_valid]
# move the colour dimension up
xs_valid = [x_valid.transpose(0, 3, 1, 2) for x_valid in xs_valid]

validation_data = (
    [xs_valid[0], xs_valid[1]], y_valid)

t_val = (time.time() - start_time_val1)
print "  took %.2f seconds" % (t_val)


if continueAnalysis:
    print "Load model weights"
    if not os.path.isfile(WEIGHTS_PATH):
        raise Exception('in ' + WEIGHTS_PATH + ' weights file not found')
    winsol.load_weights(path=WEIGHTS_PATH)
    winsol.WEIGHTS_PATH = ((WEIGHTS_PATH.split('.', 1)[0] + '_next.h5'))
elif import_conv_weights:
    print 'Import convnet weights from training with geometric forms'
    winsol.load_conv_layers(path=CONV_WEIGHT_PATH)
elif DO_LSUV_INIT:
    start_time_lsuv = time.time()
    print 'Starting LSUV initialisation'
    # TODO check influence on the first epoch of the data generation of this
    # .next()
    train_batch = input_gen.next()[0]
    if debug:
        print type(train_batch)
        print np.shape(train_batch)
    winsol.LSUV_init(train_batch)
    print "  took %.2f seconds" % (time.time() - start_time_lsuv)


# if debug:
#     print("Free GPU Mem before first step %s MiB " %
#           (sbcuda.free_gmem / 1024. / 1024.))


def save_exit():
    print "\nsaving..."
    winsol.save()
    print "Done!"
    print ' run for %s' % timedelta(seconds=(time.time() - start_time))
    pid = os.getpid()
    os.kill(pid, signal.SIGTERM)
    os._exit()


try:
    print ''
    print "losses without training on validation sample up front"
    if debug:
        print np.shape(y_valid)
        print winsol.models.keys()

    evalHist = winsol.evaluate([xs_valid[0], xs_valid[1]], y_valid=y_valid)

    if debug:
        # print("Free GPU Mem after validation check %s MiB " %
        #       (sbcuda.free_gmem / 1024. / 1024.))
        print ''

    time1 = time.time()

    # if debug:
    #     print("\nFree GPU Mem before train loop %s MiB " %
    #           (sbcuda.free_gmem / 1024. / 1024.))

    print 'starting main training'

    winsol.full_fit(data_gen=input_gen,
                    validation=validation_data,
                    samples_per_epoch=N_TRAIN,
                    validate_every=VALIDATE_EVERY,
                    nb_epochs=EPOCHS,
                    save_at_every_validation=saveAtEveryValidation,
                    class_weight=class_weights
                    )

except KeyboardInterrupt:
    print "\ngot keyboard interuption"
    save_exit()
except ValueError, e:
    print "\ngot value error"
    if debug:
        print '\t valid shape: %s' % str(np.shape(y_valid))
        print '\t shape valid data: %s ' % str((np.shape(xs_valid[0]), np.shape(xs_valid[1])))
        print '\t first valid result: %s' % y_valid[0]
        print '\t first image row: %s' % xs_valid[0][0, 0, 0]
    print ''
    print e
    save_exit()

save_exit()
