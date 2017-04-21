import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
import load_data
import realtime_augmentation as ra
import time
import sys
import json
from custom_for_keras import input_generator
from datetime import datetime, timedelta

from custom_keras_model_and_fit_capsels import kaggle_winsol

start_time = time.time()

copy_to_ram_beforehand = False

debug = True
predict = False  # not implemented
continueAnalysis = False
saveAtEveryValidation = True

get_winsol_weights = False

# only relevant if not continued and not gets winsol weights, see http://arxiv.org/abs/1511.06422 for
# describtion
# for this to work, the batch size has to be something like 128, 256, 512,
# ... reason not found
DO_LSUV_INIT = True

BATCH_SIZE = 256  # keep in mind

NUM_INPUT_FEATURES = 3

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
EPOCHS = 4
VALIDATE_EVERY = 2  # 20 # 12 # 6 # 6 # 6 # 5 #
NUM_EPOCHS_NONORM = 0.1
# this should be only a few, just .1 hopefully suffices.

INCLUDE_FLIP = False

TRAIN_LOSS_SF_PATH = "trainingNmbrs_test.txt"
# TARGET_PATH = "predictions/final/try_convnet.csv"
WEIGHTS_PATH = "analysis/final/try_test.h5"

LEARNING_RATE_SCHEDULE = {
    0: 0.4,
    2: 0.1,
    10: 0.05,
    40: 0.01,
    80: 0.005,
    120: 0.0005
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
if continueAnalysis or get_winsol_weights:
    LEARNING_RATE_SCHEDULE = {
        0: 0.1,
        20: 0.05,
        40: 0.01,
        80: 0.005
        # 0: 0.0001,
        # 500: 0.002,
        # 800: 0.0004,
        # 3200: 0.0002,
        # 4600: 0.0001,
    }


input_sizes = [(69, 69), (69, 69)]
PART_SIZE = 45

N_INPUT_VARIATION = 2


GEN_BUFFER_SIZE = 2

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


print("The training sample contains %s , the validation sample contains %s images. \n" %
      (ra.num_train,  ra.num_valid))
# train without normalisation for this fraction of the traininng sample, to get the weights in
# the right 'zone'.

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

print 'initiate winsol class'
winsol = kaggle_winsol(BATCH_SIZE=BATCH_SIZE,
                       NUM_INPUT_FEATURES=NUM_INPUT_FEATURES,
                       PART_SIZE=PART_SIZE,
                       input_sizes=input_sizes,
                       LEARNING_RATE_SCHEDULE=LEARNING_RATE_SCHEDULE,
                       MOMENTUM=MOMENTUM,
                       LOSS_PATH=TRAIN_LOSS_SF_PATH,
                       WEIGHTS_PATH=WEIGHTS_PATH, include_flip=INCLUDE_FLIP)

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
    winsol.load_weights(path=WEIGHTS_PATH)
    winsol.WEIGHTS_PATH = ((WEIGHTS_PATH.split('.', 1)[0] + '_next.h5'))
elif get_winsol_weights:
    print "import weights from run with original kaggle winner solution"
    winsol.load_weights()
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


if debug:
    print("Free GPU Mem before first step %s MiB " %
          (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024. / 1024.))


def save_exit():
    print "\nsaving..."
    winsol.save()
    print "Done!"
    print ' run for %s' % timedelta(seconds=(time.time() - start_time))
    exit()
    sys.exit(0)


try:
    print ''
    print "losses without training on validation sample up front"

    evalHist = winsol.evaluate([xs_valid[0], xs_valid[1]], y_valid=y_valid)

    if debug:
        print("Free GPU Mem after validation check %s MiB " %
              (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
               / 1024. / 1024.))
        print ''

    print "Train %s epoch without norm" % NUM_EPOCHS_NONORM

    time1 = time.time()

    no_norm_events = int(NUM_EPOCHS_NONORM * N_TRAIN)

    if no_norm_events:
        hist = winsol.fit_gen(modelname='model_noNorm',
                              data_generator=input_gen,
                              validation=validation_data,
                              samples_per_epoch=no_norm_events)

    if debug:
        print("\nFree GPU Mem before train loop %s MiB " %
              (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
               / 1024. / 1024.))

    print 'starting main training'

    if no_norm_events:
        eta = (time.time() - time1) / NUM_EPOCHS_NONORM * EPOCHS
        print 'rough ETA %s sec. -> finishes at %s' % (
            int(eta), datetime.now() + timedelta(seconds=eta))

    winsol.full_fit(data_gen=input_gen,
                    validation=validation_data,
                    samples_per_epoch=N_TRAIN,
                    validate_every=VALIDATE_EVERY,
                    nb_epochs=EPOCHS)

except KeyboardInterrupt:
    print "\ngot keyboard interuption"
    save_exit()
except ValueError:
    print "\ngot value error, could be the end of the generator in the fit"
    save_exit()

save_exit()
