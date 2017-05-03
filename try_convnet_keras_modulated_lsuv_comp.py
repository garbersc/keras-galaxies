import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
import matplotlib.pyplot as plt
import os
import load_data
import realtime_augmentation as ra
import time
import sys
import json
import functools


from custom_for_keras import input_generator
from datetime import datetime, timedelta
<<<<<<< HEAD:try_convnet_keras_modulated_lsuv_comp.py

from custom_keras_model_and_fit_capsels import kaggle_winsol
=======
from ellipse_fit import get_ellipse_kaggle_par
from custom_keras_model_ellipse import kaggle_ellipse_fit as kaggle_winsol
>>>>>>> bb89644b58f555c756cf77b6c7e440a0c0e5dfc6:try_maxout_keras_ellipse.py

start_time = time.time()

copy_to_ram_beforehand = False

debug = True
predict = False  # not implemented
continueAnalysis = False
saveAtEveryValidation = True

get_winsol_weights = False

# only relevant if not continued and not gets winsol weights, see http://arxiv.org/abs/1511.06422 for
# describtion
DO_LSUV_INIT = False
DO_LSUV_COMPARISON = True

<<<<<<< HEAD:try_convnet_keras_modulated_lsuv_comp.py
BATCH_SIZE = 256  # keep in mind
=======
BATCH_SIZE = 16  # 256  # keep in mind
>>>>>>> bb89644b58f555c756cf77b6c7e440a0c0e5dfc6:try_maxout_keras_ellipse.py

NUM_INPUT_FEATURES = 3

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
EPOCHS = 4
VALIDATE_EVERY = 2  # 20 # 12 # 6 # 6 # 6 # 5 #
NUM_EPOCHS_NONORM = 0.1
# this should be only a few, just .1 hopefully suffices.

<<<<<<< HEAD:try_convnet_keras_modulated_lsuv_comp.py
TRAIN_LOSS_SF_PATH = "trainingNmbrs_keras_modular_includeFlip_and_37relu.txt"
# TARGET_PATH = "predictions/final/try_convnet.csv"
WEIGHTS_PATH = "analysis/final/try_goodWeights.h5"
IMAGE_OUTPUT_PATH = "images_keras_modulated"
=======
NUM_ELLIPSE_PARAMS = 2

TRAIN_LOSS_SF_PATH = "trainingNmbrs_keras_ellipseOnly_" + \
    str(NUM_ELLIPSE_PARAMS) + "param_test.txt"
# TARGET_PATH = "predictions/final/try_convnet.csv"
WEIGHTS_PATH = "analysis/final/try_ellipseOnly_" + \
    str(NUM_ELLIPSE_PARAMS) + "param_test.h5"
>>>>>>> bb89644b58f555c756cf77b6c7e440a0c0e5dfc6:try_maxout_keras_ellipse.py

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
                       WEIGHTS_PATH=WEIGHTS_PATH, include_flip=False)

print "Build model"

if debug:
    print("input size: %s x %s x %s x %s" %
          (input_sizes[0][0],
           input_sizes[0][1],
           NUM_INPUT_FEATURES,
           BATCH_SIZE))

<<<<<<< HEAD:try_convnet_keras_modulated_lsuv_comp.py
winsol.init_models()

if debug:
    winsol.print_summary()

=======
winsol.init_models(input_shape=NUM_ELLIPSE_PARAMS)

if debug:
    winsol.print_summary(modelname='model_norm_ellipse', postfix='')
>>>>>>> bb89644b58f555c756cf77b6c7e440a0c0e5dfc6:try_maxout_keras_ellipse.py

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

<<<<<<< HEAD:try_convnet_keras_modulated_lsuv_comp.py
    input_gen = input_generator(train_gen)
=======
    input_gen = ellipse_par_gen(train_gen, num_par=NUM_ELLIPSE_PARAMS)
>>>>>>> bb89644b58f555c756cf77b6c7e440a0c0e5dfc6:try_maxout_keras_ellipse.py

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

<<<<<<< HEAD:try_convnet_keras_modulated_lsuv_comp.py
validation_data = (
    [xs_valid[0], xs_valid[1]], y_valid)
=======
if debug:
    print np.shape(xs_valid[0])

from numpy.linalg.linalg import LinAlgError

validation_data = ([], y_valid)
c = 0
for x in xs_valid[0]:
    try:
        validation_data[0].append(
            get_ellipse_kaggle_par(x, num_par=NUM_ELLIPSE_PARAMS))
    except LinAlgError, e:
        print 'try_conv'
        print c
        raise LinAlgError(e)
    c += 1

validation_data = (np.asarray(validation_data[0]), validation_data[1])
>>>>>>> bb89644b58f555c756cf77b6c7e440a0c0e5dfc6:try_maxout_keras_ellipse.py

t_val = (time.time() - start_time_val1)
print "  took %.2f seconds" % (t_val)

layer_formats = winsol.layer_formats
layer_names = layer_formats.keys()

imshow_c = functools.partial(
    plt.imshow, interpolation='none')  # , vmin=0.0, vmax=1.0)
imshow_g = functools.partial(
    plt.imshow, interpolation='none', cmap=plt.get_cmap('gray'))  # , vmin=0.0, vmax=1.0)


def normalize_img(img):
    min = np.amin(img)
    max = np.amax(img)
    return (img - min) / (max - min)


def _img_wall(img, norm=False):
    dim = len(np.shape(img))
    shape = np.shape(img)
    n_board_side = int(np.ceil(np.sqrt(shape[0])))
    n_board_square = int(n_board_side**2)
    if dim == 3:
        img_w = shape[1]
        img_h = shape[2]
        wall = np.zeros((n_board_side * img_w + n_board_side + 1,
                         n_board_side * img_h + n_board_side + 1))
    elif dim == 4:
        img_w = shape[2]
        img_h = shape[3]
        wall = np.zeros((shape[1], n_board_side * img_w + n_board_side + 1,
                         n_board_side * img_h + n_board_side + 1))
    else:
        raise TypeError(
            'Wrong dimension %s of the input' % dim)

    pos = [0, 0]
    for i in img:
        if pos[0] >= n_board_side:
            pos[0] = 0
            pos[1] = pos[1] + 1
        x0 = pos[0] * (img_w + 1) + 1
        x1 = (pos[0] + 1) * img_w + pos[0] + 1
        y0 = pos[1] * (img_h + 1) + 1
        y1 = (pos[1] + 1) * img_h + pos[1] + 1
        i_ = normalize_img(i) if norm else i
        if dim == 3:
            wall[x0:x1, y0:y1] = i_
        else:
            wall[:, x0:x1, y0:y1] = i_
        pos[0] = pos[0] + 1
    return wall


def print_weights(norm=False, nameprefix=''):
    if not os.path.isdir(IMAGE_OUTPUT_PATH):
        os.mkdir(IMAGE_OUTPUT_PATH)

    os.chdir(IMAGE_OUTPUT_PATH)
    weights_out_dir = 'weights'
    if norm:
        weights_out_dir += '_normalized'
    if not os.path.isdir(weights_out_dir):
        os.mkdir(weights_out_dir)
    os.chdir(weights_out_dir)

    print 'Printing weights'

    for name in layer_formats:
        if layer_formats[name] == 1:
            w, b = winsol.get_layer_weights(layer=name)
            w = np.transpose(w, (3, 0, 1, 2))

            w = _img_wall(w, norm)
            b = _img_wall(b, norm)
        # elif layer_formats[name] == 0:
        #     w, b = winsol.get_layer_weights(layer=name)
        #     w = _img_wall(w, norm)
        #     b = _img_wall(b, norm)
        else:
            continue

        for i in range(len(w)):
            imshow_g(w[i])
            if not norm:
                plt.colorbar()
            plt.savefig(
                nameprefix + 'weight_layer_%s_kernel_channel_%s.jpg' % (name, i))
            plt.close()

        imshow_g(b)
        if not norm:
            plt.colorbar()
        plt.savefig(nameprefix + 'weight_layer_%s_bias.jpg' % (name))
        plt.close()

    os.chdir('../..')


def save_exit():
    print "\nsaving..."
    winsol.save()
    print "Done!"
    print ' run for %s' % timedelta(seconds=(time.time() - start_time))
    exit()
    sys.exit(0)


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
elif DO_LSUV_COMPARISON:
    print "Load model weights"
    winsol.load_weights(path=WEIGHTS_PATH)
    winsol.WEIGHTS_PATH = ((WEIGHTS_PATH.split('.', 1)[0] + '_next.h5'))
    start_time_lsuv = time.time()
    print_weights(nameprefix='init')
    print 'Starting LSUV initialisation'
    # TODO check influence on the first epoch of the data generation of this
    # .next()
    train_batch = input_gen.next()[0]
    if debug:
        print type(train_batch)
        print np.shape(train_batch)
    winsol.LSUV_init(train_batch)
    print "  took %.2f seconds" % (time.time() - start_time_lsuv)
    print_weights(nameprefix='lsuv')
    save_exit()

if debug:
    print("Free GPU Mem before first step %s MiB " %
          (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024. / 1024.))


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
