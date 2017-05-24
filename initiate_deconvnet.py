import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
import matplotlib.pyplot as plt
import realtime_augmentation as ra

import load_data
import functools
import time
import sys
import json
import os

from termcolor import colored
from datetime import datetime, timedelta
from custom_keras_model_and_fit_capsels import kaggle_winsol
from custom_for_keras import input_generator
from deconvnet import deconvnet

##############################################
# starting parameters
##############################################
starting_time = time.time()
copy_to_ram_beforehand = False
debug = True
get_deconv_weights = False
BATCH_SIZE = 32  # keep in mind
NUM_INPUT_FEATURES = 3
EPOCHS = 1
MAKE_PLOTS = True
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
ELLIPSE_FIT = WEIGHTS_PATH.find('ellipse') >= 0
if ELLIPSE_FIT:
    postfix = '_ellipse'
DONT_LOAD_WEIGHTS = False
input_sizes = [(69, 69), (69, 69)]
PART_SIZE = 45
postfix = ''
N_INPUT_VARIATION = 2
GEN_BUFFER_SIZE = 2
# set to True if the prediction and evaluation should be done when the
# prediction file already exists
REPREDICT_EVERYTIME = False
IMAGE_OUTPUT_PATH = "images_deconv"
TEST = False  # disable this to not generate predictions on the testset

##############################################
# main
##############################################

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
if get_deconv_weights:
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
ra.num_valid -= ra.num_valid % BATCH_SIZE
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
deconv = deconvnet(BATCH_SIZE=BATCH_SIZE,
                   NUM_INPUT_FEATURES=NUM_INPUT_FEATURES,
                   PART_SIZE=PART_SIZE,
                   input_sizes=input_sizes,
                   LOSS_PATH=TRAIN_LOSS_SF_PATH,
                   WEIGHTS_PATH=WEIGHTS_PATH,
                   include_flip=included_flipped)

layer_formats = deconv.layer_formats
layer_names = layer_formats.keys()

print "Build model"

if debug:
    print("input size: %s x %s x %s x %s" %
          (input_sizes[0][0],
           input_sizes[0][1],
           NUM_INPUT_FEATURES,
           BATCH_SIZE))

deconv.init_models()

if debug:
    deconv.print_summary(postfix=postfix)


print "Load model weights..."
deconv.load_weights(path=WEIGHTS_PATH, postfix=postfix)
deconv.WEIGHTS_PATH = ((WEIGHTS_PATH.split('.', 1)[0] + '_next.h5'))
# c, y, x, filter
# y, x, c, filter
deconv_weight, conv_bias = deconv.get_layer_weights(layer='conv_0')
deconv_weight = np.transpose(deconv_weight, (1, 2, 0, 3))
deconv.models['model_deconv'].get_layer(
    'deconv_layer').set_weights([deconv_weight, ])
deconv.models['model_deconv'].get_layer('debias_layer').set_weights(
    [conv_bias, ])


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
    # deconv.save()
    print "Done!"
    print ' run for %s' % timedelta(seconds=(time.time() - start_time))
    exit()


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

    if dim == 4:
        wall = np.transpose(wall, (1, 2, 0))
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
            w, b = deconv.get_layer_weights(layer=name)
            w = np.transpose(w, (3, 0, 1, 2))

            w = _img_wall(w, norm)
            b = _img_wall(b, norm)
        elif layer_formats[name] == 0:
            w, b = deconv.get_layer_weights(layer=name)
            w = _img_wall(w, norm)
            b = _img_wall(b, norm)
        else:
            continue

    w, b = deconv.models['model_deconv'].get_layer(
        'deconv_layer').get_weights()
    w = np.transpose(w, (3, 0, 1, 2))

    w = _img_wall(w, norm)
    b = _img_wall(b, norm)

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


def reshape_output(x):
    _shape = x.shape
    if _shape[0] == 1:
        x = x.reshape((_shape[2] / _shape[3], _shape[1], _shape[3], _shape[3]))
    else:
        x = x.reshape((_shape[0], _shape[2] / _shape[3],
                       _shape[1], _shape[3], _shape[3]))
    return x


def print_output(nr_images=2, plots=False, combined_rgb=True, combined_variations=False):
    if debug:
        print 'Checking save directory...'
    if not os.path.isdir(IMAGE_OUTPUT_PATH):
        os.mkdir(IMAGE_OUTPUT_PATH)
        print 'Creating directory %s...' % (IMAGE_OUTPUT_PATH)

    os.chdir(IMAGE_OUTPUT_PATH)

    print '  Output images will be saved at dir: %s' % (IMAGE_OUTPUT_PATH)

    print 'Collecting output from Deconvnet... this may take a while...'

    output_deconv = deconv.predict(
        x=validation_data[0], modelname='model_deconv')
    output_deconv = reshape_output(output_deconv)
    if debug:
        print 'Deconv output shape:' + str(np.shape(output_deconv))
        print 'Ids of input images are: ' + str(valid_ids[0:nr_images])
    deconv.layer_formats['input_merge'] = 4

    for i, img in enumerate(output_deconv[0:nr_images]):
        image_nr = i
        if type(image_nr) == int:
            input_img = [np.asarray([validation_data[0][0][image_nr]]),
                         np.asarray([validation_data[0][1][image_nr]])]
        if debug:
            print 'Collecting output from merge layer...'
        intermediate_outputs = {}
        intermediate_outputs['input_merge'] = np.asarray(deconv.get_layer_output(
            'input_merge', input_=input_img))
        intermediate_outputs['input_merge'] = reshape_output(
            intermediate_outputs['input_merge'])
        if debug:
            print 'Shape of intermediate outputs: ' + str(np.shape(intermediate_outputs['input_merge']))
            print 'Shape of input image array: ' + str(np.shape(input_img[0][0]))

        if plots:
            print 'Creating plots for Image %s' % (valid_ids[i])
            for j, channel in enumerate(img):
                canvas, (im1, im2, im3) = plt.subplots(1, 3)
                im1.imshow(np.transpose((input_img[0][0]), (1, 2, 0)),
                           interpolation='none')
                im1.set_title('Input Image %s' % (valid_ids[i]))

                channel -= np.min(channel)
                channel = channel / np.max(channel)

                if combined_variations:  # not working yet
                    im2.imshow(_img_wall(np.transpose(
                        intermediate_outputs['input_merge'], (3, 2, 1, 0))), interpolation='none', cmap=plt.get_cmap('gray'))
                    im2.set_title('Variation %s' % j)
                else:
                    im2.imshow(np.transpose(intermediate_outputs['input_merge'][j], (
                        1, 2, 0)), interpolation='none', cmap=plt.get_cmap('gray'))
                    im2.set_title('Variation %s' % j)

                if combined_rgb:
                    fpicture = np.transpose((channel),
                                            (1, 2, 0))
                    im3.imshow(fpicture, interpolation='none')
                    im3.set_title('Deconv Variation %s' % j)
                else:
                    im3.imshow(_img_wall(channel), interpolation='none',
                               cmap=plt.get_cmap('gray'))
                    im3.set_title('Deconv Variation %s' % j)
                plt.savefig('image_%s_variation_%s.jpg' %
                            (valid_ids[i], j), dpi=300)
                plt.close()

        # for i, img in enumerate(output_deconv[0:5]):
        #     for j, channel in enumerate(img):
        #         imshow_g(channel)
        #         plt.colorbar()
        #         plt.savefig('%s_%s.jpg' % (i, j), dpi=300)
        #         plt.close()

        # if type(image_nr) == int:
        #     imshow_c(np.transpose(input_img[0][0], (1, 2, 0)))
        #     plt.savefig('input_fig_%s_rotation_0.jpg' % (image_nr))
        #     plt.close()

        #     imshow_c(np.transpose(input_img[1][0], (1, 2, 0)))
        #     plt.savefig('input_fig_%s_rotation_45.jpg' % (image_nr))
        #     plt.close()

        #     for i in range(len(input_img[0][0])):
        #         imshow_g(input_img[0][0][i])
        #         plt.savefig('input_fig_%s_rotation_0_dim_%s.jpg' %
        #                     (image_nr, i))
        #         plt.close()

        #     for i in range(len(input_img[1][0])):
        #         imshow_g(input_img[1][0][i])
        #         plt.savefig('input_fig_%s_rotation_45_dim_%s.jpg' %
        #                     (image_nr, i))
        #         plt.close()

        #     imshow_g(_img_wall(intermediate_outputs['input_merge']))
        #     plt.colorbar()
        #     plt.savefig('output_fig_%s_%s.jpg' %
        #                 (image_nr, 'input_merge'))
        #     plt.close()

    os.chdir('..')


if MAKE_PLOTS:
    print_output(plots=True)
    save_exit()

print 'Done'
# sys.exit(0)
