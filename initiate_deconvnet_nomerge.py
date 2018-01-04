import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
import matplotlib.pyplot as plt
import realtime_augmentation as ra

import load_data
import functools
import time
import os

from PIL import Image
from datetime import timedelta, date
from custom_keras_model_x_cat import kaggle_x_cat
from simple_deconv import simple_deconv
from custom_for_keras import input_generator

##############################################
# starting parameters
##############################################
starting_time = time.time()
copy_to_ram_beforehand = False
debug = True
get_deconv_weights = False
BATCH_SIZE = 1  # keep in mind
NUM_INPUT_FEATURES = 3
EPOCHS = 1
MAKE_PLOTS = True
included_flipped = True

TRAIN_LOSS_SF_PATH = 'trainingNmbrs_10cat_smaller.txt'
WEIGHTS_PATH = 'analysis/final/try_10cat_new_run.h5'
TXT_OUTPUT_PATH = '_'

input_sizes = [(69, 69), (69, 69)]
PART_SIZE = 45
postfix = ''
N_INPUT_VARIATION = 2
GEN_BUFFER_SIZE = 2

date = date.today()
IMAGE_OUTPUT_DIR = "images_deconv"
IMAGE_OUTPUT_SUFFIX = str(date)

LOAD_WEIGHTS = False
CUSTOM_WEIGHTS = False
RANDOM_WEIGHTS = False

test_image = False
WALL_OUTPUT = True

USE_SIMPLE_MODEL = True
PREDICT_MERGE = True

##############################################
# main, defining and loading stuff
##############################################

output_names = ['round', 'broad_ellipse', 'small_ellipse', 'edge_bulg',
                'edge_no_bulge', 'disc', 'spiral_1_arm', 'spiral_2_arm',
                'spiral_other', 'other']
question_slices = [slice(0, 10)]

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

y_train = np.load("data/solutions_train_10cat.npy")
red_num = 61556
ra.y_train = y_train
print 'Total amount of images avaible: ' + str(y_train.shape[0])

# split training data into training + a small validation set
ra.num_train = y_train.shape[0] - red_num

# integer division, is defining validation size
ra.num_valid = ra.num_train // 10
ra.num_valid -= ra.num_valid % BATCH_SIZE
ra.num_train -= ra.num_valid


# training num check for EV usage
if ra.num_train != 55420:
    print "num_train = %s not %s" % (ra.num_train, 55420)

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


print("validation sample contains %s images. \n" %
      (ra.num_valid))

print 'initiate deconvnet class'
# if USE_SIMPLE_MODEL:
deconv = simple_deconv(BATCH_SIZE=BATCH_SIZE,
                       NUM_INPUT_FEATURES=NUM_INPUT_FEATURES,
                       PART_SIZE=PART_SIZE,
                       input_sizes=input_sizes,
                       LOSS_PATH=TRAIN_LOSS_SF_PATH,
                       WEIGHTS_PATH=TRAIN_LOSS_SF_PATH,
                       include_flip=included_flipped)

# else:
mergedeconv = kaggle_x_cat(BATCH_SIZE=BATCH_SIZE,
                           NUM_INPUT_FEATURES=NUM_INPUT_FEATURES,
                           PART_SIZE=PART_SIZE,
                           input_sizes=input_sizes,
                           LOSS_PATH=None,
                           WEIGHTS_PATH=WEIGHTS_PATH,
                           include_flip=included_flipped)

layer_formats = mergedeconv.layer_formats
layer_names = layer_formats.keys()

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

print 'building smaller model'
conv_filters_n = tuple(len(used_conv_layers['conv_%s' % i]) for i in range(4))
print conv_filters_n
deconv.init_models(final_units=10, conv_filters_n=conv_filters_n)
mergedeconv.init_models(final_units=10, conv_filters_n=conv_filters_n)

if debug:
    deconv.print_summary(postfix=postfix)

##############################################
# defining weights
##############################################

print
if LOAD_WEIGHTS:
    print "Load model weights..."
    if not os.path.isfile(WEIGHTS_PATH):
        raise Exception('in ' + WEIGHTS_PATH + ' weights file not found')
    deconv.load_weights(path=WEIGHTS_PATH, postfix=postfix)
    deconv.WEIGHTS_PATH = ((WEIGHTS_PATH))

    # Loading conv_0 weights into deconv_0
    deconv_weight_0 = deconv.get_layer_weights(layer='conv_0')
    conv_bias_0 = deconv.get_layer_weights(layer='conv_0_bias')
    print 'Shape of conv weights:' + str(np.shape(deconv_weight_0[0]))
    print 'Shape of conv bias weights:' + str(np.shape(conv_bias_0))

    deconv.models['model_deconv'].get_layer(
        'deconv_layer_0').set_weights([deconv_weight_0][0])
    deconv.models['model_deconv'].get_layer('debias_layer_0').set_weights(
        conv_bias_0)

    # Loading conv_1 weights into deconv_1
    deconv_weight_1 = deconv.get_layer_weights(layer='conv_1')
    conv_bias_1 = deconv.get_layer_weights(layer='conv_1_bias')

    deconv.models['model_deconv'].get_layer(
        'deconv_layer_1').set_weights([deconv_weight_1][0])
    deconv.models['model_deconv'].get_layer('debias_layer_1').set_weights(
        conv_bias_1)


elif CUSTOM_WEIGHTS:
    print "Initializing model with custom weights..."
    conv_weights = deconv.get_layer_weights(layer='conv_0')
    conv_bias = deconv.get_layer_weights(layer='conv_0_bias')
    if debug:
        print 'Shape of conv weights:' + str(np.shape(conv_weights[0]))
        print 'Shape of conv bias weights:' + str(np.shape(conv_bias))

    custom_weights = np.zeros(conv_weights[0].shape)
    custom_bias = np.zeros(conv_bias[0].shape)

    custom_weights = custom_weights.transpose(3, 2, 0, 1)
    for i in range(0, len(custom_weights)):
        # for j in range(0, len(unity_weights[i])):
        #     unity_weights[i][j][5][5] = 1
        custom_weights[i][0][3][3] = 0

    print 'Filter matrix is:'
    print custom_weights[0][0]
    custom_weights = custom_weights.transpose(2, 3, 1, 0)

    deconv.models['model_deconv'].get_layer(
        'deconv_layer').set_weights([custom_weights])
    deconv.models['model_deconv'].get_layer('debias_layer').set_weights(
        [custom_bias])

    deconv.models['model_norm'].get_layer('main_seq').get_layer(
        'conv_0').set_weights([custom_weights])
    deconv.models['model_norm'].get_layer('main_seq').get_layer('conv_0_bias').set_weights(
        [custom_bias])

elif RANDOM_WEIGHTS:
    print 'Initializing model with random weights...'

    deconv_weight = deconv.get_layer_weights(layer='conv_0')
    conv_bias = deconv.get_layer_weights(layer='conv_0_bias')

    if debug:
        print 'Shape of conv weights:' + str(np.shape(deconv_weight[0]))
        print 'Shape of conv bias weights:' + str(np.shape(conv_bias))

    deconv.models['model_deconv'].get_layer(
        'deconv_layer').set_weights([deconv_weight][0])
    deconv.models['model_deconv'].get_layer('debias_layer').set_weights(
        conv_bias)

print 'Setting weights done.'
print


##############################################
# generating data
##############################################

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


def load_image(infilename, rotate=False):
    img = Image.open(infilename)
    img.load()
    if rotate:
        img.rotate(45)
    data = np.asarray(img, dtype='uint8')
    return data


if test_image:
    print
    print 'Loading test images'
    ti = load_image('test_images/testpic_1.jpg')
    ti = np.transpose((ti), (2, 0, 1))
    ti_rot = load_image('test_images/testpic_1.jpg', rotate=True)
    ti_rot = np.transpose((ti_rot), (2, 0, 1))
    ti2 = load_image('test_images/testpic_2.jpg')
    ti2 = np.transpose((ti2), (2, 0, 1))
    ti2_rot = load_image('test_images/testpic_2.jpg', rotate=True)
    ti2_rot = np.transpose((ti2_rot), (2, 0, 1))


##############################################
# auxiliary tools
##############################################

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


def reshape_output(x):
    _shape = x.shape
    if _shape[0] == 1:
        x = x.reshape((_shape[2] / _shape[3], _shape[1], _shape[3], _shape[3]))
    else:
        x = x.reshape((_shape[0], _shape[2] / _shape[3],
                       _shape[1], _shape[3], _shape[3]))
    return x


##############################################
# running deconv
##############################################

def print_output(nr_images=1, plots=True, combined_rgb=True, wall_variations=True, wall_output=True, norm_single_img=True):
    if debug:
        print
        print 'Checking save directory...'
    if not os.path.isdir(IMAGE_OUTPUT_DIR):
        os.mkdir(IMAGE_OUTPUT_DIR)
        os.chdir(IMAGE_OUTPUT_DIR)
        if not os.path.isdir(IMAGE_OUTPUT_SUFFIX):
            os.mkdir(IMAGE_OUTPUT_SUFFIX)
            os.chdir(IMAGE_OUTPUT_SUFFIX)
        print 'Creating directory ' + IMAGE_OUTPUT_DIR + '/' + IMAGE_OUTPUT_SUFFIX

    else:
        os.chdir(IMAGE_OUTPUT_DIR)
        if not os.path.isdir(IMAGE_OUTPUT_SUFFIX):
            os.mkdir(IMAGE_OUTPUT_SUFFIX)
            os.chdir(IMAGE_OUTPUT_SUFFIX)
            print 'Creating directory ' + IMAGE_OUTPUT_DIR + '/' + IMAGE_OUTPUT_SUFFIX
        else:
            os.chdir(IMAGE_OUTPUT_SUFFIX)

    print
    print '  Output images will be saved at dir: ' + IMAGE_OUTPUT_DIR + '/' + IMAGE_OUTPUT_SUFFIX
    print 'Collecting output from Deconvnet... this may take a while...'
    if debug:
        print 'Shape of validation data array at [0]: ' + str(np.shape(validation_data[0]))

    if debug:
        print
        print 'Ids of input images are: ' + str(valid_ids[0:ra.num_valid])
    mergedeconv.layer_formats['input_merge'] = 4

    if test_image and PREDICT_MERGE:
        repeat_size = ra.num_valid
        input_img = [np.ones(shape=(np.asarray([validation_data[0][0][0]]).shape)),
                     np.ones(
            shape=(np.asarray([validation_data[0][0][1]]).shape))]
        input_img[0][0, :, :, :] = ti
        input_img[0] = input_img[0].astype('uint8')
        input_img[1][0, :, :, :] = ti2
        input_img[1] = input_img[1].astype('uint8')
        merge_outputs = {}
        merge_outputs['input_merge'] = np.asarray(mergedeconv.get_layer_output(
            'input_merge', input_=input_img))
        merge_outputs['input_merge'] = reshape_output(
            merge_outputs['input_merge'])
        merge_outputs['input_merge'] = merge_outputs['input_merge'] / 255
        print 'Shape of merge outputs: ' + str(np.shape(merge_outputs['input_merge']))
        validation_data_ = (np.asarray(
            np.repeat(merge_outputs['input_merge'], repeat_size, axis=0)), validation_data[1])

        output_deconv = deconv.predict(
            x=validation_data_[0], modelname='model_deconv')

    if test_image and not PREDICT_MERGE:
        print 'Using test image for deconv'

        ''' generating test image'''
        if USE_SIMPLE_MODEL:
            repeat_size = ra.num_valid / 2
            input_img = np.ones(
                shape=np.asarray([validation_data[0][0][0]]).shape)
            input_img2 = np.ones(
                shape=np.asarray([validation_data[0][0][0]]).shape)
            input_img[0, :, :, :] = ti
            input_img2[0, :, :, :] = ti2
            input_img = np.append(input_img, input_img2, axis=0)
            input_img = input_img.astype('uint8')
            validation_data_ = (np.asarray(
                np.repeat(input_img, repeat_size, axis=0)), validation_data[1])

        else:
            repeat_size = ra.num_valid
            input_img = [np.ones(shape=(np.asarray([validation_data[0][0][0]]).shape)),
                         np.ones(
                shape=(np.asarray([validation_data[0][0][1]]).shape))]
            input_img[0][0, :, :, :] = ti
            input_img[0] = input_img[0].astype('uint8')
            input_img[1][0, :, :, :] = ti2
            input_img[1] = input_img[1].astype('uint8')
            validation_data_ = ([np.asarray(np.repeat(input_img[0], repeat_size, axis=0)),
                                 np.asarray(np.repeat(input_img[1], repeat_size, axis=0))], validation_data[1])

        output_deconv = deconv.predict(
            x=validation_data_[0], modelname='model_deconv')

    if not test_image:
        img_nr = 0

        repeat_size = ra.num_valid / 2
        merge_outputs = {}
        merge_outputs['input_merge'] = np.asarray(mergedeconv.get_layer_output(
            'input_merge', input_=validation_data[0]))
        merge_outputs['input_merge'] = reshape_output(
            merge_outputs['input_merge'])
        merge_outputs['input_merge'] = merge_outputs['input_merge']
        print 'Shape of merge outputs: ' + str(np.shape(merge_outputs['input_merge']))

        validation_data_ = (np.asarray(
            np.repeat(merge_outputs['input_merge'], repeat_size, axis=0)), validation_data[1])

        print 'Shape of validation_data_: ' + str(np.shape(validation_data_[0]))

        output_deconv = deconv.predict(
            x=validation_data_[0][img_nr], modelname='model_deconv')
    if debug:
        print
        print 'Deconv output shape:' + str(np.shape(output_deconv))
        print 'Shape of merge outputs: ' + str(np.shape(merge_outputs['input_merge'][0]))


##############################################
# plotting
##############################################

    if USE_SIMPLE_MODEL:
        if wall_output:
            if norm_single_img:
                for i, img_vars in enumerate(output_deconv[img_nr:img_nr + 16]):
                    output_deconv[i] -= np.min(output_deconv[i])
                    output_deconv[i] = output_deconv[i] / \
                        np.max(output_deconv[i])

            canvas, (im1, im2, im3) = plt.subplots(1, 3)

            im1.imshow(np.transpose(validation_data[0][0][img_nr], (1, 2, 0)))
            im1.set_title('Input image %s' % valid_ids[img_nr])

            im2.imshow(_img_wall(merge_outputs['input_merge'][img_nr]))
            im2.set_title('Variations')
            im2.axis('off')

            im3.imshow(_img_wall(output_deconv))
            im3.axis('off')

            plt.savefig('walled_output_%s' % valid_ids[img_nr], dpi=600)
            plt.close

        else:
            for i, img_vars in enumerate(output_deconv[img_nr:img_nr + 16]):
                image_nr = i
                # get images in range for pyplot:
                img_vars = img_vars / 255

                fpicture = img_vars
                fpicture = np.transpose(fpicture, (1, 2, 0))

                if norm_single_img:
                    fpicture -= np.min(fpicture)
                    fpicture = fpicture / np.max(fpicture)
                    canvas, (im1, im2) = plt.subplots(1, 2)
                    if test_image:
                        im1.imshow(np.transpose(
                            (merge_outputs['input_merge'][i]), (1, 2, 0)))
                        im1.set_title('Input Image variation %s' % i)
                    else:
                        im1.imshow(np.transpose(
                            (merge_outputs['input_merge'][img_nr][i]), (1, 2, 0)))
                        im1.set_title('Input Image variation %s' % i)

                    im2.imshow(fpicture)
                    im2.set_title('Deconv output')

                plt.savefig('deconv_test_%s.jpg' % i)
                plt.close

    else:
        # output_deconv = np.reshape(
        #     output_deconv, (ra.num_valid, 3, 16, 45, 45))
        # output_deconv = np.transpose(output_deconv, (0, 2, 1, 4, 3))

        for i, img_vars in enumerate(output_deconv[0:nr_images]):
            image_nr = i
            # get images in range for pyplot:
            img_vars = img_vars / 255

            if not norm_single_img:
                img_vars -= np.min(img_vars)
                img_vars = img_vars / np.max(img_vars)

                if type(image_nr) == int and not test_image:
                    input_img = [np.asarray([validation_data[0][0][image_nr]]),
                                 np.asarray([validation_data[0][1][image_nr]])]
            merge_outputs = {}
            merge_outputs['input_merge'] = np.asarray(deconv.get_layer_output(
                'input_merge', input_=input_img))
            merge_outputs['input_merge'] = reshape_output(
                merge_outputs['input_merge'])
            merge_outputs['input_merge'] = merge_outputs['input_merge'] / 255

        if plots:
            print 'Creating plots for Image %s' % (valid_ids[i])
            if wall_variations & wall_output:
                print 'Creating image wall for variations and output...'
                print
                fpicture = np.empty((3, 45, 45))
                fpicture[0] = img_vars[0][0]
                fpicture[2] = img_vars[1][0]
                fpicture[1] = img_vars[2][0]
                fpicture = np.transpose(fpicture, (1, 2, 0))

                if norm_single_img:
                    fpicture -= np.min(fpicture)
                    fpicture = fpicture / np.max(fpicture)

                plt.imshow(fpicture)
                plt.savefig('deconv_var_0.jpg')
                plt.close

                # fpicture = np.empty((16, 1196, 1200, 3))
                # for j in range(0, 15):
                #     plt.imshow(_img_wall(img_vars[j]))
                #     plt.axis('off')
                #     plt.savefig('rgb_batch_%s.jpg' % j, bbox_inches='tight',
                #                 frameon='false', pad_inches=0.01, dpi=300)
                #     plt.close
                #     fpicture[j] = load_image('rgb_batch_%s.jpg' % j)
                # fpicture = np.transpose(fpicture, (0, 3, 1, 2))
                # plt.imshow(_img_wall(fpicture))
                # plt.savefig('deconv_out.jpg', dpi=300)
                # plt.close

                canvas, (im1, im2, im3) = plt.subplots(1, 3)
                im1.imshow(np.transpose(
                    (input_img[0][0]), (1, 2, 0)), interpolation='none')
                im1.set_title('Input Image %s' % (valid_ids[i]))

                im2.imshow(
                    (_img_wall(merge_outputs['input_merge'])), interpolation='none')
                im2.set_title('Variations')
                im2.axis('off')

                im3.imshow(_img_wall(img_vars))
                im3.set_title('Deconv Variations')
                im3.axis('off')

                plt.savefig('image_%s_variations.jpg' %
                            (valid_ids[i]), dpi=300)
                plt.close()

            else:
                for j, channel in enumerate(img_vars):
                    # print 'Shape of single channel in img: ' +
                    # str(np.shape(channel)) debug
                    if norm_single_img:
                        channel -= np.min(channel)
                        channel = channel / np.max(channel)

                    canvas, (im1, im2, im3) = plt.subplots(1, 3)
                    im1.imshow(np.transpose((input_img[0][0]), (1, 2, 0)),
                               interpolation='none')
                    im1.set_title('Input Image %s' % (valid_ids[i]))

                    if wall_variations:
                        im2.imshow(
                            _img_wall(merge_outputs['input_merge']))
                        im2.set_title('Variation %s' % valid_ids[i])
                        im2.axis('off')
                    else:
                        im2.imshow(np.transpose(merge_outputs['input_merge'][j], (
                            1, 2, 0)), interpolation='none',)
                        im2.set_title('Variation %s' % j)

                    if wall_output:
                        if combined_rgb:
                            im3.imshow(_img_wall(img_vars))
                            im3.set_title('Deconv Variations')
                            im3.axis('off')
                        else:
                            im3.imshow(_img_wall(channel), interpolation='none',
                                       cmap=plt.get_cmap('gray'))
                            im3.set_title('Deconv Variations')
                            im3.axis('off')
                    else:
                        if combined_rgb:
                            fpicture = np.transpose((channel),
                                                    (2, 1, 0))
                            im3.imshow(fpicture, interpolation='none')
                            im3.set_title('Deconv Variation %s' % j)
                        else:
                            im3.imshow(_img_wall(channel), interpolation='none',
                                       cmap=plt.get_cmap('gray'))
                            im3.set_title('Deconv Variation %s' % j)
                            im3.axis('off')
                            plt.savefig('image_%s_variation_%s.jpg' %
                                        (valid_ids[i], j), dpi=300)
                            plt.close()

    os.chdir('..')


print_output(nr_images=16, plots=MAKE_PLOTS, combined_rgb=True,
             wall_variations=True, wall_output=WALL_OUTPUT, norm_single_img=True)
save_exit()

print 'Done'
# sys.exit(0)
