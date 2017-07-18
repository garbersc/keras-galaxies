from keras import backend as K
import numpy as np
import time
import sys
from datetime import timedelta
import os
import matplotlib.pyplot as plt
import skimage.io
import functools
from custom_keras_model_x_cat import kaggle_x_cat\
    as kaggle_winsol
import skimage
from skimage.transform import rotate
from keras_extra_layers import fPermute
starting_time = time.time()

cut_fraktion = 0.9

copy_to_ram_beforehand = False

debug = True

get_winsol_weights = False

BATCH_SIZE = 1  # keep in mind

NUM_INPUT_FEATURES = 3

included_flipped = True

WEIGHTS_PATH = "analysis/final/try_10cat_wMaxout_next_next_next_next.h5"
IMAGE_OUTPUT_PATH = "img_10cat_filter_best_2"


postfix = ''

DONT_LOAD_WEIGHTS = False

input_sizes = [(69, 69), (69, 69)]
PART_SIZE = 45

N_INPUT_VARIATION = 2

output_names = ['round', 'broad_ellipse', 'small_ellipse', 'edge_no_bulge',
                'edge_bulge', 'disc', 'spiral_1_arm', 'spiral_2_arm',
                'spiral_other', 'other']
question_slices = [slice(0, 10)]

print 'initiate winsol class'
winsol = kaggle_winsol(BATCH_SIZE=BATCH_SIZE,
                       NUM_INPUT_FEATURES=NUM_INPUT_FEATURES,
                       PART_SIZE=PART_SIZE,
                       input_sizes=input_sizes,
                       LOSS_PATH='./dummy.txt',
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

winsol.init_models(final_units=10, use_dropout=False)

if debug:
    winsol.print_summary(postfix=postfix)
    print winsol.models.keys()

if not DONT_LOAD_WEIGHTS:
    if get_winsol_weights:
        print "Import weights from run with original kaggle winner solution"
        if not winsol.getWinSolWeights(debug=True, path=WEIGHTS_PATH):
            raise UserWarning('Importing of the winsol weights did not work')

    else:
        print "Load model weights"
        winsol.load_weights(path=WEIGHTS_PATH, postfix=postfix)
        winsol.WEIGHTS_PATH = ((WEIGHTS_PATH.split('.', 1)[0] + '_next.h5'))


imshow_c = functools.partial(
    plt.imshow, interpolation='none')  # , vmin=0.0, vmax=1.0)
imshow_g = functools.partial(
    plt.imshow, interpolation='none', cmap=plt.get_cmap('gray'))  # , vmin=0.0, vmax=1.0)


def save_exit():
    # winsol.save()
    print "Done!"
    print ' run for %s' % timedelta(seconds=(time.time() - start_time))
    exit()
    sys.exit(0)


def normalize_img(img):
    min = np.amin(img)
    max = np.amax(img)
    return (img - min) / (max - min)


def _img_wall(img, norm=False):
    dim = len(np.shape(img))
    shape = np.shape(img)
    n_board_side = int(np.ceil(np.sqrt(shape[0])))
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


def print_weights(norm=False):
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
            plt.savefig('weight_layer_%s_kernel_channel_%s.jpg' % (name, i))
            plt.close()
            # print type(w[i])
            # print np.shape(w[i])
            # print np.max(w[i])
            # print np.min(w[i])
            skimage.io.imsave(
                'weight_layer_%s_kernel_channel_%s_small.jpg' % (name, i),
                np.clip(w[i], -1., 1.))

        imshow_g(b)
        if not norm:
            plt.colorbar()
        plt.savefig('weight_layer_%s_bias.jpg' % (name))
        plt.close()
        skimage.io.imsave('weight_layer_%s_bias_small.jpg' %
                          (name), np.clip(b, -1., 1.))

    os.chdir('../..')

# util function to convert a tensor into a valid image


def _deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def find_filter_max_input(layer_name='conv_3', filter_index=0,
                          step=1.):

    print layer_name
    print filter_index

    print winsol.models['model_norm'].get_layer(
        'main_seq').get_layer(layer_name).output_shape

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = winsol.models['model_norm'].get_layer(
        'main_seq').get_layer(layer_name).output

    if layer_name.find('conv_out_merge') >= 0:
        loss = K.mean(layer_output[:, filter_index])
    elif layer_name.find('conv') >= 0:
        loss = K.mean(fPermute((3, 0, 1, 2))(
            layer_output)[:, filter_index, :, :])
    elif layer_name.find('dense') >= 0 or layer_name.find('maxout') >= 0\
            or layer_name.find('dropout') >= 0:
        target = np.zeros((1, 10,), dtype='int8')
        target[:, filter_index] = 1
        # loss = K.mean(-K.categorical_crossentropy(layer_output, target))
        # loss = K.mean(K.sqrt(layer_output - target))

        loss = K.mean(layer_output[:, filter_index] -
                      K.sum(layer_output[:, :filter_index], -1) -
                      K.sum(layer_output[:, filter_index - 9:], -1))
    elif layer_name.find('cuda_out_perm') >= 0:
        loss = K.mean(
            layer_output[:, filter_index, :, :])
    else:
        raise TypeError(
            'Cant find loss definition for the layer %s' % layer_name)

    input_img_0, input_img_45 = winsol.models['model_norm'].get_layer(
        'main_seq').get_input_at(0)

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, [input_img_0, input_img_45])[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img_0, input_img_45], [loss, grads])

    # we start from a gray image with some noise
    input_img_data_0 = np.random.random(
        (1, 3, 141, 141)) * 20 + 128.

    input_img_data_45 = np.reshape(
        [rotate(img, 45) for img in input_img_data_0[0]], (1, 3, 141, 141))

    # gehen wir mal von dem standart quadratischen format aus
    n_pix = input_sizes[0][0]

    input_imgs = [input_img_data_0[:, :, 36:36 + n_pix, 36:36 + n_pix],
                  input_img_data_45[:, :, 36:36 + n_pix, 36:36 + n_pix]]

    # print type(input_imgs)
    # print np.shape(input_imgs)
    # print type(input_imgs[0])
    # print np.shape(input_imgs[0])
    # print type(input_imgs[1])
    # print np.shape(input_imgs[1])

    # run gradient ascent for 20 steps

    for i in range(1000):
        if not i % 200:
            print 'step %s' % i
        try:
            loss_value, grads_value = iterate(input_imgs)
        except IndexError, e:
            print 'step %s' % i
            print e
            break

        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

        if not i % 200:
            print loss_value
        input_imgs[0] += grads_value * step
        input_imgs[1] += grads_value * step

    return (_deprocess_image(input_imgs[0][0]), _deprocess_image(input_imgs[1][0]))

# img = find_filter_max_input('conv_0', 0)
# plt.imshow(np.transpose(img, (1, 2, 0)))
# plt.show()


for i in range(0, 4):
    layer_name = 'conv_%s' % i
    filter_list = []
    filter_list_45 = []
    print
    print layer_name
    for filter_nr in range(winsol.models['model_norm'].get_layer('main_seq')
                           .get_layer(layer_name).n_filters):
        fl_, fl_45 = find_filter_max_input(layer_name, filter_nr)
        filter_list.append(fl_)
        filter_list_45.append(fl_45)
    wall = _img_wall(filter_list)
    plt.imshow(np.transpose(wall, (1, 2, 0)))
    plt.savefig(IMAGE_OUTPUT_PATH + '/' + layer_name + '.jpg')
    wall = _img_wall(filter_list_45)
    plt.imshow(np.transpose(wall, (1, 2, 0)))
    plt.savefig(IMAGE_OUTPUT_PATH + '/' + layer_name + '_45.jpg')

# print
# for layer in winsol.models['model_norm'].get_layer(
#         'main_seq').layers:
#     print layer.name
#     print layer.output_shape
# print

# layer_name = 'dense_output'
# filter_list = []
# filter_list_45 = []
# for filter_nr in range(10):
#     fl_, fl_45 = find_filter_max_input(layer_name, filter_nr)
#     filter_list.append(fl_)
#     filter_list_45.append(fl_45)

# wall = _img_wall(filter_list)
# plt.imshow(np.transpose(wall, (1, 2, 0)))
# plt.savefig(IMAGE_OUTPUT_PATH + '/' + layer_name + '_2.jpg')

# wall = _img_wall(filter_list_45)
# plt.imshow(np.transpose(wall, (1, 2, 0)))
# plt.savefig(IMAGE_OUTPUT_PATH + '/' + layer_name + '_2_45.jpg')

print 'Done!'
