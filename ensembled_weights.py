# at some point try
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
import load_data
import realtime_augmentation as ra
import time
import sys
import glob
import json
from datetime import timedelta
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from termcolor import colored
import functools


debug = True

weights_dir = 'weight_history'
output_dir = 'img_weight_hist'


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


def print_weights(norm=False):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    weights_history_paths = glob.glob(
        os.path.join(weights_dir, "*.npy")
    )

    if debug:
        print len(weights_history_paths)

    for i, n in enumerate(weights_history_paths):
        if n.find('weights_of_conv_0_') < 0:
            weights_history_paths.remove(n)

    print 'There are %s weight files selected' % len(weights_history_paths)
    if not len(weights_history_paths):
        print weights_history_paths
        raise Warning('No weight files found in ' + weights_dir)

    base_path_l = weights_history_paths[0].split('_')[0:-1]
    base_path = ''
    for s in base_path_l:
        base_path = base_path + s + '_'

    weights_out_dir = 'weights'
    if norm:
        weights_out_dir += '_normalized'
    if not os.path.isdir(weights_out_dir):
        os.mkdir(weights_out_dir)

    print 'Printing weights'

    weight_imgs = []
    bias_imgs = []

    figs = []  # plt.figure()
    axs = []  # fig.add_subplot(111)

    for j in range(0, len(weights_history_paths), 1):
        if j > 126:  # BAD, BAD, BAD
            break
        w, b = np.load(base_path + str(j) + '.npy')
        w = np.transpose(w, (3, 0, 1, 2))

        w = _img_wall(w, norm)
        b = _img_wall(b, norm)

        for i in range(len(w)):
            if j:
                weight_imgs[i].append(
                    (axs[i].imshow(w[i], interpolation='none',
                                   cmap=plt.get_cmap('gray')),
                     axs[i].set_title('')))
            else:
                figs.append(plt.figure())
                axs.append(figs[-1].add_subplot(111))
                weight_imgs.append([(axs[i].imshow(w[i], interpolation='none',
                                                   cmap=plt.get_cmap('gray')),
                                     axs[i].set_title(''))])

        # if not norm:
            # plt.colorbar()

        if not j:
            fig_b = plt.figure()
            ax_b = fig_b.add_subplot(111)

        bias_imgs.append((ax_b.imshow(b, interpolation='none',
                                      cmap=plt.get_cmap('gray')),
                          axs[i].set_title('')))
        # if not norm:
        # plt.colorbar()

    im_ani = []
    print np.shape(weight_imgs)
    print type(weight_imgs[0][0])
    for i, k in enumerate(weight_imgs):
        im_ani.append(animation.ArtistAnimation(figs[i], k, interval=50,
                                                repeat_delay=None,
                                                ))
        out_path = output_dir + '/' + \
            base_path_l[0].split('/')[-1] + 'channel_' + str(i) + '_w.gif'
        im_ani[i].save(out_path, writer='imagemagick')
        print 'saved gif to %s' % out_path

    im_ani_b = animation.ArtistAnimation(fig_b, bias_imgs, interval=50,
                                         repeat_delay=None,
                                         )
    out_path = output_dir + '/' + base_path_l[0].split('/')[-1] + '_b.gif'
    im_ani_b.save(out_path, writer='imagemagick')
    print 'saved gif to %s' % out_path

    # plt.show()


# print_weights(norm=True)
print_weights(norm=False)

print 'Done!'
