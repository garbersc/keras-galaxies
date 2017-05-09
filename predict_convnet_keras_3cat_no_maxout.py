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
from datetime import timedelta
import os
import matplotlib.pyplot as plt
from termcolor import colored
import functools
from custom_keras_model_x_cat_x_maxout import kaggle_x_cat_x_maxout\
    as kaggle_winsol

starting_time = time.time()

copy_to_ram_beforehand = False

debug = True

get_winsol_weights = False

BATCH_SIZE = 16  # keep in mind

NUM_INPUT_FEATURES = 3

included_flipped = True

USE_BLENDED_PREDICTIONS = False
PRED_BLENDED_PATH = 'predictions/final/blended/blended_predictions.npy.gz'
if debug:
    print os.path.isfile(PRED_BLENDED_PATH)


TRAIN_LOSS_SF_PATH = 'loss_3cat.txt'
# TRAIN_LOSS_SF_PATH = "trainingNmbrs_keras_modular_includeFlip_and_37relu.txt"
# TARGET_PATH = "predictions/final/try_convnet.csv"
WEIGHTS_PATH = "analysis/final/3cat_spiral_ellipse_other.h5"
TXT_OUTPUT_PATH = 'try_3cat.txt'
WRONG_CAT_IMGS_PATH = 'wrong_categorized_3cat.json'
IMAGE_OUTPUT_PATH = "img_3cat"

# load_data.img_path = 'geometry_examples/%s.jpg'


postfix = ''

ELLIPSE_FIT = False
# NUM_ELLIPSE_PARAMS = 2
# ELLIPSE_FIT = WEIGHTS_PATH.find('ellipse') >= 0
# if ELLIPSE_FIT:
#     postfix = '_ellipse'


DONT_LOAD_WEIGHTS = False

input_sizes = [(69, 69), (69, 69)]
PART_SIZE = 45

N_INPUT_VARIATION = 2

# set to True if the prediction and evaluation should be done when the
# prediction file already exists
REPREDICT_EVERYTIME = True

# TODO built this as functions, not with the if's
DO_VALID = True  # disable this to not bother with the validation set evaluation
DO_VALID_CORR = False  # not implemented yet

VALID_CORR_OUTPUT_FILTER = np.zeros((3))

DO_TEST = False  # disable this to not generate predictions on the testset


output_names = ["spiral", 'ellipse', 'other']
question_slices = [slice(0, 3)]

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

y_train = np.load("data/solutions_train_spiral_ellipse_other.npy")
ra.y_train = y_train

# split training data into training + a small validation set
ra.num_train = y_train.shape[0]

# integer division, is defining validation size
ra.num_valid = ra.num_train // 1
ra.num_train -= ra.num_valid

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
                       WEIGHTS_PATH=WEIGHTS_PATH,
                       include_flip=included_flipped,
                       debug=debug)

layer_formats = winsol.layer_formats
layer_names = layer_formats.keys()

print "Build model"

if debug:
    print("input size: %s x %s x %s x %s" %
          (input_sizes[0][0],
           input_sizes[0][1],
           NUM_INPUT_FEATURES,
           BATCH_SIZE))

if ELLIPSE_FIT:
    winsol.init_models_ellipse(input_shape=NUM_ELLIPSE_PARAMS)
else:
    winsol.init_models()

if debug:
    winsol.print_summary(postfix=postfix)

if not DONT_LOAD_WEIGHTS:
    if get_winsol_weights:
        print "Import weights from run with original kaggle winner solution"
        if not winsol.getWinSolWeights(debug=True, path=WEIGHTS_PATH):
            raise UserWarning('Importing of the winsol weights did not work')

    else:
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


if ELLIPSE_FIT:
    validation_data = ([], y_valid)
    for x in xs_valid[0]:
        validation_data[0].append(
            # get_ellipse_kaggle_par(x, num_par=NUM_ELLIPSE_PARAMS)
        )
    validation_data = (np.asarray(validation_data[0]), validation_data[1])
else:
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
    sys.exit(0)


if USE_BLENDED_PREDICTIONS:
    predictions = load_data.load_gz(PRED_BLENDED_PATH)
    if debug:
        print os.path.isfile(PRED_BLENDED_PATH)
        print type(predictions)
        print predictions
        # print len(predictions)
        print np.shape(predictions)
elif not REPREDICT_EVERYTIME and os.path.isfile(
        target_path_valid) and os.path.isfile(TRAIN_LOSS_SF_PATH):
    print 'Loading validation predictions from %s and loss from %s ' % (
        target_path_valid, TRAIN_LOSS_SF_PATH)
    predictions = load_data.load_gz(target_path_valid)
else:
    try:
        print ''
        print 'Re-evalulating and predicting'

        if DO_VALID:
            evalHist = winsol.evaluate(
                validation_data[0], y_valid=y_valid, postfix=postfix)
            winsol.save_loss(modelname='model_norm_metrics', postfix=postfix)
            evalHist = winsol.load_loss(
                modelname='model_norm_metrics', postfix=postfix)

            print ''
            predictions = winsol.predict(
                validation_data[0], postfix=postfix)

            print "Write predictions to %s" % target_path_valid
            load_data.save_gz(target_path_valid, predictions)

    except KeyboardInterrupt:
        print "\ngot keyboard interuption"
        save_exit()
    except ValueError, e:
        print "\ngot value error, could be the end of the generator in the fit"
        print e
        save_exit()

evalHist = winsol.load_loss(modelname='model_norm_metrics', postfix=postfix)

if np.shape(predictions) != np.shape(y_valid):
    raise ValueError('prediction and validation set have different shapes, %s to %s ' % (
        np.shape(predictions), np.shape(y_valid)))

# FIXME add this counts decision tree dependent

n_global_cat_pred = [0] * len(output_names)
n_global_cat_valid = [0] * len(output_names)
n_global_cat_agrement = [0] * len(output_names)

n_sliced_cat_pred = [0] * len(output_names)
n_sliced_cat_valid = [0] * len(output_names)
n_sliced_cat_agrement = [0] * len(output_names)

n_sliced_cat_pred_wreq = [0] * len(output_names)
n_sliced_cat_valid_wreq = [0] * len(output_names)
n_sliced_cat_agrement_wreq = [0] * len(output_names)

wrong_categories = {'0to1': [], '1to0': [], '0to2': [],
                    '2to0': [], '1to2': [], '2to1': []}

for i in range(len(predictions)):
    argpred = np.argmax(predictions[i])
    argval = np.argmax(y_valid[i])
    n_global_cat_pred[argpred] += 1
    n_global_cat_valid[argval] += 1
    if argval == argpred:
        n_global_cat_agrement[argval] += 1
    else:
        if not wrong_categories['%sto%s' % (str(argval), str(argpred))]:
            print '%sto%s' % (str(argval), str(argpred))
            print valid_ids[i]
        wrong_categories['%sto%s' %
                         (str(argval), str(argpred))] += [valid_ids[i]]
        # print 'picture %s' % valid_ids[i]
        # print '\t prediction %s' % argpred
        # print '\t valid %s' % argval

    c = 0
    last_pred = [None]
    last_val = [None]
    for slice in question_slices:
        sargpred = np.argmax(predictions[i][slice])
        sargval = np.argmax(y_valid[i][slice])
        n_sliced_cat_pred[sargpred + slice.start] += 1
        n_sliced_cat_valid[sargval + slice.start] += 1
        if sargval == sargpred:
            n_sliced_cat_agrement[sargval + slice.start] += 1

        if slice == question_slices[0]:
            n_sliced_cat_pred_wreq[sargpred + slice.start] += 1
            n_sliced_cat_valid_wreq[sargval + slice.start] += 1
            last_pred += [sargpred + slice.start]
            last_val += [sargval + slice.start]
            if sargval == sargpred:
                n_sliced_cat_agrement_wreq[sargval + slice.start] += 1
        else:
            sargpred_req = None
            sargval_req = None
            if not np.argmax(predictions[i][0:3]) == 2:
                # if question_requierement[c] in last_pred:
                #     sargpred_req = sargpred
                #     n_sliced_cat_pred_wreq[sargpred + slice.start] += 1
                #     last_pred += [sargpred + slice.start]
                # if question_requierement[c] in last_val:
                #     sargval_req = sargval
                #     n_sliced_cat_valid_wreq[sargval + slice.start] += 1
                #     last_val += [sargval + slice.start]
                if sargpred_req == sargval_req and sargpred_req != None:
                    n_sliced_cat_agrement_wreq[sargval_req + slice.start] += 1
        c += 1


def P(i):
    return (float(n_sliced_cat_agrement[i]) / float(n_sliced_cat_pred[i]))\
        if n_sliced_cat_pred[i] else 0.


def R(i):
    for slice in question_slices:
        if i >= slice.start and i < slice.stop:
            false_neg = sum(n_sliced_cat_pred[slice]) - n_sliced_cat_pred[i]\
                - (sum(n_sliced_cat_agrement[slice])
                   - n_sliced_cat_agrement[i])
            return float(n_sliced_cat_agrement[i]) / float(
                n_sliced_cat_agrement[i] + false_neg)


output_dic = {}
output_dic_short_hand_names = {'rmse': 'rmse',
                               'rmse/mean': 'rmse/mean',
                               # 'global categorized prediction': 'pred',
                               # 'global categorized valid': 'val',
                               # 'global categorized agree': 'agree',
                               'slice categorized prediction': 'qPred',
                               'slice categorized valid': 'qVal',
                               'slice categorized agree': 'qAgree',
                               'precision': 'P',
                               'recall': 'R',
                               # 'slice categorized prediction including tree requierement': 'qPred_req',
                               # 'slice categorized valid including tree requieremnet': 'qVal_req',
                               # 'slice categorized agree including tree requirement': 'qAgree_req',
                               # 'precision including tree requierement': 'P_req',
                               # 'recall including tree requierement': 'R_req'
                               }

rmse_valid = evalHist['rmse'][-1]
rmse_augmented = np.sqrt(np.mean((y_valid - predictions)**2))
print "  MSE (last iteration):\t%.6f" % rmse_valid
# print '  sliced acc. (last iteration):\t%.4f' %
# evalHist['sliced_accuracy_mean'][-1]
print '  categorical acc. (last iteration):\t%.4f'\
    % evalHist['categorical_accuracy'][-1]
print "  MSE (augmented):\t%.6f  RMSE/mean: %.2f " % (rmse_augmented,
                                                      rmse_augmented / np.mean(
                                                          y_valid))
print " mean P (augmented):\t%.3f  mean R (augmented):\t%.3f " % (
    np.mean([P(i) for i in range(VALID_CORR_OUTPUT_FILTER.shape[0])]),
    np.mean([R(i) for i in range(VALID_CORR_OUTPUT_FILTER.shape[0])]))
print "  MSE output wise (augmented): P(recision), R(ecall)"

qsc = 0
for i in xrange(0, VALID_CORR_OUTPUT_FILTER.shape[0]):
    oneMSE = np.sqrt(np.mean((y_valid.T[i] - predictions.T[i])**2))
    if not str(qsc) in output_dic.keys():
        output_dic[str(qsc)] = {}
    output_dic[str(qsc)][output_names[i]] = {'rmse': float(oneMSE),
                                             'rmse/mean': float(oneMSE / np.mean(y_valid.T[i])),
                                             # 'global categorized prediction': n_global_cat_pred[i],
                                             # 'global categorized valid': n_global_cat_valid[i],
                                             # 'global categorized agree': n_global_cat_agrement[i],
                                             'slice categorized prediction': n_sliced_cat_pred[i],
                                             'slice categorized valid': n_sliced_cat_valid[i],
                                             'slice categorized agree': n_sliced_cat_agrement[i],
                                             'precision': P(i),
                                             'recall': R(i),
                                             # 'slice categorized prediction including tree requierement': n_sliced_cat_pred_wreq[i],
                                             # 'slice categorized valid including tree requieremnet': n_sliced_cat_valid_wreq[i],
                                             # 'slice categorized agree including tree requirement': n_sliced_cat_agrement_wreq[i],
                                             # 'precision including tree requierement': P_wreq(i),
                                             # 'recall including tree requierement': R_wreq(i)
                                             }
    if i in [slice.start for slice in question_slices]:
        print '----------------------------------------------------'
        qsc += 1
    if P(i) < 0.5:  # oneMSE / np.mean(y_valid.T[i]) > 1.2 * rmse_augmented / np.mean(
           # y_valid):
        print colored("    output % s ( % s): \t%.6f  RMSE/mean: % .2f \t  N sliced pred., valid, agree % i, % i, % i, P % .3f, R % .3f" % (
            output_names[i], i, oneMSE, oneMSE /
            np.mean(y_valid.T[i]),
            # n_global_cat_pred[i], n_global_cat_valid[i],
            # n_global_cat_agrement[i],
            n_sliced_cat_pred[i], n_sliced_cat_valid[i], n_sliced_cat_agrement[i],
            P(i), R(i),
            # n_sliced_cat_pred_wreq[i], n_sliced_cat_valid_wreq[i],
            # n_sliced_cat_agrement_wreq[i],
            # P_wreq(i), R_wreq(i)
        ),
            'red')
    elif P(i) > 0.9:  # oneMSE / np.mean(y_valid.T[i]) < 0.8 * rmse_augmented / np.mean(
            # y_valid):
        print colored("    output % s ( % s): \t%.6f  RMSE/mean: % .2f \t N sliced pred., valid, agree % i, % i, % i, P % .3f, R % .3f " % (
            output_names[i], i, oneMSE, oneMSE / np.mean(y_valid.T[i]),
            # n_global_cat_pred[i], n_global_cat_valid[i],
            # n_global_cat_agrement[i],
            n_sliced_cat_pred[i], n_sliced_cat_valid[i], n_sliced_cat_agrement[i],
            P(i), R(i),
            # n_sliced_cat_pred_wreq[i], n_sliced_cat_valid_wreq[i],
            # n_sliced_cat_agrement_wreq[i],
            # P_wreq(i), R_wreq(i)
        ),
            'green')
    else:
        print ("    output % s ( % s): \t%.6f  RMSE/mean: % .2f \t  N sliced pred., valid, agree % i, % i, % i, P % .3f, R % .3f " %
               (output_names[i], i, oneMSE, oneMSE / np.mean(y_valid.T[i]),
                # n_global_cat_pred[i], n_global_cat_valid[i],
                # n_global_cat_agrement[i],
                n_sliced_cat_pred[i], n_sliced_cat_valid[i],
                n_sliced_cat_agrement[i],
                P(i), R(i),
                # n_sliced_cat_pred_wreq[i], n_sliced_cat_valid_wreq[i],
                # n_sliced_cat_agrement_wreq[i],
                # P_wreq(i), R_wreq(i)
                )
               )


print 'wrong categorisations:'
for k in wrong_categories:
    print '\t %s: %.3f' % (k, float(len(wrong_categories[k])) /
                           float(len(predictions)))

with open(WRONG_CAT_IMGS_PATH, 'a') as f:
    json.dump(wrong_categories, f)

with open(TXT_OUTPUT_PATH, 'a+') as f:
    json.dump(output_dic_short_hand_names, f)
    f.write('\n')
    json.dump(output_dic, f)
    f.write('\n')

imshow_c = functools.partial(
    plt.imshow, interpolation='none')  # , vmin=0.0, vmax=1.0)
imshow_g = functools.partial(
    plt.imshow, interpolation='none', cmap=plt.get_cmap('gray'))  # , vmin=0.0, vmax=1.0)


def x_category_precision(predictions=predictions, y_valid=y_valid,
                         conditions=[]):
    counts = [[0, 0, 0]]
    for _ in conditions:
        counts += [[0, 0, 0]]
    # if debug:
    #     print np.shape(counts)
    for i, p in enumerate(predictions):
        predicted = [True] * (len(conditions) + 1)
        valid = [True] * (len(conditions) + 1)
        for j, cond in enumerate(conditions):
            for sup_cond in cond:
                predicted[j] *= (np.argmax(
                    p[question_slices[sup_cond[0]]]) == sup_cond[1])
                valid[j] *= (np.argmax(
                    y_valid[i][question_slices[sup_cond[0]]]) == sup_cond[1])
            if predicted[j]:
                counts[j][0] += 1
            if valid[j]:
                counts[j][1] += 1
            if predicted[j] and valid[j]:
                counts[j][2] += 1
        predicted[-1] = (np.sum(predicted[0:-1]) == 0)
        valid[-1] = (np.sum(valid[0:-1]) == 0)
        if predicted[-1]:
            counts[-1][0] += 1
        if valid[-1]:
            counts[-1][1] += 1
        if predicted[-1] and valid[-1]:
            counts[-1][2] += 1
        if np.sum(predicted) != 1 or np.sum(valid) != 1:
            raise UserWarning(
                'conditions in x_category_precision were not exclusive for image %s') % i
        # if debug and not i % 1000:
        #     print predicted
        #     print valid
        #     print counts

    # if debug:
    #     print counts
    P_s = [(float(c[2]) / c[0]) for c in counts]
    R_s = [(float(c[2]) / (2 * c[2] + sum(
        [d[0] - d[2] for d in counts]) - c[0]))for c in counts]
    print
    print 'mean P:\t %.3f' % np.mean(P_s)
    print 'mean R:\t %.3f' % np.mean(R_s)
    for i, c in enumerate(counts):
        print 'condition %s: \t pred,val,agree: %.3f \t P: %.3f R: %.3f' % (
            i, c, P_s[i], R_s[i])


def valid_scatter():
    print 'Do scatter plots'
    print '  they will be saved in the folder %s ' % IMAGE_OUTPUT_PATH
    if not os.path.isdir(IMAGE_OUTPUT_PATH):
        os.mkdir(IMAGE_OUTPUT_PATH)
    # plt.gray()
    os.chdir(IMAGE_OUTPUT_PATH)
    if not os.path.isdir("ValidScatter"):
        os.mkdir("ValidScatter")
    os.chdir("ValidScatter")

    if debug:
        print output_names

    for i in xrange(0, len(output_names)):
        y = predictions.T[i]
        x = y_valid.T[i]
        fig, ax = plt.subplots()
        fit = np.polyfit(x, y, deg=1)
        ax.plot(x, fit[0] * x + fit[1], color='red')
        ax.scatter(x, y, s=0.1)
        plt.ylabel('prediction')
        plt.xlabel('target')
        plt.title("valid %s" % (output_names[i]))
        oneMSE = np.sqrt(np.mean((y_valid.T[i] - predictions.T[i])**2))
        plt.text(60, .025, 'RMSE: %s , RMSE/mean: %s ' %
                 (oneMSE, oneMSE / np.mean(y_valid.T[i])))
        plt.savefig("validScatter_%s_%s.jpg" % (i, output_names[i]))
        plt.close()

    os.chdir("../..")


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


def print_filters(image_nr=0, norm=False):
    if not os.path.isdir(IMAGE_OUTPUT_PATH):
        os.mkdir(IMAGE_OUTPUT_PATH)

    print "Print filtered"

    image_nr = image_nr
    if type(image_nr) == int:
        input_img = [np.asarray([validation_data[0][0][image_nr]]),
                     np.asarray([validation_data[0][1][image_nr]])]
    elif image_nr == 'ones':
        input_img = [np.ones(shape=(np.asarray([validation_data[0][0][0]]).shape)), np.ones(
            shape=(np.asarray([validation_data[0][0][0]]).shape))]
    elif image_nr == 'zeros':
        input_img = [np.zeros(shape=(np.asarray([validation_data[0][0][0]]).shape)), np.zeroes(
            shape=(np.asarray([validation_data[0][0][0]]).shape))]

    print '  getting outputs'

    intermediate_outputs = {}
    for n in layer_names:
        intermediate_outputs[n] = np.asarray(winsol.get_layer_output(
            n, input_=input_img))
        intermediate_outputs[n] = intermediate_outputs[n][0]
        if layer_formats[n] <= 0:
            board_side = int(np.ceil(np.sqrt(len(intermediate_outputs[n]))))
            board_square = int(board_side**2)
            intermediate_outputs[n] = np.append(
                intermediate_outputs[n], [0] * (board_square - len(intermediate_outputs[n])))
            intermediate_outputs[n] = np.reshape(
                intermediate_outputs[n], (board_side, board_side))

    os.chdir(IMAGE_OUTPUT_PATH)
    intermed_out_dir = 'intermediate_outputs'
    if norm:
        intermed_out_dir += '_norm'
    if not os.path.isdir(intermed_out_dir):
        os.mkdir(intermed_out_dir)
    os.chdir(intermed_out_dir)

    print '  output images will be saved at %s/%s' % (IMAGE_OUTPUT_PATH, intermed_out_dir)

    print '  plotting outputs'

    if type(image_nr) == int:
        imshow_c(np.transpose(input_img[0][0], (1, 2, 0)))
        plt.savefig('input_fig_%s_rotation_0.jpg' % (image_nr))
        plt.close()

        imshow_c(np.transpose(input_img[1][0], (1, 2, 0)))
        plt.savefig('input_fig_%s_rotation_45.jpg' % (image_nr))
        plt.close()

        for i in range(len(input_img[0][0])):
            imshow_g(input_img[0][0][i])
            plt.savefig('input_fig_%s_rotation_0_dim_%s.jpg' % (image_nr, i))
            plt.close()

        for i in range(len(input_img[1][0])):
            imshow_g(input_img[1][0][i])
            plt.savefig('input_fig_%s_rotation_45_dim_%s.jpg' %
                        (image_nr, i))
            plt.close()

    for n in layer_names:
        if layer_formats[n] > 0:
            imshow_g(_img_wall(intermediate_outputs[n], norm))
            if not norm:
                plt.colorbar()
            plt.savefig('output_fig_%s_%s.jpg' %
                        (image_nr, n))
            plt.close()
        else:
            imshow_g(normalize_img(
                intermediate_outputs[n]) if norm else intermediate_outputs[n])
            if not norm:
                plt.colorbar()
            plt.savefig('output_fig_%s_%s.jpg' %
                        (image_nr, n))
            plt.close()
    os.chdir('../..')


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

        imshow_g(b)
        if not norm:
            plt.colorbar()
        plt.savefig('weight_layer_%s_bias.jpg' % (name))
        plt.close()

    os.chdir('../..')


# x_category_precision(predictions=predictions, y_valid=y_valid)
# print_weights(norm=True)
# print_weights(norm=False)
print_filters(0, norm=True)
print_filters(1, norm=True)
print_filters(4, norm=True)
print_filters(2, norm=True)
print_filters(3, norm=True)
print_filters(5, norm=True)

valid_scatter()

save_exit()
