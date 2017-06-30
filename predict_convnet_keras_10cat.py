import matplotlib.lines as mlines
import warnings
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
import skimage.io
from termcolor import colored
import functools
from custom_for_keras import sliced_accuracy_mean, sliced_accuracy_std, rmse,\
    lr_function
from ellipse_fit import get_ellipse_kaggle_par
# from custom_keras_model_and_fit_capsels import kaggle_winsol
from custom_keras_model_x_cat import kaggle_x_cat\
    as kaggle_winsol
import skimage

starting_time = time.time()

cut_fraktion = 0.9

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


TRAIN_LOSS_SF_PATH = 'loss_10cat_bw.txt'
# TRAIN_LOSS_SF_PATH = "trainingNmbrs_keras_modular_includeFlip_and_37relu.txt"
# TARGET_PATH = "predictions/final/try_convnet.csv"
WEIGHTS_PATH = "analysis/final/try_10cat_wMaxout_next_next_next_next.h5"
TXT_OUTPUT_PATH = 'try_10cat_bw.txt'
WRONG_CAT_IMGS_PATH = 'wrong_categorized_10cat_bw.json'
IMAGE_OUTPUT_PATH = "img_10cat_bw"


postfix = ''
NUM_ELLIPSE_PARAMS = 2
ELLIPSE_FIT = False
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

# N_Corr_Filter_Images = np.sum(VALID_CORR_OUTPUT_FILTER)

DO_VALIDSTUFF_ON_TRAIN = True

DO_TEST = False  # disable this to not generate predictions on the testset

VALID_CORR_OUTPUT_FILTER = np.ones((10))

DO_TEST = False  # disable this to not generate predictions on the testset


output_names = ['round', 'broad_ellipse', 'small_ellipse', 'edge_no_bulge',
                'edge_bulge', 'disc', 'spiral_1_arm', 'spiral_2_arm',
                'spiral_other', 'other']
question_slices = [slice(0, 10)]

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

y_train = np.load("data/solutions_train_10cat.npy")
y_train_cert = np.load('data/solution_certainties_train_10cat.npy')

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

y_train_cert = y_train_cert[num_train:]

valid_ids = ra.valid_ids
train_ids = ra.train_ids

train_indices = np.arange(num_train
                          )
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

winsol.init_models(final_units=10)

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


print "Set up data loading"

ds_transforms = [
    ra.build_ds_transform(3.0, target_size=input_sizes[0]),
    ra.build_ds_transform(
        3.0, target_size=input_sizes[1])
    + ra.build_augmentation_transform(rotation=45)
]


def tripple_gray(img):
    gray = skimage.color.rgb2gray(img)
    return np.stack((gray, gray, gray), axis=-1)


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

# make to bw
xs_valid = [np.asarray([tripple_gray(x) for x in x_valid])
            for x_valid in xs_valid]

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
    sys.exit(0)


if USE_BLENDED_PREDICTIONS:
    predictions = load_data.load_gz(PRED_BLENDED_PATH)
    if debug:
        print os.path.isfile(PRED_BLENDED_PATH)
        print type(predictions)
        print predictions
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
                [xs_valid[0], xs_valid[1]], y_valid=y_valid, postfix='')
            # validation_data[0], y_valid=y_valid, postfix=postfix)
            winsol.save_loss(modelname='model_norm_metrics',
                             postfix=postfix)
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

print evalHist.keys

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

n_sliced_cat_pred_wcut = [0] * len(output_names)
n_sliced_cat_valid_wcut = [0] * len(output_names)
n_sliced_cat_agrement_wcut = [0] * len(output_names)

wrong_cat_cutted = []

categories = np.zeros((10, 10))
val_l = []
pred_l = []

val_l_cutted = []
pred_l_cutted = []

n_pred_2nd_agree = [0] * len(output_names)
n_pred_3rd_agree = [0] * len(output_names)


def arg_nthmax(arr, n=2):
    arr_ = arr
    for _ in range(n - 1):
        arr_[np.argmax(arr_)] = np.amin(arr_)
    return np.argmax(arr_)


for i in range(len(predictions)):
    argpred = np.argmax(predictions[i])
    argval = np.argmax(y_valid[i])
    n_global_cat_pred[argpred] += 1
    n_global_cat_valid[argval] += 1
    if argval == argpred:
        n_global_cat_agrement[argval] += 1
    elif argval == arg_nthmax(predictions[i]):
        n_pred_2nd_agree[argval] += 1
    elif argval == arg_nthmax(predictions[i], 3):
        n_pred_3rd_agree[argval] += 1

    categories[argval, argpred] += 1.
    val_l.append(argval)
    pred_l.append(argpred)
    c = 0
    for slice in question_slices:
        sargpred = np.argmax(predictions[i][slice])
        cutpred = predictions[i][slice][sargpred] /\
            sum(predictions[i][slice]) > cut_fraktion
        sargval = np.argmax(y_valid[i][slice])
        n_sliced_cat_pred[sargpred + slice.start] += 1

        if cutpred:
            n_sliced_cat_pred_wcut[sargpred + slice.start] += 1
            n_sliced_cat_valid_wcut[sargval + slice.start] += 1
            val_l_cutted.append(argval)
            pred_l_cutted.append(argpred)
            if sargval != sargpred:
                # print '%sto%s' % (str(argval), str(argpred))
                # print valid_ids[i]
                wrong_cat_cutted.append(('%sto%s' % (str(argval),
                                                     str(argpred)),
                                         i))

        n_sliced_cat_valid[sargval + slice.start] += 1

        if sargval == sargpred:
            n_sliced_cat_agrement[sargval + slice.start] += 1
            if cutpred:
                n_sliced_cat_agrement_wcut[sargval + slice.start] += 1
        c += 1

print '\nfirst hit precision: %.3f' % (float(np.sum(n_global_cat_agrement)) / float(np.sum(n_global_cat_pred)))
print 'second hit precision: %.3f' % (float(np.sum(n_pred_2nd_agree)) / float(np.sum(n_global_cat_pred) - np.sum(n_global_cat_agrement)))
print 'third hit precision: %.3f\n' % (float(np.sum(n_pred_3rd_agree)) / float(np.sum(n_global_cat_pred) - np.sum(n_global_cat_agrement) - np.sum(n_pred_2nd_agree)))


def pred_to_val_hist(path=IMAGE_OUTPUT_PATH, also_cutted=True):
    weights_l = []

    for p, v in zip(pred_l, val_l):
        weights_l.append(1. / float(n_global_cat_valid[v]))

    # print categories

    plt.hist2d(pred_l, val_l, bins=10, range=[[0., 10.], [0., 10.]])
    cb = plt.colorbar()
    cb.set_label('# categorised')
    plt.xlabel('predicted category')
    plt.ylabel('validation category')
    plt.xticks([a + 0.5 for a in range(10)], output_names, rotation=90)
    plt.yticks([a + 0.5 for a in range(10)], output_names)
    plt.tight_layout()
    plt.savefig(path + '/categories.eps')
    cb.remove()

    plt.hist2d(pred_l, val_l, bins=10, weights=weights_l,
               range=[[0., 10.], [0., 10.]])
    cb = plt.colorbar()
    cb.set_label('# categorised / # validation in this category')
    plt.xlabel('predicted category')
    plt.ylabel('validation category')
    plt.tight_layout()
    plt.savefig(path + '/categories_normToVal.eps')
    cb.remove()

    weights_l_cutted = []

    for p, v in zip(pred_l_cutted, val_l_cutted):
        weights_l_cutted.append(1. / float(n_sliced_cat_valid_wcut[v]))

        # print categories

    plt.hist2d(pred_l_cutted, val_l_cutted, bins=10,
               range=[[0., 10.], [0., 10.]])
    cb = plt.colorbar()
    cb.set_label('# categorised')
    plt.xlabel('predicted category')
    plt.ylabel('validation category')
    plt.tight_layout()
    plt.savefig(path + '/categories_cutted.eps')
    cb.remove()

    plt.hist2d(pred_l_cutted, val_l_cutted, bins=10, weights=weights_l_cutted,
               range=[[0., 10.], [0., 10.]])
    cb = plt.colorbar()
    cb.set_label('# categorised / # validation in this category')
    plt.xlabel('predicted category')
    plt.ylabel('validation category')
    plt.tight_layout()
    plt.savefig(path + '/categories_normToVal_cutted.eps')
    cb.remove()


def P_base(n_pred, n_agree):
    return (float(n_agree) / float(n_pred))\
        if n_pred else 0.


def P_i(i, n_pred, n_agree):
    return P_base(n_pred[i], n_agree[i])


P = functools.partial(P_i, n_pred=n_sliced_cat_pred,
                      n_agree=n_sliced_cat_agrement)


def R_base(n_pred, n_agree, n_false_neg):
    return float(n_agree) / float(
        n_pred + n_false_neg) if n_pred or n_false_neg else 0.


def R_i(i, sli, n_pred, n_agree):
    if i >= sli.start and i < sli.stop:
        false_neg = sum(n_pred[sli]) - n_pred[i] - (
            sum(n_agree[sli]) - n_agree[i])
        return R_base(n_pred[i], n_agree[i], false_neg)
    else:
        warnings.warn('question number %i is not in slice %s' % (i, sli))


def R_i_slices(i, slices, n_pred, n_agree):
    for sli in slices:
        if i >= sli.start and i < sli.stop:
            return R_i(i, sli, n_pred, n_agree)
        else:
            continue
    else:
        warnings.warn('question number %i is not in one of the slices' % (i))


R = functools.partial(R_i_slices, slices=question_slices,
                      n_pred=n_sliced_cat_pred, n_agree=n_sliced_cat_agrement)


# def R(i):
#     for slice in question_slices:
#         if i >= slice.start and i < slice.stop:
#             false_neg = sum(n_sliced_cat_pred[slice]) - n_sliced_cat_pred[i] - (
#                 sum(n_sliced_cat_agrement[slice]) - n_sliced_cat_agrement[i])
#             return float(n_sliced_cat_agrement[i]) / float(
#                 n_sliced_cat_agrement[i] + false_neg)


def P_wcut(i):
    return (float(n_sliced_cat_agrement_wcut[i]) / float(
        n_sliced_cat_pred_wcut[i])) if n_sliced_cat_pred_wcut[i] else 0.


def R_wcut(i):
    for slice in question_slices:
        if i >= slice.start and i < slice.stop:
            false_neg = sum(n_sliced_cat_pred_wcut[slice]) -\
                n_sliced_cat_pred_wcut[i] - (
                sum(n_sliced_cat_agrement_wcut[slice]) -
                n_sliced_cat_agrement_wcut[i])
            return float(n_sliced_cat_agrement_wcut[i]) / float(
                n_sliced_cat_agrement_wcut[i] + false_neg) if (
                n_sliced_cat_agrement_wcut[i] + false_neg) else 0.


output_dic = {}
output_dic_short_hand_names = {'rmse': 'rmse',
                               'rmse/mean': 'rmse/mean',
                               'slice categorized prediction': 'qPred',
                               'slice categorized valid': 'qVal',
                               'slice categorized agree': 'qAgree',
                               'precision': 'P',
                               'recall': 'R',
                               }

rmse_valid = evalHist['rmse'][-1]
rmse_augmented = np.sqrt(np.mean((y_valid - predictions)**2))
print "  MSE (last iteration):\t%.6f" % float(rmse_valid)
print '  categorical acc. (last iteration):\t%.4f' % float(evalHist['categorical_accuracy'][-1])
print "  MSE (augmented):\t%.6f  RMSE/mean: %.2f " % (float(rmse_augmented),
                                                      float(rmse_augmented) / float(np.mean(
                                                          y_valid)))
print " mean P (augmented):\t%.3f  mean R (augmented):\t%.3f " % (
    np.mean([P(i) for i in range(VALID_CORR_OUTPUT_FILTER.shape[0])]),
    np.mean([R(i) for i in range(VALID_CORR_OUTPUT_FILTER.shape[0])]))
print " mean P (with Cut):\t%.3f  mean R (with Cut):\t%.3f ,\t cut is on %s, mean cut eff. %.2f" % (
    np.mean([P_wcut(i) for i in range(VALID_CORR_OUTPUT_FILTER.shape[0])]),
    np.mean([R_wcut(i) for i in range(VALID_CORR_OUTPUT_FILTER.shape[0])]),
    cut_fraktion,
    np.mean([float(n_sliced_cat_pred_wcut[i]) / float(
        n_sliced_cat_pred[i]) if n_sliced_cat_pred[i] else 0.
        for i in range(VALID_CORR_OUTPUT_FILTER.shape[0])]))

P_wcut_mean_noEmpty = []
for i in range(VALID_CORR_OUTPUT_FILTER.shape[0]):
    if n_sliced_cat_pred_wcut[i]:
        P_wcut_mean_noEmpty.append(P_wcut(i))
P_wcut_mean_noEmpty = np.mean(P_wcut_mean_noEmpty)

R_wcut_mean_noEmpty = []
for i in range(VALID_CORR_OUTPUT_FILTER.shape[0]):
    if n_sliced_cat_pred_wcut[i]:
        R_wcut_mean_noEmpty.append(R_wcut(i))
R_wcut_mean_noEmpty = np.mean(R_wcut_mean_noEmpty)

cut_eff_noEmpty = []
for i in range(VALID_CORR_OUTPUT_FILTER.shape[0]):
    if n_sliced_cat_pred[i]:
        cut_eff_noEmpty.append(float(n_sliced_cat_pred_wcut[i]) / float(
            n_sliced_cat_pred[i]))
cut_eff_noEmpty = np.mean(cut_eff_noEmpty)

print " without zero entry classes:\n mean P (with Cut):\t%.3f  mean R (with Cut):\t%.3f" % (
    P_wcut_mean_noEmpty,
    R_wcut_mean_noEmpty)
print 'mean cut eff, without zero uncuted pred. %.2f' % (cut_eff_noEmpty)
print "  MSE output wise (augmented): P(recision), R(ecall)"

qsc = 0
for i in xrange(0, VALID_CORR_OUTPUT_FILTER.shape[0]):
    oneMSE = np.sqrt(np.mean((y_valid.T[i] - predictions.T[i])**2))
    if not str(qsc) in output_dic.keys():
        output_dic[str(qsc)] = {}
    output_dic[str(qsc)][output_names[i]] = {'rmse': float(oneMSE),
                                             'rmse/mean': float(oneMSE / np.mean(y_valid.T[i])),
                                             'slice categorized prediction': n_sliced_cat_pred[i],
                                             'slice categorized valid': n_sliced_cat_valid[i],
                                             'slice categorized agree': n_sliced_cat_agrement[i],
                                             'precision': P(i),
                                             'recall': R(i),
                                             }
    if i in [slice.start for slice in question_slices]:
        print '----------------------------------------------------'
        qsc += 1
    if P(i) < 0.5:  # oneMSE / np.mean(y_valid.T[i]) > 1.2 * rmse_augmented / np.mean(
           # y_valid):
        print colored("    output % s ( % s): \t%.6f  RMSE/mean: % .2f \t  N sliced pred., valid, agree % i, % i, % i, P % .3f, R % .3f, wCut(eff.%.2f): pred., valid, agree % i, % i, % i, P % .3f, R % .3f" % (
            output_names[i], i, oneMSE, oneMSE /
            np.mean(y_valid.T[i]),
            # n_global_cat_pred[i], n_global_cat_valid[i],
            # n_global_cat_agrement[i],
            n_sliced_cat_pred[i], n_sliced_cat_valid[i], n_sliced_cat_agrement[i],
            P(i), R(i),
            float(n_sliced_cat_pred_wcut[i]) / float(
                n_sliced_cat_pred[i]) if n_sliced_cat_pred[i] else 0.,
            n_sliced_cat_pred_wcut[i], n_sliced_cat_valid_wcut[i],
            n_sliced_cat_agrement_wcut[i],
            P_wcut(i), R_wcut(i)
        ),
            'red')
    elif P(i) > 0.9:  # oneMSE / np.mean(y_valid.T[i]) < 0.8 * rmse_augmented / np.mean(
            # y_valid):
        print colored("    output % s ( % s): \t%.6f  RMSE/mean: % .2f \t N sliced pred., valid, agree % i, % i, % i, P % .3f, R % .3f, wCut(eff.%.2f): pred., valid, agree % i, % i, % i, P % .3f, R % .3f " % (
            output_names[i], i, oneMSE, oneMSE / np.mean(y_valid.T[i]),
            # n_global_cat_pred[i], n_global_cat_valid[i],
            # n_global_cat_agrement[i],
            n_sliced_cat_pred[i], n_sliced_cat_valid[i], n_sliced_cat_agrement[i],
            P(i), R(i),
            float(n_sliced_cat_pred_wcut[i]) / float(
                n_sliced_cat_pred[i]) if n_sliced_cat_pred[i] else 0.,
            n_sliced_cat_pred_wcut[i], n_sliced_cat_valid_wcut[i],
            n_sliced_cat_agrement_wcut[i],
            P_wcut(i), R_wcut(i)
        ),
            'green')
    else:
        print ("    output % s ( % s): \t%.6f  RMSE/mean: % .2f \t  N sliced pred., valid, agree % i, % i, % i, P % .3f, R % .3f, wCut(eff.%.2f): pred., valid, agree % i, % i, % i, P % .3f, R % .3f " %
               (output_names[i], i, oneMSE, oneMSE / np.mean(y_valid.T[i]),
                n_sliced_cat_pred[i], n_sliced_cat_valid[i],
                n_sliced_cat_agrement[i],
                P(i), R(i),
                float(n_sliced_cat_pred_wcut[i]) / float(
                    n_sliced_cat_pred[i]) if n_sliced_cat_pred[i] else 0.,
                n_sliced_cat_pred_wcut[i], n_sliced_cat_valid_wcut[i],
                n_sliced_cat_agrement_wcut[i],
                P_wcut(i), R_wcut(i)
                )
               )

with open(TXT_OUTPUT_PATH, 'a+') as f:
    json.dump(output_dic_short_hand_names, f)
    f.write('\n')
    json.dump(output_dic, f)
    f.write('\n')

imshow_c = functools.partial(
    plt.imshow, interpolation='none')  # , vmin=0.0, vmax=1.0)
imshow_g = functools.partial(
    plt.imshow, interpolation='none', cmap=plt.get_cmap('gray'))  # , vmin=0.0, vmax=1.0)


def try_different_cut_fraktion(cut_fraktions=map(lambda x: float(x) / 20.,
                                                 range(8, 20)),
                               figname='different_cuts.eps'):
    print
    print 'Testing different fraction cuts:'

    cut_fraktions = cut_fraktions

    n_wcut_pred = []
    n_wcut_valid = []
    n_wcut_agree = []

    n_agree_total = []
    n_selected_total = []

    for _ in cut_fraktions:
        n_wcut_pred.append([0] * len(output_names))
        n_wcut_valid.append([0] * len(output_names))
        n_wcut_agree.append([0] * len(output_names))
        n_agree_total.append(0)
        n_selected_total.append(0)

    for i in range(len(predictions)):
        for slic in question_slices:
            sargpred = np.argmax(predictions[i][slic])
            q_frak_pred = predictions[i][slic][sargpred] / \
                sum(predictions[i][slic])
            sargval = np.argmax(y_valid[i][slic])

            for j, cut_val in enumerate(cut_fraktions):
                if q_frak_pred > cut_val:
                    n_wcut_pred[j][sargval + slic.start] += 1
                    n_wcut_valid[j][sargpred + slic.start] += 1
                    n_selected_total[j] = n_selected_total[j] + 1
                    if sargval == sargpred:
                        n_wcut_agree[j][sargval + slic.start] += 1
                        n_agree_total[j] = n_agree_total[j] + 1

    Ps_no_zero = []
    Rs_no_zero = []
    effs = []
    signigicance = []  # agree/sqrt(pred-agree)
    effs_sig = []

    P_total = [float(a) / float(s)
               for a, s in zip(n_agree_total, n_selected_total)]
    signif_total = [float(a) / np.sqrt(s - a)
                    for a, s in zip(n_agree_total, n_selected_total)]

    Ps = [np.mean([P_i(i, param[0], param[1]) for i in range(
        VALID_CORR_OUTPUT_FILTER.shape[0])]) for param in zip(n_wcut_pred,
                                                              n_wcut_agree)]
    Rs = [np.mean([R_i_slices(i, slices=question_slices, n_pred=param[0],
                              n_agree=param[1]) for i in range(
        VALID_CORR_OUTPUT_FILTER.shape[0])])
        for param in zip(n_wcut_pred, n_wcut_agree)]

    if debug:
        print n_sliced_cat_pred[0:3]
        print n_wcut_pred[0][0:3]

    def _ePReS(n_pred, n_agree):
        eff_mean = []
        eff_mean_s = []
        P_wcut_mean_noEmpty = []
        R_wcut_mean_noEmpty = []
        signi = []
        for i in range(VALID_CORR_OUTPUT_FILTER.shape[0]):
            if n_sliced_cat_pred[i]:
                eff_mean.append(float(n_pred[i]) / float(
                    n_wcut_pred[0][i]))
            if n_sliced_cat_agrement[i] and n_wcut_agree[0][i]:
                eff_mean_s.append(float(n_agree[i]) / float(
                    n_wcut_agree[0][i]))
            if n_pred[i]:
                P_wcut_mean_noEmpty.append(P_i(i, n_pred, n_agree))
                R_wcut_mean_noEmpty.append(R_i_slices(
                    i, question_slices, n_pred, n_agree))
                if n_agree[i]:
                    signi.append(
                        float(n_agree[i]) / np.sqrt(float(n_pred[i]
                                                          - n_agree[i])))
        return (np.mean(eff_mean),
                np.mean(P_wcut_mean_noEmpty),
                np.mean(R_wcut_mean_noEmpty),
                np.mean(signi),
                np.mean(eff_mean_s))

    for p, a in zip(n_wcut_pred, n_wcut_agree):
        _e, _P, _R, _s, _es = _ePReS(p, a)
        Ps_no_zero.append(_P)
        Rs_no_zero.append(_R)
        effs.append(_e)
        effs_sig.append(_es)
        signigicance.append(_s)

    if debug:
        print 'cut_fraktions'
        print cut_fraktions
        print 'effs'
        print effs
        print 'effs_sig'
        print effs_sig
        print 'signigicance / 120'
        print [s / 120. for s in signigicance]
        print 'Ps'
        print Ps
        print 'Rs'
        print Rs
        print 'Ps_no_zero'
        print Ps_no_zero
        print 'Rs_no_zero'
        print Rs_no_zero

    plots = []
    label_h = []

    # plt.subplot2grid((1, 1), (0, 0), colspan=1)

    plots.append(plt.plot(cut_fraktions, effs, 'r-', label="eff"))
    label_h.append(mlines.Line2D([], [], color='red', label='eff'))

    plots.append(plt.plot(
        cut_fraktions, effs_sig, 'b-', label="eff sig"))
    label_h.append(mlines.Line2D([], [], color='blue', label='eff sig'))

    # plots.append(plt.plot(cut_fraktions, [
    #              s / 120. for s in signigicance], 'g-', label="signif/120"))

    plots.append(plt.plot(cut_fraktions, [
        s / 250. for s in signif_total], 'g-', label="signif/250"))

    label_h.append(mlines.Line2D([], [], color='green', label='signif/250'))

    plots.append(plt.plot(cut_fraktions, Ps_no_zero, 'r.', label="Ps no zero"))
    label_h.append(mlines.Line2D([], [], color='red', marker='.',
                                 markersize=15, linewidth=0, label='mean P, no 0.'))

    plots.append(plt.plot(cut_fraktions, Rs_no_zero, 'b.', label="Rs no zero"))
    label_h.append(mlines.Line2D([], [], color='blue', marker='.',
                                 markersize=15, linewidth=0, label='mean R, no 0.'))

    # plots.append(plt.plot(cut_fraktions, Ps, 'r.', label="Ps"))
    plots.append(plt.plot(cut_fraktions, P_total, 'ro', label="Ps"))
    label_h.append(mlines.Line2D([], [], color='red', marker='o',
                                 markersize=15, linewidth=0, label='P'))

    # plots.append(plt.plot(cut_fraktions, Rs, 'b.', label="Rs"))
    # label_h.append(mlines.Line2D([], [], color='blue', marker='.',
    #                              markersize=15, linewidth=0, label='R'))

    plt.legend(handles=label_h, loc='lower left')  # , bbox_to_anchor=(
    # 1.05, 1), loc=2, borderaxespad=0.)

    plt.ylabel('Value')
    plt.xlabel('Cut Value')
    # plt.show()

    plt.tight_layout()

    plt.savefig(figname)

    plots = []
    label_h = []

    plt.subplot(121)

    plots.append(plt.plot(Rs_no_zero, Ps_no_zero, 'r-', label="no zero"))
    label_h.append(mlines.Line2D([], [], color='red', label='no zero'))

    plots.append(plt.plot(Rs, Ps, 'b-', label=""))
    label_h.append(mlines.Line2D([], [], color='blue', label='with zero'))

    plt.legend(handles=label_h, bbox_to_anchor=(
        1.05, 1), loc=2, borderaxespad=0.)

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.savefig('ROC_test3.eps')


def pixel_correlations(useTruth=False, dirname='InOutCorr'):
    if not useTruth:
        predict = predictions
        dirname = dirname + '_valid'
    else:
        predict = y_valid
        dirname = dirname + '_truth'

    pixels_color0 = []
    pixels_color1 = []
    pixels_color2 = []

    input_img = validation_data[0][0].transpose(1, 2, 3, 0)
    # if b==0: print input_img.shape
    pixels_color0.append(input_img[0])
    pixels_color1.append(input_img[1])
    pixels_color2.append(input_img[2])

    print "begin correlation calculation"
    pixels_color0_stack = np.dstack(pixels_color0)
    pixels_color1_stack = np.dstack(pixels_color1)
    pixels_color2_stack = np.dstack(pixels_color2)

    if not os.path.isdir(IMAGE_OUTPUT_PATH):
        os.mkdir(IMAGE_OUTPUT_PATH)
    # plt.gray()
    os.chdir(IMAGE_OUTPUT_PATH)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    os.chdir(dirname)

    for i in xrange(0, VALID_CORR_OUTPUT_FILTER.shape[0]):
        if not VALID_CORR_OUTPUT_FILTER[i]:
            continue
        print "begin correlation of output %s" % i
        corr_image_line_c0 = np.zeros(
            input_sizes[0][0] * input_sizes[0][1])
        corr_image_line_c1 = np.zeros(
            input_sizes[0][0] * input_sizes[0][1])
        corr_image_line_c2 = np.zeros(
            input_sizes[0][0] * input_sizes[0][1])
        pixels_colors0_line = np.reshape(
            pixels_color0_stack, (input_sizes[0][0] * input_sizes[0][1],
                                  pixels_color0_stack.shape[2]))
        pixels_colors1_line = np.reshape(
            pixels_color1_stack, (input_sizes[0][0] * input_sizes[0][1],
                                  pixels_color1_stack.shape[2]))
        pixels_colors2_line = np.reshape(
            pixels_color2_stack, (input_sizes[0][0] * input_sizes[0][1],
                                  pixels_color2_stack.shape[2]))

        for j in xrange(0, input_sizes[0][0] * input_sizes[0][1]):
            if j == 0:
                print pixels_colors0_line[j].shape
                print predict.T[i].shape
            corr_image_line_c0[j] = np.corrcoef(
                pixels_colors0_line[j][:predict.shape[0]],
                predict.T[i])[1][0]
            corr_image_line_c1[j] = np.corrcoef(
                pixels_colors1_line[j][:predict.shape[0]],
                predict.T[i])[1][0]
            corr_image_line_c2[j] = np.corrcoef(
                pixels_colors2_line[j][:predict.shape[0]],
                predict.T[i])[1][0]

        # correlation_output_images.append(np.reshape(corr_image_line,(input_sizes[0][0],input_sizes[0][1])))

        # Needs to be in row,col order
        plt.imshow(np.reshape(np.fabs(corr_image_line_c0), (
            input_sizes[0][0], input_sizes[0][1])), interpolation='none',
            vmin=0.0, vmax=0.4)
        plt.colorbar()
        plt.savefig("inputCorrelationToOutput%s%s_c0_Red.jpg" %
                    (i, output_names[i]))
        plt.close()

        skimage.io.imsave("inputCorrelationToOutput%s%s_c0_Red_small.jpg" %
                          (i, output_names[i]), np.reshape(
                              np.fabs(corr_image_line_c0), (
                                  input_sizes[0][0], input_sizes[0][1])) / 0.4)

        # Needs to be in row,col order
        if np.max(np.fabs(np.dstack([corr_image_line_c0,
                                     corr_image_line_c1,
                                     corr_image_line_c2]))) > 0.4:
            print np.max(np.fabs(np.dstack([corr_image_line_c0,
                                            corr_image_line_c1,
                                            corr_image_line_c2])))

        plt.imshow(np.reshape(np.fabs(np.dstack([corr_image_line_c0,
                                                 corr_image_line_c1,
                                                 corr_image_line_c2])) / 0.4, (
            input_sizes[0][0], input_sizes[0][1], 3)),
            interpolation='none',
            vmin=0.0, vmax=0.1)
        # plt.colorbar()
        plt.savefig("inputCorrelationToOutput%s%s_RGB.jpg" %
                    (i, output_names[i]))
        plt.close()

        skimage.io.imsave("inputCorrelationToOutput%s%s_RGB_small.jpg" %
                          (i, output_names[i]), np.reshape(np.fabs(
                              np.dstack([
                                  corr_image_line_c0,
                                  corr_image_line_c1,
                                  corr_image_line_c2])) / 0.4, (
                              input_sizes[0][0],
                              input_sizes[0][1], 3)))

        # # Needs to be in row,col order
        # plt.imshow(np.reshape(corr_image_line_c2, (
        #     input_sizes[0][0], input_sizes[0][1])), interpolation='none', vmin=-0.4, vmax=0.4)
        # plt.colorbar()
        # plt.savefig("inputCorrelationToOutput%s%s_c2_Blue.jpg" %
        #             (i, output_names[i]))
        # plt.close()

    os.chdir("../..")


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


def highest_conv_activation(img_nr=None, img_id=None, layername='conv_0',
                            n_highest=5,
                            order='both', path_base='highest_activations',
                            verbose=1):
    if img_nr and img_id:
        print 'Warning: got image number and id, will use id.'
        img_nr = list(valid_ids).index(img_id)
    elif img_id:
        img_nr = list(valid_ids).index(img_id)
    elif img_nr:
        img_id = valid_ids[img_nr]

    if verbose:
        print 'highest activations in layer %s of image %s (%s)' % (layername,
                                                                    img_id, img_nr)

    input_img = [np.asarray([validation_data[0][0][img_nr]]),
                 np.asarray([validation_data[0][1][img_nr]])]

    save_dic = {}

    if order == 'both' or order == 'global':
        global_max = []
        l_out = np.asarray(winsol.get_layer_output(
            layer=layername, input_=input_img))
        if debug:
            print np.shape(l_out)
        if verbose:
            print '\t global'
        l_out = np.mean(l_out, axis=(2, 3))
        if debug:
            print np.shape(l_out)
        for i in range(n_highest):
            max_ch = np.argmax(l_out)
            val = l_out[0, max_ch]
            l_out[0, max_ch] = 0.
            global_max.append((max_ch, float(val)))
            if verbose:
                print '\t filter %i, with mean activation %.3f'\
                    % global_max[-1]
        save_dic['global'] = global_max

    if order == 'both' or order == 'local':
        local_max = []
        l_out = np.asarray(winsol.get_layer_output(
            layer=layername, input_=input_img))
        if debug:
            print np.shape(l_out)
        if verbose:
            print '\t local:'
        for i in range(n_highest):
            max_ch = np.argmax(l_out[0]) / l_out.shape[2] / l_out.shape[3]
            x = np.argmax(l_out[0, max_ch]) / l_out.shape[3]
            y = np.argmax(l_out[0, max_ch, x])
            val = l_out[0, max_ch, x, y]
            l_out[0, max_ch, x, y] = 0.
            x = float(x) / float(l_out.shape[2])
            y = float(y) / float(l_out.shape[3])
            local_max.append((max_ch, x, y, float(val)))
            if verbose:
                print '\t filter %i at %.2f %.2f, with activation %.3f'\
                    % local_max[-1]
        save_dic['local'] = local_max

    with open(path_base + '_' + str(img_id) + '.json', 'w') as f:
        json.dump(save_dic, f)


def print_filters(image_nr=0, norm=False):
    if not os.path.isdir(IMAGE_OUTPUT_PATH):
        os.mkdir(IMAGE_OUTPUT_PATH)

    print "Print filtered"

    image_nr = image_nr
    if type(image_nr) == int:
        input_img = [np.asarray([validation_data[0][0][image_nr]]),
                     np.asarray([validation_data[0][1][image_nr]])]
    elif image_nr == 'ones':
        input_img = [np.ones(shape=(np.asarray(
            [validation_data[0][0][0]]).shape)), np.ones(
            shape=(np.asarray([validation_data[0][0][0]]).shape))]
    elif image_nr == 'zeros':
        input_img = [np.zeros(shape=(np.asarray(
            [validation_data[0][0][0]]).shape)), np.zeroes(
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
                intermediate_outputs[n], [0] * (board_square - len(
                    intermediate_outputs[n])))
            intermediate_outputs[n] = np.reshape(
                intermediate_outputs[n], (board_side, board_side))

    os.chdir(IMAGE_OUTPUT_PATH)
    intermed_out_dir = 'intermediate_outputs'
    if norm:
        intermed_out_dir += '_norm'
    if not os.path.isdir(intermed_out_dir):
        os.mkdir(intermed_out_dir)
    os.chdir(intermed_out_dir)

    print '  output images will be saved at %s/%s' % (IMAGE_OUTPUT_PATH,
                                                      intermed_out_dir)

    print '  plotting outputs'

    if type(image_nr) == int:
        imshow_c(np.transpose(input_img[0][0], (1, 2, 0)))
        plt.savefig('input_fig_%s_rotation_0.jpg' % (image_nr))
        plt.close()
        skimage.io.imsave('input_fig_%s_rotation_0_small.jpg' % (
            image_nr), np.transpose(input_img[0][0], (1, 2, 0)) /
            np.max(input_img[0][0]))

        imshow_c(np.transpose(input_img[1][0], (1, 2, 0)))
        plt.savefig('input_fig_%s_rotation_45.jpg' % (image_nr))
        plt.close()
        skimage.io.imsave('input_fig_%s_rotation_45_small.jpg' % (
            image_nr), np.transpose(input_img[1][0], (1, 2, 0)) /
            np.max(input_img[1][0]))

        for i in range(len(input_img[0][0])):
            imshow_g(input_img[0][0][i])
            plt.savefig('input_fig_%s_rotation_0_dim_%s.jpg' % (image_nr, i))
            plt.close()
            skimage.io.imsave('input_fig_%s_rotation_0_dim_%s_small.jpg' %
                              (image_nr, i), input_img[0][0][i] /
                              np.max(input_img[0][0][i]))

        for i in range(len(input_img[1][0])):
            imshow_g(input_img[1][0][i])
            plt.savefig('input_fig_%s_rotation_45_dim_%s.jpg' %
                        (image_nr, i))
            plt.close()
            skimage.io.imsave('input_fig_%s_rotation_45_dim_%s_small.jpg' %
                              (image_nr, i), input_img[1][0][i] /
                              np.max(input_img[1][0][i]))

    for n in layer_names:
        if layer_formats[n] > 0:
            imshow_g(_img_wall(intermediate_outputs[n], norm))
            if not norm:
                plt.colorbar()
            plt.savefig('output_fig_%s_%s.jpg' %
                        (image_nr, n))
            plt.close()
            skimage.io.imsave('output_fig_%s_%s_small.jpg' %
                              (image_nr, n), _img_wall(
                                  intermediate_outputs[n], norm) /
                              np.max(_img_wall(
                                  intermediate_outputs[n], norm)))

        else:
            imshow_g(normalize_img(
                intermediate_outputs[n]) if norm else intermediate_outputs[n])
            if not norm:
                plt.colorbar()
            plt.savefig('output_fig_%s_%s.jpg' %
                        (image_nr, n))
            plt.close()
            skimage.io.imsave('output_fig_%s_%s_small.jpg' %
                              (image_nr, n), normalize_img(
                                  intermediate_outputs[n]) if norm
                              else intermediate_outputs[n])

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
            # print type(w[i])
            # print np.shape(w[i])
            # print np.max(w[i])
            # print np.min(w[i])
            skimage.io.imsave(
                'weight_layer_%s_kernel_channel_%s_small.jpg' % (name, i), np.clip(w[i], -1., 1.))

        imshow_g(b)
        if not norm:
            plt.colorbar()
        plt.savefig('weight_layer_%s_bias.jpg' % (name))
        plt.close()
        skimage.io.imsave('weight_layer_%s_bias_small.jpg' %
                          (name), np.clip(b, -1., 1.))

    os.chdir('../..')


def get_best_id(category_name, n=1):
    dtype = []
    dtype.append(('img_nr', int))
    for q in output_names:
        dtype.append((q, float))

    print len(dtype)
    print len(predictions[0])
    print type(predictions[0])
    print len(tuple(np.append(np.array(valid_ids[0]), predictions[0])))

    predictions_dtyped = np.array([], dtype=dtype)

    for id, line in zip(valid_ids, predictions):
        predictions_dtyped = np.append(
            predictions_dtyped, np.asarray(
                tuple(np.append(np.array(id), line)), dtype=dtype))

    return np.sort(predictions_dtyped, order=category_name)['img_nr'][
        -1] if n == 1 else np.sort(predictions_dtyped, order=category_name)[
            'img_nr'][
            -1 - n: len(predictions_dtyped['img_nr'])]


def save_wrong_cat_cutted():
    if not os.path.isdir(IMAGE_OUTPUT_PATH + '/wrong_cat'):
        os.mkdir(IMAGE_OUTPUT_PATH + '/wrong_cat/')
    for i in wrong_cat_cutted:
        plt.imsave(IMAGE_OUTPUT_PATH + '/wrong_cat/' +
                   i[0] + '_' + str(valid_ids[i[1]]) + '.jpg',
                   np.transpose(validation_data[0][0][i[1]], (1, 2, 0)))


# pred_to_val_hist()

# save_wrong_cat_cutted()

# print_weights(norm=True)
# print_weights(norm=False)

# valid_scatter()

# print_filters(2, norm=True)
# print_filters(3, norm=True)

# highest_conv_activation(img_id=get_best_id('RoundCigar'))
# highest_conv_activation(img_id=get_best_id('Spiral2Arm'))
# highest_conv_activation(img_id=get_best_id('Lense'))

# print_filters(list(valid_ids).index(get_best_id('RoundCigar')))
# print_filters(list(valid_ids).index(get_best_id('Spiral2Arm')))
# print_filters(list(valid_ids).index(get_best_id('Lense')))

# print
# print
# print 'RoundCompletly:'
# for id in get_best_id('RoundCompletly', 5):
#     print 'predicted with %.3f' % predictions[list(valid_ids).index(id)][
#         output_names.index('RoundCompletly')]
#     highest_conv_activation(img_id=id)
# print
# print
# print 'Spiral3Arm:'
# for id in get_best_id('Spiral3Arm', 5):
#     print 'predicted with %.3f' % predictions[list(valid_ids).index(id)][
#         output_names.index('Spiral3Arm')]
#     highest_conv_activation(img_id=id)
#     print

# try_different_cut_fraktion(cut_fraktions=map(lambda x: float(
#     x) / 80., range(32, 80)), figname=IMAGE_OUTPUT_PATH + '/10_cat_new.eps')

# pixel_correlations(True)
# pixel_correlations()

# print_weights()
# print_weights(True)

save_exit()
