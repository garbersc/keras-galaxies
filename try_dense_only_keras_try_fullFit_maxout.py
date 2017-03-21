import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
# import pandas as pd
import keras.backend as T
import load_data
import realtime_augmentation as ra
import time
import csv
import os
import sys
import cPickle as pickle
from datetime import datetime, timedelta
import json

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, MaxPooling1D, Dropout, Input, Convolution2D, MaxoutDense
from keras.layers.core import Lambda, Flatten, Reshape, Permute
from keras.optimizers import SGD, Adam
from keras.engine.topology import Merge
from keras.callbacks import LearningRateScheduler
from keras import initializations
import functools

from keras_extra_layers import kerasCudaConvnetPooling2DLayer, kerasCudaConvnetConv2DLayer, fPermute
from custom_for_keras import kaggle_MultiRotMergeLayer_output, OptimisedDivGalaxyOutput, kaggle_input, kaggle_sliced_accuracy, dense_weight_init_values, rmse
import custom_for_ev_space as ev

# import matplotlib.pyplot as plt
# plt.ion()
# import utils

debug = True
predict = False

continueAnalysis = False
saveAtEveryValidation = True

getWinSolWeights = False

if getWinSolWeights:
    WINSOL_PATH = "analysis/final/try_convent_gpu1_win_sol_net_on_0p0775_validation.pkl"
    analysis = np.load(WINSOL_PATH)
    l_weights = analysis['param_values']
    # w_pairs=[]
    # for i in range(len(l_weights)/2):
    #	w_pairs.append([l_weights[2*i],l_weights[2*i+1]])
    w_kSorted = []
    for i in range(len(l_weights) / 2):
        w_kSorted.append(l_weights[-2 - 2 * i])
        w_kSorted.append(l_weights[-1 - 2 * i])


CATEGORISED = False
y_train = np.load("data/solutions_train.npy")
if CATEGORISED:
    y_train = np.load("data/solutions_train_categorised.npy")
ra.y_train = y_train

# split training data into training + a small validation set
ra.num_train = y_train.shape[0]

ra.num_valid = ra.num_train // 10  # integer division, is defining validation size
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


BATCH_SIZE = 128  # keep in mind

NUM_INPUT_FEATURES = 3

LEARNING_RATE_SCHEDULE = {  # if adam is used the learning rate doesnt follow the schedule
    0: 0.1,
    20: 0.05,
    40: 0.01,
    80: 0.005
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
if continueAnalysis or getWinSolWeights:
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


MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
# 2000#1008#10000 # 30000 # this should be a multiple of the batch size,
# ideally.
N_TRAIN = num_train
EPOCHS = 100
VALIDATE_EVERY = 10  # 20 # 12 # 6 # 6 # 6 # 5 #
# else computing the analysis data does not work correctly, since it
# assumes that the validation set is still loaded.
print("The training sample contains %s , the validation sample contains %s images. \n" %
      (ra.num_train,  ra.num_valid))
# train without normalisation for this many epochs, to get the weights in
# the right 'zone'.
NUM_EPOCHS_NONORM = 0.1
# this should be only a few, just 1 hopefully suffices.

# FIXME does ist run for part batches one day??? yes!
'''
while (N_TRAIN) % BATCH_SIZE:
	N_TRAIN-=1

if debug: print N_TRAIN
'''

USE_ADAM = True

USE_LLERROR = False  # TODO not implemented

USE_WEIGHTS = False  # TODO not implemented

if USE_LLERROR and USE_WEIGHTS:
    print 'combination of weighted classes and log loss fuction not implemented yet'

WEIGHTS = np.ones((37))
# WEIGHTS[2]=1  #star or artifact
WEIGHTS[3] = 1.5  # edge on yes
WEIGHTS[4] = 1.5  # edge on no
# WEIGHTS[5]=1  #bar feature yes
# WEIGHTS[7]=1  #spiral arms yes
# WEIGHTS[14]=1  #anything odd? no
# WEIGHTS[18]=1  #ring
# WEIGHTS[19]=1  #lence
# WEIGHTS[20]=1  #disturbed
# WEIGHTS[21]=1  #irregular
# WEIGHTS[22]=1  #other
# WEIGHTS[23]=1  #merger
# WEIGHTS[24]=1  #dust lane
WEIGHTS = WEIGHTS / WEIGHTS[WEIGHTS.argmax()]

GEN_BUFFER_SIZE = 1

TRAIN_LOSS_SF_PATH = "trainingNmbrs_keras_ev_adam_3moreDense_sigmoid_OrthInit.txt"

#TARGET_PATH = "predictions/final/try_convnet.csv"
WEIGHTS_PATH = "analysis/final/try_convent_keras_ev_adam_3moreDense_sigmoid_OrthInit.h5"

with open(TRAIN_LOSS_SF_PATH, 'a')as f:
    if continueAnalysis:
        f.write('#continuing from ')
        f.write(WEIGHTS_PATH)
    f.write("#The training is running for %s epochs, each with %s images. The validation sample contains %s images. \n" % (
        EPOCHS, N_TRAIN, ra.num_valid))
    f.write("#round  ,time, mean_train_loss , mean_valid_loss, mean_sliced_accuracy, mean_train_loss_test, mean_accuracy \n")


input_sizes = [(69, 69), (69, 69)]
PART_SIZE = 45

N_INPUT_VARIATION = 2

REDUCTION = 4  # 3000

print 'load translation matrixes'

rot_m = np.load('tests/u_formPcaTest_middled_55420_3.npy')
mid_m = np.load('tests/mean_cropped_train_set.npy')


def ev_translation(x): return ev.rotate(
    ev.middle(ev.flat_galax_imgs(x), mid_m), rot_m, REDUCTION)
#ev_translation = lambda x:  ev.middle( ev.flat_galax_imgs( x ) , mid_m )


print "Build model"

if debug:
    print("input size: %s x %s x %s x %s" %
          (input_sizes[0][0], input_sizes[0][1], NUM_INPUT_FEATURES, BATCH_SIZE))

input_tensor = Input(batch_shape=(BATCH_SIZE, NUM_INPUT_FEATURES,
                                  input_sizes[0][0], input_sizes[0][1]), dtype='float32', name='input_tensor')
input_tensor_45 = Input(batch_shape=(BATCH_SIZE, NUM_INPUT_FEATURES,
                                     input_sizes[0][0], input_sizes[0][1]), dtype='float32', name='input_tensor_45')

input_0 = Lambda(lambda x: x, output_shape=(NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1]), batch_input_shape=(
    BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1]), name='lambda_input_0')
input_45 = Lambda(lambda x: x, output_shape=(NUM_INPUT_FEATURES, input_sizes[1][0], input_sizes[1][1]), batch_input_shape=(
    BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1]), name='lambda_input_45')

model1 = Sequential()
model1.add(input_0)

model2 = Sequential()
model2.add(input_45)

if debug:
    print model1.output_shape
if debug:
    print model2.output_shape

model = Sequential()

model.add(Merge([model1, model2], mode=kaggle_input, output_shape=lambda x: ((model1.output_shape[0] + model2.output_shape[0]) * 2 * N_INPUT_VARIATION,
                                                                             NUM_INPUT_FEATURES, PART_SIZE, PART_SIZE), arguments={'part_size': PART_SIZE, 'n_input_var': N_INPUT_VARIATION, 'include_flip': False, 'random_flip': True}))

if debug:
    print model.output_shape

model.add(Lambda(ev_translation, output_shape=(REDUCTION, )))


if debug:
    print model.output_shape

# newShape=lambda x : (model.output_shape[0]/8,
# 48600)#x[0]*x[1]*x[2]*x[3]//BATCH_SIZE)

# print newShape(model.output_shape)

# merge input variations
model.add(Lambda(lambda x: x.reshape((x.shape[0] / 4 / N_INPUT_VARIATION, x.shape[1] * 4 * N_INPUT_VARIATION)),
                 output_shape=lambda x: (x[0] / 4 / N_INPUT_VARIATION, x[1] * 4 * N_INPUT_VARIATION)))

# needed for the pylearn moduls used by kerasCudaConvnetConv2DLayer and
# kerasCudaConvnetPooling2DLayer


if debug:
    print model.output_shape

# model.add(Dropout(0.5))
# model.add(Dense(output_dim=2048 , activation='sigmoid', weights =
# dense_weight_init_values(model.output_shape[-1],2048,orth_style=True) ))
# #TODO  different output_dim?

# model.add(Dropout(0.5))
#model.add(Dense(output_dim=2048 , activation='sigmoid', weights = dense_weight_init_values(model.output_shape[-1],2048,orth_style=True) ))

# model.add(Dropout(0.5))
#model.add(Dense(output_dim=2048 , activation='sigmoid', weights = dense_weight_init_values(model.output_shape[-1],2048,orth_style=True) ))

if debug:
    print model.output_shape

model.add(Dropout(0.5))
model.add(MaxoutDense(output_dim=2048, nb_feature=2, weights=dense_weight_init_values(
    model.output_shape[-1], 2048, nb_feature=2)))
model.add(Dropout(0.5))

model.add(MaxoutDense(output_dim=2048, nb_feature=2, weights=dense_weight_init_values(
    model.output_shape[-1], 2048, nb_feature=2)))
model.add(Dropout(0.5))

model.add(Dense(output_dim=37, weights=dense_weight_init_values(
    model.output_shape[-1], 37, w_std=0.01, b_init_val=0.1)))

if debug:
    print model.output_shape

model_seq = model([input_tensor, input_tensor_45])

output_layer_norm = Lambda(function=OptimisedDivGalaxyOutput, output_shape=lambda x: x, arguments={
                           'normalised': True, 'categorised': CATEGORISED})(model_seq)
output_layer_noNorm = Lambda(function=OptimisedDivGalaxyOutput, output_shape=lambda x: x, arguments={
                             'normalised': False, 'categorised': CATEGORISED})(model_seq)

model_norm = Model(
    input=[input_tensor, input_tensor_45], output=output_layer_norm)
model_norm_metrics = Model(
    input=[input_tensor, input_tensor_45], output=output_layer_norm)
model_noNorm = Model(
    input=[input_tensor, input_tensor_45], output=output_layer_noNorm)

if debug:
    print model_norm.output_shape

if debug:
    print model_noNorm.output_shape


current_lr = LEARNING_RATE_SCHEDULE[0]


def lr_function(e):
    global current_lr
    if e in LEARNING_RATE_SCHEDULE:
        _current_lr = LEARNING_RATE_SCHEDULE[e]
        current_lr = _current_lr
    else:
        _current_lr = current_lr
    return _current_lr


lr_callback = LearningRateScheduler(lr_function)

if getWinSolWeights:
    w_load_worked = False
    for l in model_norm.layers:
        if debug:
            print '---'
        if debug:
            print len(l.get_weights())
        l_weights = l.get_weights()
        if len(l_weights) == len(w_kSorted):
            if debug:
                for i in range(len(l_weights)):
                    print type(l_weights[i])
                    print np.shape(l_weights[i])
                    if not np.shape(l_weights[i]) == np.shape(w_kSorted[i]):
                        "somethings wrong with the loaded weight shapes"
            l.set_weights(w_kSorted)
            w_load_worked = True
    if not w_load_worked:
        print "no matching weight length were found"
if not USE_ADAM:
    # , metrics=[rmse, 'categorical_accuracy',kaggle_sliced_accuracy])
    model_norm.compile(loss='mean_squared_error', optimizer=SGD(
        lr=LEARNING_RATE_SCHEDULE[0], momentum=MOMENTUM, nesterov=True))
    model_noNorm.compile(loss='mean_squared_error', optimizer=SGD(
        lr=LEARNING_RATE_SCHEDULE[0], momentum=MOMENTUM, nesterov=True))

    model_norm_metrics.compile(loss='mean_squared_error', optimizer=SGD(
        lr=LEARNING_RATE_SCHEDULE[0], momentum=MOMENTUM, nesterov=True), metrics=[rmse, 'categorical_accuracy', kaggle_sliced_accuracy])

else:
    adam = Adam(
        lr=LEARNING_RATE_SCHEDULE[0], beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)

    # , metrics=[rmse, 'categorical_accuracy',kaggle_sliced_accuracy])
    model_norm.compile(loss='mean_squared_error', optimizer=adam)
    model_noNorm.compile(loss='mean_squared_error', optimizer=adam)

    model_norm_metrics.compile(loss='mean_squared_error', optimizer=adam, metrics=[
                               rmse, 'categorical_accuracy', kaggle_sliced_accuracy])

model_norm.summary()

if continueAnalysis:
    print "Load model weights"
    model_norm.load_weights(WEIGHTS_PATH)
    WEIGHTS_PATH = ((WEIGHTS_PATH.split('.', 1)[0] + '_next.h5'))


print "Set up data loading"

ds_transforms = [
    ra.build_ds_transform(3.0, target_size=input_sizes[0]),
    ra.build_ds_transform(
        3.0, target_size=input_sizes[1]) + ra.build_augmentation_transform(rotation=45)
]

num_input_representations = len(ds_transforms)

augmentation_params = {
    'zoom_range': (1.0 / 1.3, 1.3),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': True,
}

augmented_data_gen = ra.realtime_augmented_data_gen(num_chunks=EPOCHS, chunk_size=N_TRAIN,
                                                    augmentation_params=augmentation_params, ds_transforms=ds_transforms,
                                                    target_sizes=input_sizes)

post_augmented_data_gen = ra.post_augment_brightness_gen(
    augmented_data_gen, std=0.5)

train_gen = post_augmented_data_gen
# train_gen = load_data.buffered_gen_mp(post_augmented_data_gen,
# buffer_size=GEN_BUFFER_SIZE)    augmentation buffering will not work
# with the keras .fit

'''
def create_train_gen():
    """
    this generates the training data in order, for postprocessing. Do not use this for actual training.
    """
    data_gen_train = ra.realtime_fixed_augmented_data_gen(train_indices, 'train',
        ds_transforms=ds_transforms, chunk_size=N_TRAIN, target_sizes=input_sizes)
    return load_data.buffered_gen_mp(data_gen_train, buffer_size=GEN_BUFFER_SIZE)
'''


def create_valid_gen():
    data_gen_valid = ra.realtime_fixed_augmented_data_gen(valid_indices, 'train',
                                                          ds_transforms=ds_transforms, chunk_size=N_TRAIN, target_sizes=input_sizes)
    # load_data.buffered_gen_mp(data_gen_valid, buffer_size=GEN_BUFFER_SIZE)
    return data_gen_valid


'''
def create_test_gen():
    data_gen_test = ra.realtime_fixed_augmented_data_gen(test_indices, 'test',
        ds_transforms=ds_transforms, chunk_size=N_TRAIN, target_sizes=input_sizes)
    return load_data.buffered_gen_mp(data_gen_test, buffer_size=GEN_BUFFER_SIZE)
'''

print "Preprocess validation data upfront"
start_time = time.time()
xs_valid = [[] for _ in xrange(num_input_representations)]

for data, length in create_valid_gen():
    for x_valid_list, x_chunk in zip(xs_valid, data):
        x_valid_list.append(x_chunk[:length])

xs_valid = [np.vstack(x_valid) for x_valid in xs_valid]
# move the colour dimension up
xs_valid = [x_valid.transpose(0, 3, 1, 2) for x_valid in xs_valid]

t_val = (time.time() - start_time)
print "  took %.2f seconds" % (t_val)


if debug:
    print("Free GPU Mem before first step %s MiB " %
          (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024. / 1024.))


print ''
print "losses without training on validation sample up front"

evalHistdic = {}
for n in model_norm_metrics.metrics_names:
    evalHistdic[n] = []

evalHist = model_norm_metrics.evaluate(
    x=[xs_valid[0], xs_valid[1]], y=y_valid, batch_size=BATCH_SIZE, verbose=1)


for i in range(len(model_norm_metrics.metrics_names)):
    print "   %s : %.3f" % (model_norm_metrics.metrics_names[i], evalHist[i])
    evalHistdic[model_norm_metrics.metrics_names[i]].append(evalHist[i])


if debug:
    print("Free GPU Mem after validation check %s MiB " %
          (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024. / 1024.))
print ''
print "load and augment data, ETA %.f s" % (t_val * N_TRAIN / num_valid * 2)
start_time = time.time()
chunk_data, chunk_length = train_gen.next()
y_chunk = chunk_data.pop()  # last element is labels.
xs_chunk = chunk_data

# need to transpose the chunks to move the 'channels' dimension up
xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]


l0_input_var = xs_chunk[0]
l0_45_input_var = xs_chunk[1]
l6_target_var = y_chunk

print "  took %.2f seconds" % (time.time() - start_time)

print ''
print "Train %s epoch without norm" % NUM_EPOCHS_NONORM

time1 = time.time()

no_norm_events = int(NUM_EPOCHS_NONORM * N_TRAIN)
hist = model_noNorm.fit(x=[l0_input_var[:no_norm_events], l0_45_input_var[:no_norm_events]], y=l6_target_var[:no_norm_events], validation_data=(
    [xs_valid[0], xs_valid[1]], y_valid), batch_size=BATCH_SIZE, nb_epoch=1, verbose=1, callbacks=[lr_callback])  # loss is squared!!!


hists = hist.history

if debug:
    print("\nFree GPU Mem before train loop %s MiB " %
          (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024. / 1024.))

epochs_run = 0
epoch_togo = EPOCHS

print 'starting main training'
eta = (time.time() - time1) / NUM_EPOCHS_NONORM * EPOCHS
print 'rough ETA %s sec. -> finishes at %s' % (int(eta), datetime.now() + timedelta(seconds=eta))

timedeltas = []

try:
    for i in range(EPOCHS / VALIDATE_EVERY if not EPOCHS % VALIDATE_EVERY else EPOCHS / VALIDATE_EVERY + 1):
        time1 = time.time()
        print ''
        print "epochs run: %s - epochs to go: %s " % (epochs_run, epoch_togo)

        hist = model_norm.fit(x=[l0_input_var, l0_45_input_var], y=l6_target_var, validation_data=([xs_valid[0], xs_valid[1]], y_valid), batch_size=BATCH_SIZE, nb_epoch=np.min(
            [epoch_togo, VALIDATE_EVERY]) + epochs_run, initial_epoch=epochs_run, verbose=1, callbacks=[lr_callback])

        for k in hists:
            hists[k] += hist.history[k]

        epoch_togo -= np.min([epoch_togo, VALIDATE_EVERY])
        epochs_run += np.min([epoch_togo, VALIDATE_EVERY])

        print ''
        print 'validate:'
        evalHist = model_norm_metrics.evaluate(
            x=[xs_valid[0], xs_valid[1]], y=y_valid, batch_size=BATCH_SIZE, verbose=1)
        for i in range(len(model_norm_metrics.metrics_names)):
            print "   %s : %.3f" % (model_norm_metrics.metrics_names[i], evalHist[i])
            evalHistdic[model_norm_metrics.metrics_names[i]].append(
                evalHist[i])

        if saveAtEveryValidation:
            print "saving weights"
            model_norm.save_weights(WEIGHTS_PATH)
        elif ((i + 1) == (EPOCHS / VALIDATE_EVERY if not EPOCHS % VALIDATE_EVERY else EPOCHS / VALIDATE_EVERY + 1)):
            print "saving weights"
            model_norm.save_weights(WEIGHTS_PATH)

        if (i == 0) and debug:
            print("\nFree GPU Mem in train loop %s MiB " % (
                sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024. / 1024.))

        timedeltas.append(time.time() - time1)
        if len(timedeltas) > 10:
            timedeltas = timedeltas[-10:]
        print 'estimated finish: %s' % (datetime.now() + timedelta(seconds=(np.mean(timedeltas) * epoch_togo / VALIDATE_EVERY)))

except KeyboardInterrupt:
    print "\ngot keyboard interuption"
    print "\nsaving weights"
    model_norm.save_weights(WEIGHTS_PATH)
    print 'saving loss and metric history'
    with open(TRAIN_LOSS_SF_PATH, 'a')as f:
        f.write("#after keyboard interuption:\n")
        f.write("#eval losses and metrics:\n")
        f.write(json.dumps(evalHistdic))
        f.write("\n")
        f.write("#fit losses:\n")
        f.write(json.dumps(hists))
        f.write("\n")

    exit()
    sys.exit(0)


with open(TRAIN_LOSS_SF_PATH, 'a')as f:
    f.write("#eval losses and metrics:\n")
    f.write(json.dumps(evalHistdic))
    f.write("\n")
    f.write("#fit losses:\n")
    f.write(json.dumps(hists))
    f.write("\n")


print "Done!"
exit()
sys.exit(0)
