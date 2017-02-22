import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
# import pandas as pd
import theano
import theano.tensor as T
import layers
import cc_layers
import custom
import load_data
import realtime_augmentation as ra
import time
import csv
import os
import cPickle as pickle
from datetime import datetime, timedelta

# import matplotlib.pyplot as plt 
# plt.ion()
# import utils

debug = False
predict = False

continueAnalysis = False
saveAtEveryLearningRate = True

y_train = np.load("data/solutions_train.npy")
#y_train = np.load("data/solutions_train_categorised.npy")
ra.y_train=y_train

# split training data into training + a small validation set
ra.num_train = y_train.shape[0]

ra.num_valid = ra.num_train // 10 # integer division, is defining validation size
ra.num_train -= ra.num_valid

ra.y_valid = ra.y_train[ra.num_train:]
ra.y_train = ra.y_train[:ra.num_train]

load_data.num_train=y_train.shape[0]
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


BATCH_SIZE = 16
NUM_INPUT_FEATURES = 3

LEARNING_RATE_SCHEDULE = {
    0: 0.04,
    1800: 0.004,
    2300: 0.0004,
}

if continueAnalysis:
    LEARNING_RATE_SCHEDULE = {
        0: 0.002,	
        200: 0.004,
        800: 0.0004,
	1500: 0.0001,
        3200: 0.0002,
        4600: 0.0001,
    }




MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
CHUNK_SIZE = 10000 # 30000 # this should be a multiple of the batch size, ideally.
NUM_CHUNKS = 2500 # 3000 # 1500 # 600 # 600 # 600 # 500 
VALIDATE_EVERY = 20 # 12 # 6 # 6 # 6 # 5 # validate only every 5 chunks. MUST BE A DIVISOR OF NUM_CHUNKS!!!
# else computing the analysis data does not work correctly, since it assumes that the validation set is still loaded.
print("The training is running for %s chunks, each with %s images. That are about %s epochs. The validation sample contains %s images. \n" % (NUM_CHUNKS,CHUNK_SIZE,CHUNK_SIZE*NUM_CHUNKS / (ra.num_train-CHUNK_SIZE) ,  ra.num_valid ))
NUM_CHUNKS_NONORM = 1  # train without normalisation for this many chunks, to get the weights in the right 'zone'.
# this should be only a few, just 1 hopefully suffices.

USE_ADAM = False

USE_LLERROR=False

USE_WEIGHTS=False

if USE_LLERROR and USE_WEIGHTS: print 'combination of weighted classes and log loss fuction not implemented yet'

WEIGHTS=np.ones((37))
#WEIGHTS[2]=1  #star or artifact
WEIGHTS[3]=1.5  #edge on yes
WEIGHTS[4]=1.5  #edge on no
#WEIGHTS[5]=1  #bar feature yes
#WEIGHTS[7]=1  #spiral arms yes
#WEIGHTS[14]=1  #anything odd? no
#WEIGHTS[18]=1  #ring
#WEIGHTS[19]=1  #lence
#WEIGHTS[20]=1  #disturbed
#WEIGHTS[21]=1  #irregular
#WEIGHTS[22]=1  #other
#WEIGHTS[23]=1  #merger
#WEIGHTS[24]=1  #dust lane
WEIGHTS=WEIGHTS/WEIGHTS[WEIGHTS.argmax()]

GEN_BUFFER_SIZE = 1

TRAIN_LOSS_SF_PATH = "trainingNmbrs_gpu1_data_in_RAM.txt" 

TARGET_PATH = "predictions/final/try_convnet.csv"
ANALYSIS_PATH = "analysis/final/try_convent_gpu1_data_in_RAM.pkl"

with open(TRAIN_LOSS_SF_PATH, 'a')as f:
	if continueAnalysis: 
		f.write('#continuing from ')
		f.write(ANALYSIS_PATH)
	f.write("#wRandFlip \n")
	f.write("#The training is running for %s chunks, each with %s images. That are about %s epochs. The validation sample contains %s images. \n" % (NUM_CHUNKS,CHUNK_SIZE,CHUNK_SIZE*NUM_CHUNKS / (ra.num_train-CHUNK_SIZE) ,  ra.num_valid ))
	f.write("#round  ,time, mean_train_loss , mean_valid_loss \n")

# # need to load the full training data anyway to extract the validation set from it. 
# # alternatively we could create separate validation set files.
# DATA_TRAIN_PATH = "data/images_train_color_cropped33_singletf.npy.gz"
# DATA2_TRAIN_PATH = "data/images_train_color_8x_singletf.npy.gz"
# DATA_VALIDONLY_PATH = "data/images_validonly_color_cropped33_singletf.npy.gz"
# DATA2_VALIDONLY_PATH = "data/images_validonly_color_8x_singletf.npy.gz"
# DATA_TEST_PATH = "data/images_test_color_cropped33_singletf.npy.gz"
# DATA2_TEST_PATH = "data/images_test_color_8x_singletf.npy.gz"

# FEATURES_PATTERN = "features/try_convnet_chunked_ra_b3sched.%s.npy"

if continueAnalysis:
    print "Loading model data"
    analysis = np.load(ANALYSIS_PATH)

print "Set up data loading"
# TODO: adapt this so it loads the validation data from JPEGs and does the processing realtime

input_sizes = [(69, 69), (69, 69)]

ds_transforms = [
    ra.build_ds_transform(3.0, target_size=input_sizes[0]),
    ra.build_ds_transform(3.0, target_size=input_sizes[1]) + ra.build_augmentation_transform(rotation=45)
    ]

num_input_representations = len(ds_transforms)

augmentation_params = {
    'zoom_range': (1.0 / 1.3, 1.3),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': True,
}

augmented_data_gen = ra.realtime_augmented_data_gen(num_chunks=NUM_CHUNKS, chunk_size=CHUNK_SIZE,
                                                    augmentation_params=augmentation_params, ds_transforms=ds_transforms,
                                                    target_sizes=input_sizes)

post_augmented_data_gen = ra.post_augment_brightness_gen(augmented_data_gen, std=0.5)

train_gen = load_data.buffered_gen_mp(post_augmented_data_gen, buffer_size=GEN_BUFFER_SIZE)






def create_train_gen():
    """
    this generates the training data in order, for postprocessing. Do not use this for actual training.
    """
    data_gen_train = ra.realtime_fixed_augmented_data_gen(train_indices, 'train',
        ds_transforms=ds_transforms, chunk_size=CHUNK_SIZE, target_sizes=input_sizes)
    return load_data.buffered_gen_mp(data_gen_train, buffer_size=GEN_BUFFER_SIZE)


def create_valid_gen():
    data_gen_valid = ra.realtime_fixed_augmented_data_gen(valid_indices, 'train',
        ds_transforms=ds_transforms, chunk_size=CHUNK_SIZE, target_sizes=input_sizes)
    return load_data.buffered_gen_mp(data_gen_valid, buffer_size=GEN_BUFFER_SIZE)


def create_test_gen():
    data_gen_test = ra.realtime_fixed_augmented_data_gen(test_indices, 'test',
        ds_transforms=ds_transforms, chunk_size=CHUNK_SIZE, target_sizes=input_sizes)
    return load_data.buffered_gen_mp(data_gen_test, buffer_size=GEN_BUFFER_SIZE)


print "Preprocess validation data upfront"
start_time = time.time()
xs_valid = [[] for _ in xrange(num_input_representations)]

for data, length in create_valid_gen():
    for x_valid_list, x_chunk in zip(xs_valid, data):
        x_valid_list.append(x_chunk[:length])

xs_valid = [np.vstack(x_valid) for x_valid in xs_valid]
xs_valid = [x_valid.transpose(0, 3, 1, 2) for x_valid in xs_valid] # move the colour dimension up


print "  took %.2f seconds" % (time.time() - start_time)



print "Build model"
if debug : print("input size: %s x %s x %s x %s" % (input_sizes[0][0],input_sizes[0][1],NUM_INPUT_FEATURES,BATCH_SIZE))
l0 = layers.Input2DLayer(BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1])
l0_45 = layers.Input2DLayer(BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[1][0], input_sizes[1][1])

l0r = layers.MultiRotSliceLayer([l0, l0_45], part_size=45, include_flip=True)

#l0r = layers.MultiRotSliceLayer([l0, l0_45], part_size=45, random_flip=True)

l0s = cc_layers.ShuffleBC01ToC01BLayer(l0r) 

l1a = cc_layers.CudaConvnetConv2DLayer(l0s, n_filters=32, filter_size=6, weights_std=0.01, init_bias_value=0.1, dropout=0.0, partial_sum=1, untie_biases=True)
l1 = cc_layers.CudaConvnetPooling2DLayer(l1a, pool_size=2)
#l12 = cc_layers.CudaConvnetPooling2DLayer(l1a, pool_size=4)

#l3a = cc_layers.CudaConvnetConv2DLayer(l12, n_filters=64, filter_size=7, pad=0, weights_std=0.1, init_bias_value=0.1, dropout=0.0, partial_sum=1, untie_biases=True)
#l3 = cc_layers.CudaConvnetPooling2DLayer(l3a, pool_size=2)

l2a = cc_layers.CudaConvnetConv2DLayer(l1, n_filters=64, filter_size=5, weights_std=0.01, init_bias_value=0.1, dropout=0.0, partial_sum=1, untie_biases=True)
l2 = cc_layers.CudaConvnetPooling2DLayer(l2a, pool_size=2)

l3a = cc_layers.CudaConvnetConv2DLayer(l2, n_filters=128, filter_size=3, weights_std=0.01, init_bias_value=0.1, dropout=0.0, partial_sum=1, untie_biases=True)
l3b = cc_layers.CudaConvnetConv2DLayer(l3a, n_filters=128, filter_size=3, pad=0, weights_std=0.1, init_bias_value=0.1, dropout=0.0, partial_sum=1, untie_biases=True)
l3 = cc_layers.CudaConvnetPooling2DLayer(l3b, pool_size=2)

#l3s = cc_layers.ShuffleC01BToBC01Layer(l3)
l3s = cc_layers.ShuffleC01BToBC01Layer(l3)

j3 = layers.MultiRotMergeLayer(l3s, num_views=4) # 2) # merge convolutional parts


l4a = layers.DenseLayer(j3, n_outputs=4096, weights_std=0.001, init_bias_value=0.01, dropout=0.5, nonlinearity=layers.identity)
#l4a = layers.DenseLayer(j3, n_outputs=4096, weights_std=0.001, init_bias_value=0.01)

l4b = layers.FeatureMaxPoolingLayer(l4a, pool_size=2, feature_dim=1, implementation='reshape')
#l4bc = layers.FeatureMaxPoolingLayer(l4a, pool_size=2, feature_dim=1, implementation='reshape')
l4c = layers.DenseLayer(l4b, n_outputs=4096, weights_std=0.001, init_bias_value=0.01, dropout=0.5, nonlinearity=layers.identity)
l4 = layers.FeatureMaxPoolingLayer(l4c, pool_size=2, feature_dim=1, implementation='reshape')

## l5 = layers.DenseLayer(l4, n_outputs=37, weights_std=0.01, init_bias_value=0.0, dropout=0.5, nonlinearity=custom.clip_01) #  nonlinearity=layers.identity)
l5 = layers.DenseLayer(l4, n_outputs=37, weights_std=0.01, init_bias_value=0.1, dropout=0.5, nonlinearity=layers.identity)
#l5 = layers.DenseLayer(l4bc, n_outputs=37, weights_std=0.01, init_bias_value=0.1, nonlinearity=layers.identity)

## l6 = layers.OutputLayer(l5, error_measure='mse')
l6 = custom.OptimisedDivGalaxyOutputLayer(l5,categorised=False) # this incorporates the constraints on the output (probabilities sum to one, weighting, etc.)

if debug : print("output shapes: l0 %s , l0r %s , l1 %s , l3 %s , j3 %s , l4 %s , l5 %s" % ( l0.get_output_shape(), l0r.get_output_shape(), l1.get_output_shape(), l3.get_output_shape(), j3.get_output_shape(), l4.get_output_shape(), l5.get_output_shape()))

#train_loss_nonorm = l6.error(normalisation=False)
#train_loss = l6.error() # but compute and print this!
#valid_loss = l6.error(dropout_active=False)

if USE_WEIGHTS:
    train_loss_nonorm = l6.error_weighted(WEIGHTS,normalisation=False)
    train_loss = l6.error_weighted(WEIGHTS)
    valid_loss_weighted =l6.error_weighted(WEIGHTS,dropout_active=False)
else:
    train_loss_nonorm = l6.error(normalisation=False)
    train_loss = l6.error() # but compute and print this!
valid_loss = l6.error(dropout_active=False)#,normalisation=False) #change this back as soon as possible

if USE_LLERROR:
    train_loss_nonorm = l6.ll_error(normalisation=False)
    train_loss = l6.ll_error() # but compute and print this!
    #valid_loss = l6.ll_error(dropout_active=False)

valid_loss_ll = l6.ll_error(dropout_active=False)#,normalisation=False) #change this back as soon as possible

all_parameters = layers.all_parameters(l6)
all_bias_parameters = layers.all_bias_parameters(l6)

xs_shared = [theano.shared(np.zeros((1,1,1,1), dtype=theano.config.floatX)) for _ in xrange(num_input_representations)] 
y_shared = theano.shared(np.zeros((1,1), dtype=theano.config.floatX))

learning_rate = theano.shared(np.array(LEARNING_RATE_SCHEDULE[0], dtype=theano.config.floatX))

idx = T.lscalar('idx')

givens = {
    l0.input_var: xs_shared[0][idx*BATCH_SIZE:(idx+1)*BATCH_SIZE],
    l0_45.input_var: xs_shared[1][idx*BATCH_SIZE:(idx+1)*BATCH_SIZE],
    l6.target_var: y_shared[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE],
}

# updates = layers.gen_updates(train_loss, all_parameters, learning_rate=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
updates_nonorm = layers.gen_updates_nesterov_momentum_no_bias_decay(train_loss_nonorm, all_parameters, all_bias_parameters, learning_rate=learning_rate, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
updates = layers.gen_updates_nesterov_momentum_no_bias_decay(train_loss, all_parameters, all_bias_parameters, learning_rate=learning_rate, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

if USE_ADAM: 
    updates_nonorm = layers.adam(train_loss_nonorm, all_parameters,learning_rate=learning_rate)
    updates = layers.adam(train_loss, all_parameters,learning_rate=learning_rate)
else:
    updates_nonorm = layers.gen_updates_nesterov_momentum_no_bias_decay(train_loss_nonorm, all_parameters, all_bias_parameters, learning_rate=learning_rate, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    updates = layers.gen_updates_nesterov_momentum_no_bias_decay(train_loss, all_parameters, all_bias_parameters, learning_rate=learning_rate, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

train_nonorm = theano.function([idx], train_loss_nonorm, givens=givens, updates=updates_nonorm)
train_norm = theano.function([idx], train_loss, givens=givens, updates=updates)
compute_loss = theano.function([idx], valid_loss, givens=givens) # dropout_active=False

#train_nonorm_ll = theano.function([idx], train_loss_nonorm_ll, givens=givens)
#train_norm_ll = theano.function([idx], train_loss_ll, givens=givens)
compute_loss_ll = theano.function([idx], valid_loss_ll, givens=givens) # dropout_active=False

if USE_WEIGHTS: 
	compute_loss_weighted = theano.function([idx], valid_loss_weighted, givens=givens)
compute_output = theano.function([idx], l6.predictions(dropout_active=False), givens=givens, on_unused_input='ignore') # not using the labels, so theano complains
compute_features = theano.function([idx], l4.output(dropout_active=False), givens=givens, on_unused_input='ignore')
#compute_features = theano.function([idx], l4bc.output(dropout_active=False), givens=givens, on_unused_input='ignore')

if continueAnalysis:
    print "Load model parameters"
    layers.set_param_values(l6, analysis['param_values'])

print "Train model"
start_time = time.time()
prev_time = start_time

num_batches_valid = x_valid.shape[0] // BATCH_SIZE
losses_train = []
losses_valid = []

param_stds = []

current_lr=LEARNING_RATE_SCHEDULE[0]

with open(TRAIN_LOSS_SF_PATH, 'a')as f:
 f.write("#  starting learning rate to %.6f \n" % current_lr )

for e in xrange(NUM_CHUNKS):
    print "Chunk %d/%d" % (e + 1, NUM_CHUNKS)
    if e==0: print("Free GPU Mem before training step %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
    chunk_data, chunk_length = train_gen.next()
    y_chunk = chunk_data.pop() # last element is labels.
    xs_chunk = chunk_data

    # need to transpose the chunks to move the 'channels' dimension up
    xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

    # train without normalisation for the first # chunks.
    if e >= NUM_CHUNKS_NONORM or continueAnalysis:
        train = train_norm
    else:
        train = train_nonorm

    print "  load training data onto GPU"
    for x_shared, x_chunk in zip(xs_shared, xs_chunk):
        x_shared.set_value(x_chunk)
    y_shared.set_value(y_chunk)
    num_batches_chunk = x_chunk.shape[0] // BATCH_SIZE

    # import pdb; pdb.set_trace()

    print "  batch SGD"
    losses = []
    #losses_ll = []
    losses_weighted = []
    for b in xrange(num_batches_chunk):
        #if b % 1000 == 0:
            #print "  batch %d/%d" % (b + 1, num_batches_chunk)

        loss = train(b)
        losses.append(loss)
        #print "  loss: %s" % loss
        #loss_ll = train_ll(b)
        #losses_ll.append(loss_ll)

    if e==0: print("Free GPU Mem after training step %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
    
    if USE_LLERROR: mean_train_loss = np.mean(losses)
    else: mean_train_loss = np.sqrt(np.mean(losses))
	  #mean_train_loss_ll = np.mean(losses_ll)
    if USE_LLERROR:  print "  crossentropy loss (LL):\t\t%.6f" % mean_train_loss
    else:  print "  mean training loss (RMSE):\t\t%.6f" % mean_train_loss
    #print "  mean training loss (LL):\t\t%.6f" % mean_train_loss_ll
    losses_train.append(mean_train_loss)

    # store param stds during training
    param_stds.append([p.std() for p in layers.get_param_values(l6)])  

    if ((e + 1) % VALIDATE_EVERY) == 0:
        print
        print "VALIDATING"
        print "  load validation data onto GPU"
        for x_shared, x_valid in zip(xs_shared, xs_valid):
            x_shared.set_value(x_valid)
        y_shared.set_value(y_valid)

        print "  compute losses"
        losses = []
	losses_ll = []
	losses_weighted = []
        for b in xrange(num_batches_valid):
            # if b % 1000 == 0:
            #     print "  batch %d/%d" % (b + 1, num_batches_valid)
            loss = compute_loss(b)
	    #print "  loss: %s" % loss
	    if USE_WEIGHTS:  loss_weighted=compute_loss_weighted(b)
            losses.append(loss)
	    if USE_WEIGHTS:  losses_weighted.append(loss_weighted)
            loss_ll = compute_loss_ll(b)
	    #print "  loss_ll: %s" % loss_ll
            losses_ll.append(loss_ll)

    	#if USE_LLERROR: mean_valid_loss = np.mean(losses)
        #else: 
	mean_valid_loss = np.sqrt(np.mean(losses))
    	mean_valid_loss_ll = np.mean(losses_ll)

        if USE_WEIGHTS:  mean_valid_loss_weighted = np.sqrt(np.mean(losses_weighted))

        print "  mean validation loss (RMSE):\t\t%.6f" % mean_valid_loss
        print "  mean validation loss (LL):\t\t%.6f" % mean_valid_loss_ll
        if USE_WEIGHTS:  print "  mean weighted validation loss :\t\t%.6f" % mean_valid_loss_weighted
        losses_valid.append(mean_valid_loss)
        if (e+1)==VALIDATE_EVERY: 
		print("Free GPU Mem after validation step %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
		with open(TRAIN_LOSS_SF_PATH, 'a')as f:
			f.write("#Free GPU Mem after validation step %s MiB \n" % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))

    now = time.time()
    time_since_start = now - start_time
    time_since_prev = now - prev_time
    prev_time = now
    est_time_left = time_since_start * (float(NUM_CHUNKS - (e + 1)) / float(e + 1))
    eta = datetime.now() + timedelta(seconds=est_time_left)
    eta_str = eta.strftime("%c")
    print "  %s since start (%.2f s)" % (load_data.hms(time_since_start), time_since_prev)
    print "  estimated %s to go (ETA: %s)" % (load_data.hms(est_time_left), eta_str)
    print
    if (e > VALIDATE_EVERY):  #lets try do save all losses
	with open(TRAIN_LOSS_SF_PATH, 'a')as f:
	   if USE_WEIGHTS:  f.write(" %s , %s , %s , %s, %s \n" % (e+1,time_since_start, mean_train_loss,mean_valid_loss,mean_valid_loss_weighted) )
	   else:  f.write(" %s , %s , %s , %s, %s \n" % (e+1,time_since_start, mean_train_loss,mean_valid_loss,mean_valid_loss_ll) )
    else:
    	with open(TRAIN_LOSS_SF_PATH, 'a')as f:
	   if USE_WEIGHTS:  f.write(" %s , %s , %s , 0.0, 0.0 \n" % (e+1,time_since_start, mean_train_loss) )
	   else: f.write(" %s , %s , %s , 0.0, 0.0 \n" % (e+1,time_since_start, mean_train_loss) )

    if (not USE_ADAM) and ( (e+1) in LEARNING_RATE_SCHEDULE ):
	if saveAtEveryLearningRate:
  		print("saving parameters after learning rate %s " % (current_lr))
		LR_PATH = ((ANALYSIS_PATH.split('.',1)[0]+'toLearningRate%s.'+ANALYSIS_PATH.split('.',1)[1])%current_lr)
		with open(LR_PATH, 'w') as f:
    			pickle.dump({
        			'ids': valid_ids[:num_batches_valid * BATCH_SIZE], # note that we need to truncate the ids to a multiple of the batch size.
      			 	#'predictions': all_predictions,
       				'targets': y_valid,
        			'mean_train_loss': mean_train_loss,
        			'mean_valid_loss': mean_valid_loss,
        			'time_since_start': time_since_start,
        			'losses_train': losses_train,
        			'losses_valid': losses_valid,
        			'param_values': layers.get_param_values(l6),
        			'param_stds': param_stds,
    				}, f, pickle.HIGHEST_PROTOCOL)

        current_lr = LEARNING_RATE_SCHEDULE[(e+1)]
        learning_rate.set_value(LEARNING_RATE_SCHEDULE[(e+1)])
        print "  setting learning rate to %.6f" % current_lr
	with open(TRAIN_LOSS_SF_PATH, 'a')as f:
	   f.write("#  setting learning rate to %.6f \n" % current_lr )

del chunk_data, xs_chunk, x_chunk, y_chunk, xs_valid, x_valid # memory cleanup


print "Compute predictions on validation set for analysis in batches"
predictions_list = []
for b in xrange(num_batches_valid):
    # if b % 1000 == 0:
    #     print "  batch %d/%d" % (b + 1, num_batches_valid)

    predictions = compute_output(b)
    predictions_list.append(predictions)

all_predictions = np.vstack(predictions_list)

# postprocessing: clip all predictions to 0-1
all_predictions[all_predictions > 1] = 1.0
all_predictions[all_predictions < 0] = 0.0

if continueAnalysis : ANALYSIS_PATH=ANALYSIS_PATH.split('.',1)[0]+'_next.'+ANALYSIS_PATH.split('.',1)[1]
print "Write validation set predictions to %s" % ANALYSIS_PATH
with open(ANALYSIS_PATH, 'w') as f:
    pickle.dump({
        'ids': valid_ids[:num_batches_valid * BATCH_SIZE], # note that we need to truncate the ids to a multiple of the batch size.
        'predictions': all_predictions,
        'targets': y_valid,
        'mean_train_loss': mean_train_loss,
        'mean_valid_loss': mean_valid_loss,
        'time_since_start': time_since_start,
        'losses_train': losses_train,
        'losses_valid': losses_valid,
        'param_values': layers.get_param_values(l6),
        'param_stds': param_stds,
    }, f, pickle.HIGHEST_PROTOCOL)


del predictions_list, all_predictions # memory cleanup


# print "Loading test data"
# x_test = load_data.load_gz(DATA_TEST_PATH)
# x2_test = load_data.load_gz(DATA2_TEST_PATH)
# test_ids = np.load("data/test_ids.npy")
# num_test = x_test.shape[0]
# x_test = x_test.transpose(0, 3, 1, 2) # move the colour dimension up.
# x2_test = x2_test.transpose(0, 3, 1, 2)
# create_test_gen = lambda: load_data.array_chunker_gen([x_test, x2_test], chunk_size=CHUNK_SIZE, loop=False, truncate=False, shuffle=False)


if not predict:
	exit()

print "Computing predictions on test data"
predictions_list = []
for e, (xs_chunk, chunk_length) in enumerate(create_test_gen()):
    print "Chunk %d" % (e + 1)
    xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk] # move the colour dimension up.

    for x_shared, x_chunk in zip(xs_shared, xs_chunk):
        x_shared.set_value(x_chunk)

    num_batches_chunk = int(np.ceil(chunk_length / float(BATCH_SIZE)))  # need to round UP this time to account for all data

    # make predictions for testset, don't forget to cute off the zeros at the end
    for b in xrange(num_batches_chunk):
        # if b % 1000 == 0:
        #     print "  batch %d/%d" % (b + 1, num_batches_chunk)

        predictions = compute_output(b)
        predictions_list.append(predictions)


all_predictions = np.vstack(predictions_list)
all_predictions = all_predictions[:num_test] # truncate back to the correct length

# postprocessing: clip all predictions to 0-1
all_predictions[all_predictions > 1] = 1.0
all_predictions[all_predictions < 0] = 0.0


print "Write predictions to %s" % TARGET_PATH
# test_ids = np.load("data/test_ids.npy")

with open(TARGET_PATH, 'wb') as csvfile:
    writer = csv.writer(csvfile) # , delimiter=',', quoting=csv.QUOTE_MINIMAL)

    # write header
    writer.writerow(['GalaxyID', 'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6'])

    # write data
    for k in xrange(test_ids.shape[0]):
        row = [test_ids[k]] + all_predictions[k].tolist()
        writer.writerow(row)

print "Gzipping..."
os.system("gzip -c %s > %s.gz" % (TARGET_PATH, TARGET_PATH))


del all_predictions, predictions_list, xs_chunk, x_chunk # memory cleanup


# # need to reload training data because it has been split and shuffled.
# # don't need to reload test data
# x_train = load_data.load_gz(DATA_TRAIN_PATH)
# x2_train = load_data.load_gz(DATA2_TRAIN_PATH)
# x_train = x_train.transpose(0, 3, 1, 2) # move the colour dimension up
# x2_train = x2_train.transpose(0, 3, 1, 2)
# train_gen_features = load_data.array_chunker_gen([x_train, x2_train], chunk_size=CHUNK_SIZE, loop=False, truncate=False, shuffle=False)
# test_gen_features = load_data.array_chunker_gen([x_test, x2_test], chunk_size=CHUNK_SIZE, loop=False, truncate=False, shuffle=False)


# for name, gen, num in zip(['train', 'test'], [train_gen_features, test_gen_features], [x_train.shape[0], x_test.shape[0]]):
#     print "Extracting feature representations for all galaxies: %s" % name
#     features_list = []
#     for e, (xs_chunk, chunk_length) in enumerate(gen):
#         print "Chunk %d" % (e + 1)
#         x_chunk, x2_chunk = xs_chunk
#         x_shared.set_value(x_chunk)
#         x2_shared.set_value(x2_chunk)

#         num_batches_chunk = int(np.ceil(chunk_length / float(BATCH_SIZE)))  # need to round UP this time to account for all data

#         # compute features for set, don't forget to cute off the zeros at the end
#         for b in xrange(num_batches_chunk):
#             if b % 1000 == 0:
#                 print "  batch %d/%d" % (b + 1, num_batches_chunk)

#             features = compute_features(b)
#             features_list.append(features)

#     all_features = np.vstack(features_list)
#     all_features = all_features[:num] # truncate back to the correct length

#     features_path = FEATURES_PATTERN % name 
#     print "  write features to %s" % features_path
#     np.save(features_path, all_features)


print "Done!"
