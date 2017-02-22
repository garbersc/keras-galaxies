import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
# import pandas as pd
import keras.backend as T
import load_data
import realtime_augmentation as ra
import time
import csv
import os
import cPickle as pickle
from datetime import datetime, timedelta

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, MaxPooling1D, Dropout, Input, Convolution2D
from keras.layers.core import Lambda, Flatten, Reshape, Permute
from keras.optimizers import SGD, Adam
from keras.engine.topology import Merge
from keras.callbacks import LearningRateScheduler
from keras import initializations
import functools

from keras_extra_layers import kerasCudaConvnetPooling2DLayer, kerasCudaConvnetConv2DLayer, fPermute
from custom_for_keras import kaggle_MultiRotMergeLayer_output, OptimisedDivGalaxyOutput, kaggle_input, kaggle_sliced_accuracy, dense_weight_init_values

# import matplotlib.pyplot as plt 
# plt.ion()
# import utils

debug = True
predict = False

continueAnalysis = False
saveAtEveryLearningRate = True

getWinSolWeights = True

if getWinSolWeights:
	WINSOL_PATH = "analysis/final/try_convent_gpu1_win_sol_net_on_0p0775_validation.pkl"
	analysis = np.load(WINSOL_PATH)
	l_weights =  analysis['param_values']
	#w_pairs=[]
	#for i in range(len(l_weights)/2):
    	#	w_pairs.append([l_weights[2*i],l_weights[2*i+1]])
	w_kSorted=[]
	for i in range(len(l_weights)/2):
		w_kSorted.append(l_weights[-2-2*i])
		w_kSorted.append(l_weights[-1-2*i])		


CATEGORISED = False
y_train = np.load("data/solutions_train.npy")
if  CATEGORISED: y_train = np.load("data/solutions_train_categorised.npy")
ra.y_train=y_train

# split training data into training + a small validation set
ra.num_train = y_train.shape[0]

ra.num_valid = ra.num_train // 20 # integer division, is defining validation size
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

LEARNING_RATE_SCHEDULE = {  #if adam is used the learning rate doesnt follow the schedule
    0: 0.04,
    200: 0.05,
    400: 0.001,
    800: 0.0005
    #500: 0.04,
    #0: 0.01,
    #1800: 0.004,
    #2300: 0.0004,
   # 0: 0.08,
   # 50: 0.04,
   # 2000: 0.008,
   # 3200: 0.0008,
   # 4600: 0.0004,
}
if continueAnalysis or getWinSolWeights :
    LEARNING_RATE_SCHEDULE = {
        0: 0.0001,	
        500: 0.002,
        #800: 0.0004,
        3200: 0.0002,
        4600: 0.0001,
    }


MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
CHUNK_SIZE = 1008#1008#10000 # 30000 # this should be a multiple of the batch size, ideally.
NUM_CHUNKS = 5#1000#7000 #2500 # 3000 # 1500 # 600 # 600 # 600 # 500   
VALIDATE_EVERY = 5 #20 # 12 # 6 # 6 # 6 # 5 # validate only every 5 chunks. MUST BE A DIVISOR OF NUM_CHUNKS!!!
# else computing the analysis data does not work correctly, since it assumes that the validation set is still loaded.
print("The training is running for %s chunks, each with %s images. That are about %s epochs. The validation sample contains %s images. \n" % (NUM_CHUNKS,CHUNK_SIZE,CHUNK_SIZE*NUM_CHUNKS / (ra.num_train-CHUNK_SIZE) ,  ra.num_valid ))
NUM_CHUNKS_NONORM =  1 # train without normalisation for this many chunks, to get the weights in the right 'zone'.
# this should be only a few, just 1 hopefully suffices.

USE_ADAM = False #TODO not implemented

USE_LLERROR=False #TODO not implemented

USE_WEIGHTS=False #TODO not implemented

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

TRAIN_LOSS_SF_PATH = "trainingNmbrs_keras_start_w_winsolWeights.txt"

#TARGET_PATH = "predictions/final/try_convnet.csv"
ANALYSIS_PATH = "analysis/final/try_convent_keras_start_w_winsolWeights.pkl"

with open(TRAIN_LOSS_SF_PATH, 'a')as f:
	if continueAnalysis: 
		f.write('#continuing from ')
		f.write(ANALYSIS_PATH)
	f.write("#wRandFlip \n")
	f.write("#The training is running for %s chunks, each with %s images. That are about %s epochs. The validation sample contains %s images. \n" % (NUM_CHUNKS,CHUNK_SIZE,CHUNK_SIZE*NUM_CHUNKS / (ra.num_train-CHUNK_SIZE) ,  ra.num_valid ))
	f.write("#round  ,time, mean_train_loss , mean_valid_loss, mean_sliced_accuracy, mean_train_loss_test, mean_accuracy \n")


if continueAnalysis:
    print "Loading model-loss data"
    analysis = np.load(ANALYSIS_PATH)

print "Set up data loading"

input_sizes = [(69, 69), (69, 69)]
PART_SIZE=45

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


N_INPUT_VARIATION=2



print "Build model"

if debug : print("input size: %s x %s x %s x %s" % (input_sizes[0][0],input_sizes[0][1],NUM_INPUT_FEATURES,BATCH_SIZE))

input_tensor = Input(batch_shape=(BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1]) , dtype='float32', name='input_tensor')
input_tensor_45 = Input(batch_shape=(BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1]) , dtype='float32', name='input_tensor_45') 

input_0 = Lambda(lambda x: x,output_shape=(NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1]),batch_input_shape=(BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1]),name='lambda_input_0')
input_45 = Lambda(lambda x: x,output_shape=(NUM_INPUT_FEATURES, input_sizes[1][0], input_sizes[1][1]),batch_input_shape=(BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1]),name='lambda_input_45')

model1 = Sequential()
model1.add(input_0)

model2 = Sequential()
model2.add(input_45)

if debug :print model1.output_shape

model = Sequential()

model.add(Merge([model1, model2], mode=kaggle_input , output_shape=lambda x: (BATCH_SIZE*4*N_INPUT_VARIATION, NUM_INPUT_FEATURES, PART_SIZE, PART_SIZE) , arguments={'part_size':PART_SIZE, 'n_input_var': N_INPUT_VARIATION, 'include_flip':False, 'random_flip':True}  ))

if debug : print model.output_shape

model.add(fPermute((1,2,3,0)))

if debug : print model.output_shape

model.add(kerasCudaConvnetConv2DLayer(n_filters=32, filter_size=6 , untie_biases=True)) 
if debug : print model.output_shape
model.add(kerasCudaConvnetPooling2DLayer())

if debug : print model.output_shape

model.add(kerasCudaConvnetConv2DLayer(n_filters=64, filter_size=5 , untie_biases=True)) 
if debug : print model.output_shape
model.add(kerasCudaConvnetPooling2DLayer())

model.add(kerasCudaConvnetConv2DLayer(n_filters=128, filter_size=3 , untie_biases=True)) 
model.add(kerasCudaConvnetConv2DLayer(n_filters=128, filter_size=3,  weights_std=0.1 , untie_biases=True )) 

if debug : print model.output_shape

model.add(kerasCudaConvnetPooling2DLayer())

if debug : print model.output_shape

model.add(fPermute((3,0,1,2)))

if debug : print model.output_shape

model.add(Lambda(function=kaggle_MultiRotMergeLayer_output, output_shape=lambda x : ( x[0]//4//N_INPUT_VARIATION, (x[1]*x[2]*x[3]*4* N_INPUT_VARIATION) ) , arguments={'num_views':N_INPUT_VARIATION, 'mb_size':16}) ) 

if debug : print model.output_shape

#model.add(Dense(output_dim=4096, init=functools.partial(initializations.normal, scale=0.001) )) 
model.add(Dense(output_dim=4096, weights = dense_weight_init_values(4096,4096) )) 
model.add(Dropout(0.5))
model.add(Reshape((4096,1))) 
model.add(MaxPooling1D())
model.add(Reshape((2048,)))
model.add(Dense(output_dim=4096, weights = dense_weight_init_values(2048,4096) ))

if debug : print model.output_shape

model.add(Dropout(0.5))
model.add(Reshape((4096,1)))
model.add(MaxPooling1D())

if debug : print model.output_shape

model.add(Reshape((2048,)))
model.add(Dense(output_dim=37, weights = dense_weight_init_values(2048,37 ,w_std = 0.01 , b_init_val = 0.1 ) ))
model.add(Dropout(0.5))

if debug : print model.output_shape

model_seq=model([input_tensor,input_tensor_45])

output_layer_norm = Lambda(function=OptimisedDivGalaxyOutput , output_shape=lambda x: x ,arguments={'mb_size': BATCH_SIZE,'normalised':True,'categorised':CATEGORISED})(model_seq)
output_layer_noNorm = Lambda(function=OptimisedDivGalaxyOutput , output_shape=lambda x: x ,arguments={'mb_size': BATCH_SIZE,'normalised':False,'categorised':CATEGORISED})(model_seq)

model_norm=Model(input=[input_tensor,input_tensor_45],output=output_layer_norm)
model_noNorm=Model(input=[input_tensor,input_tensor_45],output=output_layer_noNorm)

if debug : print model_norm.output_shape

if debug : print model_noNorm.output_shape


current_lr=LEARNING_RATE_SCHEDULE[0]
def lr_function(e):
	global current_lr
	if e in LEARNING_RATE_SCHEDULE: 
		_current_lr = LEARNING_RATE_SCHEDULE[e]
		current_lr = _current_lr
	else: _current_lr =  current_lr
	return _current_lr
lr_callback = LearningRateScheduler(lr_function)

if getWinSolWeights:
	w_load_worked = False
	for l in model_norm.layers:
		if debug: print '---'
		if debug: print len(l.get_weights())
		l_weights = l.get_weights()
		if len(l_weights)==len(w_kSorted):
			if debug:
				for i in range(len(l_weights)):
    					print type(l_weights[i])
    					print np.shape(l_weights[i])
					if not np.shape(l_weights[i]) == np.shape(w_kSorted[i]): "somethings wrong with the loaded weight shapes"
			l.set_weights(w_kSorted)
			w_load_worked = True
	if not w_load_worked: print "no matching weight length were found"

model_norm.compile(loss='mean_squared_error', optimizer=SGD(lr=LEARNING_RATE_SCHEDULE[0], momentum=MOMENTUM, nesterov=True), metrics=['categorical_accuracy',kaggle_sliced_accuracy])
model_noNorm.compile(loss='mean_squared_error', optimizer=SGD(lr=LEARNING_RATE_SCHEDULE[0], momentum=MOMENTUM, nesterov=True), metrics=['categorical_accuracy'])

#adam = Adam(lr=LEARNING_RATE_SCHEDULE[0], beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)

#model_norm.compile(loss='mean_squared_error', optimizer=adam, metrics=['categorical_accuracy',kaggle_sliced_accuracy])
#model_noNorm.compile(loss='mean_squared_error', optimizer=adam, metrics=['categorical_accuracy'])

#xs_shared = [T.variable(np.zeros((1,1,1,1), dtype='float32')) for _ in xrange(num_input_representations)] 
#y_shared = T.variable(np.zeros((1,1), dtype='float32'))

xs_shared = [np.zeros((1,1,1,1), dtype='float32') for _ in xrange(num_input_representations)] 
y_shared = np.zeros((1,1), dtype='float32')

if continueAnalysis:#TODO
    print "Load model weights"
    model_norm.load_weights(filepath)

print "Train model"
start_time = time.time()
prev_time = start_time

num_batches_valid = x_valid.shape[0] // BATCH_SIZE
losses_train = []
losses_valid = []


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
    if e >= NUM_CHUNKS_NONORM or continueAnalysis or getWinSolWeights:
        model_inUse = model_norm
        #if  USE_LLERROR:  train = train_norm_ll
    else:
        model_inUse = model_noNorm
        #if  USE_LLERROR:  train = train_nonorm_ll

    num_batches_chunk = x_chunk.shape[0] // BATCH_SIZE

    #update learning rate, use chunks on epoch function #FIXME reanable lr schedule
    #lr_callback.model = model_inUse
    #lr_callback.on_epoch_begin(e)
    #if e in LEARNING_RATE_SCHEDULE: print "  setting learning rate to %.6f" % current_lr

    print "  batch SGD"
    losses = []
    losses_test = []
    #losses_ll = []
    losses_weighted = []
    for b in xrange(num_batches_chunk):

	l0_input_var= xs_chunk[0][b*BATCH_SIZE:(b+1)*BATCH_SIZE]
    	l0_45_input_var= xs_chunk[1][b*BATCH_SIZE:(b+1)*BATCH_SIZE]
	l6_target_var=  y_chunk[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
	
	if ((e + 1) % VALIDATE_EVERY) == 0: loss_test = model_inUse.test_on_batch([l0_input_var,l0_45_input_var], l6_target_var )[0]
	lossaccuracy = model_inUse.train_on_batch( [l0_input_var,l0_45_input_var] , l6_target_var )
	loss = lossaccuracy[0]

        losses.append(loss)
        if ((e + 1) % VALIDATE_EVERY) == 0: losses_test.append(loss_test)

    if e==0: print("Free GPU Mem after training step %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
    
    mean_train_loss = np.sqrt(np.mean(losses))
    if ((e + 1) % VALIDATE_EVERY) == 0: mean_train_loss_test = np.sqrt(np.mean(losses_test))
    #else: mean_train_loss = np.sqrt(np.mean(losses))
	  #mean_train_loss_ll = np.mean(losses_ll)
    print "  mean training loss (RMSE):\t\t%.6f" % mean_train_loss
    if ((e + 1) % VALIDATE_EVERY) == 0:  print "  mean test loss (RMSE):\t\t%.6f" % mean_train_loss_test
    #print "  mean training loss (LL):\t\t%.6f" % mean_train_loss_ll
    losses_train.append(mean_train_loss)


    if ((e + 1) % VALIDATE_EVERY) == 0:
        print
        print "VALIDATING"

        print "  compute losses"
        losses = []
    	accuracies = []
	sliced_accuracies = []
	sliced_accuracies_std = []
	losses_ll = []
	losses_weighted = []
        for b in xrange(num_batches_valid):
            # if b % 1000 == 0:
            #     print "  batch %d/%d" % (b + 1, num_batches_valid)

	    l0_input_var= xs_valid[0][b*BATCH_SIZE:(b+1)*BATCH_SIZE]
    	    l0_45_input_var= xs_valid[1][b*BATCH_SIZE:(b+1)*BATCH_SIZE]
	    l6_target_var=  y_valid[b*BATCH_SIZE:(b+1)*BATCH_SIZE]

	    lossaccuracy = model_norm.test_on_batch([l0_input_var,l0_45_input_var] , l6_target_var)
	    #loss = model_norm.test_on_batch([l0_input_var,l0_45_input_var] , l6_target_var )[0]	

	    loss = lossaccuracy[0]
	    accuracy = lossaccuracy[1]
	    sliced_accuracy = lossaccuracy[2]
	    sliced_accuracy_std = lossaccuracy[3]
	
	    #if b==0 or b==(num_batches_valid-1):    print lossaccuracy
            #loss = compute_loss(b)
	    #print "  loss: %s" % loss
	    #if USE_WEIGHTS:  loss_weighted=compute_loss_weighted(b)
            losses.append(loss)
	    accuracies.append(accuracy)
	    sliced_accuracies.append(sliced_accuracy)
	    sliced_accuracies_std.append(sliced_accuracy_std)
	    #if USE_WEIGHTS:  losses_weighted.append(loss_weighted)
            #loss_ll = compute_loss_ll(b)
	    #print "  loss_ll: %s" % loss_ll
            #losses_ll.append(loss_ll)

    	#if USE_LLERROR: mean_valid_loss = np.mean(losses)
        #else: 
	mean_valid_loss = np.sqrt(np.mean(losses))
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_sliced_accuracy = np.mean(sliced_accuracies)
        mean_sliced_accuracy_std = np.mean(sliced_accuracies_std)
    	#mean_valid_loss_ll = np.mean(losses_ll)


        #if USE_WEIGHTS:  mean_valid_loss_weighted = np.sqrt(np.mean(losses_weighted))

        print "  mean validation loss (RMSE):\t\t%.6f" % mean_valid_loss
        print "  mean validation accuracy:\t\t%.6f" % mean_accuracy
	print "  mean validation sliced accuracy:\t\t%.6f" % mean_sliced_accuracy
        print "  std validation accuracy:\t\t%.6f" % std_accuracy
	print "  mean validation sliced accuracy std:\t\t%.6f" % mean_sliced_accuracy_std
        #print "  mean validation loss (LL):\t\t%.6f" % mean_valid_loss_ll
        if USE_WEIGHTS:  print "  mean weighted validation loss :\t\t%.6f" % mean_valid_loss_weighted
	
	#print 'adam lr: %s' % adam.get_config()['lr']

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
	   else:  f.write(" %s , %s , %s , %s, %s, %s, %s  \n" % (e+1,time_since_start, mean_train_loss,mean_valid_loss,mean_sliced_accuracy, mean_train_loss_test, mean_accuracy))#,mean_valid_loss_ll) )
    else:
    	with open(TRAIN_LOSS_SF_PATH, 'a')as f:
	   if USE_WEIGHTS:  f.write(" %s , %s , %s , 0.0, 0.0 \n" % (e+1,time_since_start, mean_train_loss) )
	   else: f.write(" %s , %s , %s , 0.0 , 0.0, 0.0, 0.0 \n" % (e+1,time_since_start, mean_train_loss) )

    if (not USE_ADAM) and ( (e+1) in LEARNING_RATE_SCHEDULE ):
	if saveAtEveryLearningRate:
  		print("saving parameters after learning rate %s " % (current_lr)) 
		LR_PATH = ((ANALYSIS_PATH.split('.',1)[0]+'toLearningRate%s.'+ANALYSIS_PATH.split('.',1)[1])%current_lr)
		with open(LR_PATH, 'w') as f:
    			pickle.dump({
        			'ids': valid_ids[:num_batches_valid * BATCH_SIZE], # note that we need to truncate the ids to a multiple of the batch size.
       				'targets': y_valid,
        			'mean_train_loss': mean_train_loss,
        			'mean_valid_loss': mean_valid_loss,
        			'time_since_start': time_since_start,
        			'losses_train': losses_train,
        			'losses_valid': losses_valid
    				}, f, pickle.HIGHEST_PROTOCOL)
		LR_PATH = ((ANALYSIS_PATH.split('.',1)[0]+'toLearningRate%s.h5')%current_lr)
		model_inUse.save_weights(LR_PATH)
		
	with open(TRAIN_LOSS_SF_PATH, 'a')as f:
	   f.write("#  setting learning rate to %.6f \n" % current_lr )




print "Compute predictions on validation set for analysis in batches"
predictions_list = []

for b in xrange(num_batches_valid):
    # if b % 1000 == 0:
    #     print "  batch %d/%d" % (b + 1, num_batches_valid)

    l0_input_var= xs_valid[0][b*BATCH_SIZE:(b+1)*BATCH_SIZE]
    l0_45_input_var= xs_valid[1][b*BATCH_SIZE:(b+1)*BATCH_SIZE]

    predictions = model_norm.predict_on_batch([l0_input_var,l0_45_input_var] )
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
        'losses_valid': losses_valid
    }, f, pickle.HIGHEST_PROTOCOL)
LR_PATH = (ANALYSIS_PATH.split('.',1)[0]+'.h5')
model_norm.save_weights(LR_PATH)


del chunk_data, xs_chunk, x_chunk, y_chunk, xs_valid, x_valid # memory cleanup
del predictions_list, all_predictions # memory cleanup




if not predict:
	exit()

'''

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

'''

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
