"""
Load an analysis file and redo the predictions on the validation set / test set,
this time with augmented data and averaging. Store them as numpy files.
"""

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
import matplotlib.pyplot as plt



y_train = np.load("data/solutions_train.npy")
print("available solution: %s " % (y_train.shape[0]))
ra.y_train=y_train

# split training data into training + a small validation set
ra.num_train = y_train.shape[0]

ra.num_valid = ra.num_train // 20 # integer division, is defining validation size
ra.num_train -= ra.num_valid

print ra.num_train

ra.y_valid = ra.y_train[ra.num_train:]
ra.y_train = ra.y_train[:ra.num_train]

y_train=y_train[:ra.num_train]

print("train solution: %s " % (y_train.shape[0]))

load_data.num_train=y_train.shape[0]
load_data.train_ids = np.load("data/train_ids.npy")

ra.load_data.num_train = load_data.num_train
ra.load_data.train_ids = load_data.train_ids

ra.valid_ids = load_data.train_ids[ra.num_train:]
ra.train_ids = load_data.train_ids[:ra.num_train]

BATCH_SIZE = 16 # 16
NUM_INPUT_FEATURES = 3

CHUNK_SIZE = 1008 # 10000 # this should be a multiple of the batch size

# ANALYSIS_PATH = "analysis/try_convnet_cc_multirot_3x69r45_untied_bias.pkl"
ANALYSIS_PATH = "analysis/final/try_convent_edgeTime1p5.pkl"

DO_VALID = True # disable this to not bother with the validation set evaluation
DO_VALID_CORR = False
DO_VALID_SCATTER = False

VALID_CORR_OUTPUT_FILTER = np.zeros((37))
VALID_CORR_OUTPUT_FILTER[2]=1  #star or artifact
VALID_CORR_OUTPUT_FILTER[3]=1  #edge on yes
VALID_CORR_OUTPUT_FILTER[4]=1  #edge on no
VALID_CORR_OUTPUT_FILTER[5]=1  #bar feature yes
VALID_CORR_OUTPUT_FILTER[7]=1  #spiral arms yes
VALID_CORR_OUTPUT_FILTER[14]=1  #anything odd? no
VALID_CORR_OUTPUT_FILTER[18]=1  #ring
VALID_CORR_OUTPUT_FILTER[19]=1  #lence
VALID_CORR_OUTPUT_FILTER[20]=1  #disturbed
VALID_CORR_OUTPUT_FILTER[21]=1  #irregular
VALID_CORR_OUTPUT_FILTER[22]=1  #other
VALID_CORR_OUTPUT_FILTER[23]=1  #merger
VALID_CORR_OUTPUT_FILTER[24]=1  #dust lane

N_Corr_Filter_Images=np.sum(VALID_CORR_OUTPUT_FILTER); 

DO_VALIDSTUFF_ON_TRAIN = True

DO_TEST = False # disable this to not generate predictions on the testset
DO_PRINT_FILTERS = False

IMAGE_OUTPUT_PATH = "images_wColorbar_newYear2_realValid"

output_names=["smooth","featureOrdisk","NoGalaxy","EdgeOnYes","EdgeOnNo","BarYes","BarNo","SpiralYes","SpiralNo","BulgeNo","BulgeJust","BulgeObvious","BulgDominant","OddYes","OddNo","RoundCompletly","RoundBetween","RoundCigar","Ring","Lense","Disturbed","Irregular","Other","Merger","DustLane","BulgeRound","BlulgeBoxy","BulgeNo2","SpiralTight","SpiralMedium","SpiralLoose","Spiral1Arm","Spiral2Arm","Spiral3Arm","Spiral4Arm","SpiralMoreArms","SpiralCantTell"]

target_filename = os.path.basename(ANALYSIS_PATH).replace(".pkl", ".npy.gz")
target_path_valid = os.path.join("predictions/final/augmented/valid", target_filename)
target_path_test = os.path.join("predictions/final/augmented/test", target_filename)


print "Loading model data etc."
analysis = np.load(ANALYSIS_PATH)

input_sizes = [(69, 69), (69, 69)]

ds_transforms = [
    ra.build_ds_transform(3.0, target_size=input_sizes[0]),
    ra.build_ds_transform(3.0, target_size=input_sizes[1]) + ra.build_augmentation_transform(rotation=45)]

num_input_representations = len(ds_transforms)

# split training data into training + a small validation set
num_train = load_data.num_train
num_valid = ra.num_valid
#num_train -= num_valid
num_test = load_data.num_test

print("num test %s " % (num_test))

valid_ids = load_data.train_ids[num_train:]
train_ids = load_data.train_ids[:num_train]
test_ids = load_data.test_ids

train_indices = np.arange(num_train)
valid_indices = np.arange(num_train, num_train+num_valid)
test_indices = np.arange(num_test)

y_valid = np.load("data/solutions_train.npy")[num_train:]

print("validation solution: %s" % y_valid.shape[0])

if(num_train!=load_data.num_train!=ra.num_train):
	print "training numbers problem"
	print num_train
	print load_data.num_train
	print ra.num_train

print "Build model"
l0 = layers.Input2DLayer(BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1])
l0_45 = layers.Input2DLayer(BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[1][0], input_sizes[1][1])

#l0r = layers.MultiRotSliceLayer([l0, l0_45], part_size=45, include_flip=True)

l0r = layers.MultiRotSliceLayer([l0, l0_45], part_size=45, include_flip=False)

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

j3 = layers.MultiRotMergeLayer(l3s, num_views=2) # 4) # merge convolutional parts


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
l6 = custom.OptimisedDivGalaxyOutputLayer(l5) # this incorporates the constraints on the output (probabilities sum to one, weighting, etc.)


print("output shapes: l0 %s , l0r %s , l0s %s , l1a %s , l2a %s, l3 %s , j3 %s , l4 %s , l5 %s  " % ( l0.get_output_shape(), l0r.get_output_shape(), l0s.get_output_shape(), l1a.get_output_shape(),l2a.get_output_shape(), l3.get_output_shape(), j3.get_output_shape(), l4.get_output_shape(), l5.get_output_shape()))


xs_shared = [theano.shared(np.zeros((1,1,1,1), dtype=theano.config.floatX)) for _ in xrange(num_input_representations)]

idx = T.lscalar('idx')

givens = {
    l0.input_var: xs_shared[0][idx*BATCH_SIZE:(idx+1)*BATCH_SIZE],
    l0_45.input_var: xs_shared[1][idx*BATCH_SIZE:(idx+1)*BATCH_SIZE],
}

compute_output = theano.function([idx], l6.predictions(dropout_active=False), givens=givens)


print "Load model parameters"
layers.set_param_values(l6, analysis['param_values'])

print "Create generators"
# set here which transforms to use to make predictions
augmentation_transforms = []
for zoom in [1 / 1.2, 1.0, 1.2]:
    for angle in np.linspace(0, 360, 10, endpoint=False):
        augmentation_transforms.append(ra.build_augmentation_transform(rotation=angle, zoom=zoom))
        augmentation_transforms.append(ra.build_augmentation_transform(rotation=(angle + 180), zoom=zoom, shear=180)) # flipped

print "  %d augmentation transforms." % len(augmentation_transforms)

print(CHUNK_SIZE)
print(input_sizes)

#augmented_data_gen_valid = ra.realtime_fixed_augmented_data_gen(valid_indices, 'train', augmentation_transforms=augmentation_transforms, chunk_size=CHUNK_SIZE, target_sizes=input_sizes, ds_transforms=ds_transforms)
augmented_data_gen_valid = ra.realtime_fixed_augmented_data_gen(valid_indices, 'train',  chunk_size=CHUNK_SIZE, target_sizes=input_sizes, ds_transforms=ds_transforms)
valid_gen = load_data.buffered_gen_mp(augmented_data_gen_valid, buffer_size=1)

augmented_data_gen_train = ra.realtime_fixed_augmented_data_gen(train_indices, 'train',  chunk_size=CHUNK_SIZE, target_sizes=input_sizes, ds_transforms=ds_transforms)
train_gen = load_data.buffered_gen_mp(augmented_data_gen_train, buffer_size=1)

#augmented_data_gen_test = ra.realtime_fixed_augmented_data_gen(test_indices, 'test', augmentation_transforms=augmentation_transforms, chunk_size=CHUNK_SIZE, target_sizes=input_sizes, ds_transforms=ds_transforms)
augmented_data_gen_test = ra.realtime_fixed_augmented_data_gen(test_indices, 'test',  chunk_size=CHUNK_SIZE, target_sizes=input_sizes, ds_transforms=ds_transforms)
test_gen = load_data.buffered_gen_mp(augmented_data_gen_test, buffer_size=1)


approx_num_chunks_train = int(np.ceil(num_train * len(augmentation_transforms) / float(CHUNK_SIZE)))
approx_num_chunks_valid = int(np.ceil(num_valid * len(augmentation_transforms) / float(CHUNK_SIZE)))
approx_num_chunks_test = int(np.ceil(num_test * len(augmentation_transforms) / float(CHUNK_SIZE)))

print "Approximately %d chunks for the validation set" % approx_num_chunks_valid
print "Approximately %d chunks for the training set" % approx_num_chunks_train
print "Approximately %d chunks for the test set" % approx_num_chunks_test


if DO_VALID:

    pixels_color0=[]
    pixels_color1=[]
    pixels_color2=[]

#correlation_output_images=[]

    given0 = {
    l0.input_var: xs_shared[0][idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    }


    l0_output_v = theano.function([idx], l0.output(), givens=given0)

    print
    print "VALIDATION SET"
    print "Compute predictions"
    if DO_VALID_CORR: print "Pixel Correllations will be calculated"
    predictions_list = []
    start_time = time.time()

    for e, (chunk_data, chunk_length) in enumerate(valid_gen):
        print "Chunk %d" % (e + 1)
        xs_chunk = chunk_data

        # need to transpose the chunks to move the 'channels' dimension up
        xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

        print "  load data onto GPU"
        for x_shared, x_chunk in zip(xs_shared, xs_chunk):
            x_shared.set_value(x_chunk)
        num_batches_chunk = int(np.ceil(chunk_length / float(BATCH_SIZE)))

        # make predictions, don't forget to cute off the zeros at the end
        predictions_chunk_list = []
        for b in xrange(num_batches_chunk):
            if b % 1000 == 0:
                print "  batch %d/%d" % (b + 1, num_batches_chunk)
		
            if DO_VALID_CORR: 
		input_img = l0_output_v(b).transpose(1,2,3,0)
		#if b==0: print input_img.shape
		pixels_color0.append(input_img[0])
		pixels_color1.append(input_img[1])
		pixels_color2.append(input_img[2])
		#pixels_color0=np.dstack(input_img[0])
		#if b==0: print pixels_color0.shape
		
            predictions = compute_output(b)
	    #if b==0: print predictions.shape
            predictions_chunk_list.append(predictions)

        predictions_chunk = np.vstack(predictions_chunk_list)
	#print predictions_chunk.shape
        predictions_chunk = predictions_chunk[:chunk_length] # cut off zeros / padding
	#print predictions_chunk.shape
        #print "  compute average over transforms"
        #predictions_chunk_avg = predictions_chunk.reshape(-1, len(augmentation_transforms), 37).mean(1)

        #predictions_list.append(predictions_chunk_avg)
	
	predictions_list.append(predictions_chunk)
	#print predictions_list.shape
        time_since_start = time.time() - start_time
        print "  %s since start" % load_data.hms(time_since_start)


    all_predictions = np.vstack(predictions_list)

    # postprocessing: clip all predictions to 0-1
    all_predictions[all_predictions > 1] = 1.0
    all_predictions[all_predictions < 0] = 0.0

    #print all_predictions.shape

    if DO_VALID_CORR:
	print "begin correlation calculation"
	pixels_color0_stack=np.dstack(pixels_color0)
	pixels_color1_stack=np.dstack(pixels_color1)
	pixels_color2_stack=np.dstack(pixels_color2)
 
	#print pixels_color0_stack.shape
	if not os.path.isdir(IMAGE_OUTPUT_PATH): os.mkdir(IMAGE_OUTPUT_PATH)  
    	#plt.gray()	
	os.chdir(IMAGE_OUTPUT_PATH)
	if not os.path.isdir("InOutCorr"): os.mkdir("InOutCorr")
	os.chdir("InOutCorr")
    	for i in xrange(0,VALID_CORR_OUTPUT_FILTER.shape[0]):
		if not VALID_CORR_OUTPUT_FILTER[i]: continue
		print "begin correlation of output %s" % i
		corr_image_line_c0=np.zeros(input_sizes[0][0]*input_sizes[0][1])
		corr_image_line_c1=np.zeros(input_sizes[0][0]*input_sizes[0][1])
		corr_image_line_c2=np.zeros(input_sizes[0][0]*input_sizes[0][1])
		pixels_colors0_line=np.reshape(pixels_color0_stack,(input_sizes[0][0]*input_sizes[0][1],pixels_color0_stack.shape[2]))
		pixels_colors1_line=np.reshape(pixels_color1_stack,(input_sizes[0][0]*input_sizes[0][1],pixels_color1_stack.shape[2]))
		pixels_colors2_line=np.reshape(pixels_color2_stack,(input_sizes[0][0]*input_sizes[0][1],pixels_color2_stack.shape[2]))

		for j in xrange(0,input_sizes[0][0]*input_sizes[0][1]) :
			if j == 0: 
				print pixels_colors0_line[j].shape
				print all_predictions.T[i].shape
			corr_image_line_c0[j]=np.corrcoef( pixels_colors0_line[j][:all_predictions.shape[0]] , all_predictions.T[i] )[1][0]
			corr_image_line_c1[j]=np.corrcoef( pixels_colors1_line[j][:all_predictions.shape[0]] , all_predictions.T[i] )[1][0]
			corr_image_line_c2[j]=np.corrcoef( pixels_colors2_line[j][:all_predictions.shape[0]] , all_predictions.T[i] )[1][0]

		#correlation_output_images.append(np.reshape(corr_image_line,(input_sizes[0][0],input_sizes[0][1])))

      		plt.imshow( np.reshape( corr_image_line_c0, (input_sizes[0][0],input_sizes[0][1]) ) , interpolation='none',vmin=-0.4,vmax=0.4 ) #Needs to be in row,col order
		plt.colorbar()
      		plt.savefig("inputCorrelationToOutput%s%s_c0_Red.jpg" % (i,output_names[i]))
		plt.close()

      		plt.imshow( np.reshape( corr_image_line_c1, (input_sizes[0][0],input_sizes[0][1]) ) , interpolation='none' ,vmin=-0.4,vmax=0.4) #Needs to be in row,col order
		plt.colorbar()
      		plt.savefig("inputCorrelationToOutput%s%s_c1_Green.jpg" % (i,output_names[i]))
		plt.close()

      		plt.imshow( np.reshape( corr_image_line_c2, (input_sizes[0][0],input_sizes[0][1]) ) , interpolation='none',vmin=-0.4,vmax=0.4 ) #Needs to be in row,col order
		plt.colorbar()
      		plt.savefig("inputCorrelationToOutput%s%s_c2_Blue.jpg" % (i,output_names[i]))
		plt.close()

	os.chdir("../..")


    print "Write predictions to %s" % target_path_valid
    load_data.save_gz(target_path_valid, all_predictions)

    print "Evaluate"
    rmse_valid = analysis['losses_valid'][-1]
    rmse_augmented = np.sqrt(np.mean((y_valid - all_predictions)**2))
    print "  MSE (last iteration):\t%.6f" % rmse_valid
    print "  MSE (augmented):\t%.6f  RMSE/mean: %s " % ( rmse_augmented , rmse_augmented/np.mean(y_valid) )
    print "  MSE output wise (augmented):"
    from termcolor import colored
    for i in xrange(0,VALID_CORR_OUTPUT_FILTER.shape[0]):
	oneMSE = np.sqrt(np.mean((y_valid.T[i] - all_predictions.T[i])**2))
  	if oneMSE/np.mean(y_valid.T[i])>1.2*rmse_augmented/np.mean(y_valid):
		print colored("    output %s (%s): \t%.6f  RMSE/mean: %s " %(output_names[i],i,oneMSE,oneMSE/np.mean(y_valid.T[i])),'red')
	elif oneMSE/np.mean(y_valid.T[i])<0.8*rmse_augmented/np.mean(y_valid):
		print colored("    output %s (%s): \t%.6f  RMSE/mean: %s " %(output_names[i],i,oneMSE,oneMSE/np.mean(y_valid.T[i])),'green')
	else:
		print ("    output %s (%s): \t%.6f  RMSE/mean: %s " %(output_names[i],i,oneMSE,oneMSE/np.mean(y_valid.T[i])))
    
    if DO_VALID_SCATTER:
	if not os.path.isdir(IMAGE_OUTPUT_PATH): os.mkdir(IMAGE_OUTPUT_PATH)  
    	#plt.gray()	
	os.chdir(IMAGE_OUTPUT_PATH)
	if not os.path.isdir("ValidScatter"): os.mkdir("ValidScatter")
	os.chdir("ValidScatter")
   
   	for i in xrange(0,VALID_CORR_OUTPUT_FILTER.shape[0]):
		y = all_predictions.T[i]
		x = y_valid.T[i]
		fig, ax = plt.subplots()
		fit = np.polyfit(x, y, deg=1)
		ax.plot(x, fit[0] * x + fit[1], color='red')
		ax.scatter(x, y)
  		plt.ylabel('prediction')
  		plt.xlabel('target')
		plt.title("valid %s"%(output_names[i]))
		oneMSE = np.sqrt(np.mean((y_valid.T[i] - all_predictions.T[i])**2))
		plt.text(60, .025, 'RMSE: %s , RMSE/mean: %s '%(oneMSE,oneMSE/np.mean(y_valid.T[i])))
		plt.savefig("validScatter_%s_%s.jpg" % (i,output_names[i]))
		plt.close()

	os.chdir("../..")





if DO_VALIDSTUFF_ON_TRAIN: 

    pixels_color0=[]
    pixels_color1=[]
    pixels_color2=[]

#correlation_output_images=[]

    given0 = {
    l0.input_var: xs_shared[0][idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    }


    l0_output_v = theano.function([idx], l0.output(), givens=given0)

    print
    print "TRAIN SET"
    print "Compute predictions"
    if DO_VALID_CORR: print "Pixel Correllations will be calculated"
    predictions_list = []
    start_time = time.time()

    for e, (chunk_data, chunk_length) in enumerate(train_gen):
        print "Chunk %d" % (e + 1)
        xs_chunk = chunk_data

        # need to transpose the chunks to move the 'channels' dimension up
        xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

        print "  load data onto GPU"
        for x_shared, x_chunk in zip(xs_shared, xs_chunk):
            x_shared.set_value(x_chunk)
        num_batches_chunk = int(np.ceil(chunk_length / float(BATCH_SIZE)))

        # make predictions, don't forget to cute off the zeros at the end
        predictions_chunk_list = []
        for b in xrange(num_batches_chunk):
            if b % 1000 == 0:
                print "  batch %d/%d" % (b + 1, num_batches_chunk)
		
            if DO_VALID_CORR: 
		input_img = l0_output_v(b).transpose(1,2,3,0)
		#if b==0: print input_img.shape
		pixels_color0.append(input_img[0])
		pixels_color1.append(input_img[1])
		pixels_color2.append(input_img[2])
		#pixels_color0=np.dstack(input_img[0])
		#if b==0: print pixels_color0.shape
		
            predictions = compute_output(b)
	    #if b==0: print predictions.shape
            predictions_chunk_list.append(predictions)

        predictions_chunk = np.vstack(predictions_chunk_list)
	#print predictions_chunk.shape
        predictions_chunk = predictions_chunk[:chunk_length] # cut off zeros / padding
	#print predictions_chunk.shape
        #print "  compute average over transforms"
        #predictions_chunk_avg = predictions_chunk.reshape(-1, len(augmentation_transforms), 37).mean(1)

        #predictions_list.append(predictions_chunk_avg)
	
	predictions_list.append(predictions_chunk)
	#print predictions_list.shape
        time_since_start = time.time() - start_time
        print "  %s since start" % load_data.hms(time_since_start)


    all_predictions = np.vstack(predictions_list)

    # postprocessing: clip all predictions to 0-1
    all_predictions[all_predictions > 1] = 1.0
    all_predictions[all_predictions < 0] = 0.0

    #print all_predictions.shape

    if DO_VALID_CORR:
	print "begin correlation calculation"
	pixels_color0_stack=np.dstack(pixels_color0)
	pixels_color1_stack=np.dstack(pixels_color1)
	pixels_color2_stack=np.dstack(pixels_color2)
 
	#print pixels_color0_stack.shape
	if not os.path.isdir(IMAGE_OUTPUT_PATH): os.mkdir(IMAGE_OUTPUT_PATH)  
    	#plt.gray()	
	os.chdir(IMAGE_OUTPUT_PATH)
	if not os.path.isdir("InOutCorr_Train"): os.mkdir("InOutCorr_Train")
	os.chdir("InOutCorr_Train")
    	for i in xrange(0,VALID_CORR_OUTPUT_FILTER.shape[0]):
		if not VALID_CORR_OUTPUT_FILTER[i]: continue
		print "begin correlation of output %s" % i
		corr_image_line_c0=np.zeros(input_sizes[0][0]*input_sizes[0][1])
		corr_image_line_c1=np.zeros(input_sizes[0][0]*input_sizes[0][1])
		corr_image_line_c2=np.zeros(input_sizes[0][0]*input_sizes[0][1])
		pixels_colors0_line=np.reshape(pixels_color0_stack,(input_sizes[0][0]*input_sizes[0][1],pixels_color0_stack.shape[2]))
		pixels_colors1_line=np.reshape(pixels_color1_stack,(input_sizes[0][0]*input_sizes[0][1],pixels_color1_stack.shape[2]))
		pixels_colors2_line=np.reshape(pixels_color2_stack,(input_sizes[0][0]*input_sizes[0][1],pixels_color2_stack.shape[2]))

		for j in xrange(0,input_sizes[0][0]*input_sizes[0][1]) :
			if j == 0: 
				print pixels_colors0_line[j].shape
				print all_predictions.T[i].shape
			corr_image_line_c0[j]=np.corrcoef( pixels_colors0_line[j][:all_predictions.shape[0]] , all_predictions.T[i] )[1][0]
			corr_image_line_c1[j]=np.corrcoef( pixels_colors1_line[j][:all_predictions.shape[0]] , all_predictions.T[i] )[1][0]
			corr_image_line_c2[j]=np.corrcoef( pixels_colors2_line[j][:all_predictions.shape[0]] , all_predictions.T[i] )[1][0]
			print 'debug: pixel %s didnt crash' % j

		#correlation_output_images.append(np.reshape(corr_image_line,(input_sizes[0][0],input_sizes[0][1])))
		
		print "debug: after treain correcoeff-loop for filter %s " % i

      		plt.imshow( np.reshape( corr_image_line_c0, (input_sizes[0][0],input_sizes[0][1]) ) , interpolation='none',vmin=-0.4,vmax=0.4 ) #Needs to be in row,col order
		plt.colorbar()
      		plt.savefig("inputCorrelationToOutput%s%s_c0_Red.jpg" % (i,output_names[i]))
		plt.close()

      		plt.imshow( np.reshape( corr_image_line_c1, (input_sizes[0][0],input_sizes[0][1]) ) , interpolation='none' ,vmin=-0.4,vmax=0.4) #Needs to be in row,col order
		plt.colorbar()
      		plt.savefig("inputCorrelationToOutput%s%s_c1_Green.jpg" % (i,output_names[i]))
		plt.close()

      		plt.imshow( np.reshape( corr_image_line_c2, (input_sizes[0][0],input_sizes[0][1]) ) , interpolation='none',vmin=-0.4,vmax=0.4 ) #Needs to be in row,col order
		plt.colorbar()
      		plt.savefig("inputCorrelationToOutput%s%s_c2_Blue.jpg" % (i,output_names[i]))
		plt.close()

	os.chdir("../..")

    #print "Write predictions to %s" % target_path_valid
    #load_data.save_gz(target_path_valid, all_predictions)

    print "Evaluate"
    y_train=y_train[:all_predictions.shape[0]]
    rmse_train = analysis['losses_train'][-1]
    rmse_augmented = np.sqrt(np.mean((y_train - all_predictions)**2))
    print "  MSE (last iteration):\t%.6f" % rmse_train
    print "  MSE (augmented):\t%.6f  RMSE/mean: %s " % ( rmse_augmented , rmse_augmented/np.mean(y_train) )
    print "  MSE output wise (augmented):"
    from termcolor import colored
    for i in xrange(0,VALID_CORR_OUTPUT_FILTER.shape[0]):
	oneMSE = np.sqrt(np.mean((y_train.T[i] - all_predictions.T[i])**2))
  	if oneMSE>1.2*rmse_augmented:
		print colored("    output %s (%s): \t%.6f  RMSE/mean: %s " %(output_names[i],i,oneMSE,oneMSE/np.mean(y_train.T[i])),'red')
	elif oneMSE<0.8*rmse_augmented:
		print colored("    output %s (%s): \t%.6f  RMSE/mean: %s " %(output_names[i],i,oneMSE,oneMSE/np.mean(y_train.T[i])),'green')
	else:
		print ("    output %s (%s): \t%.6f  RMSE/mean: %s " %(output_names[i],i,oneMSE,oneMSE/np.mean(y_train.T[i])))
    
    if DO_VALID_SCATTER:
	if not os.path.isdir(IMAGE_OUTPUT_PATH): os.mkdir(IMAGE_OUTPUT_PATH)  
    	#plt.gray()	
	os.chdir(IMAGE_OUTPUT_PATH)
	if not os.path.isdir("TrainScatter"): os.mkdir("TrainScatter")
	os.chdir("TrainScatter")
   
   	for i in xrange(0,VALID_CORR_OUTPUT_FILTER.shape[0]):
		y = all_predictions.T[i]
		x = y_train.T[i]
		fig, ax = plt.subplots()
		fit = np.polyfit(x, y, deg=1)
		ax.plot(x, fit[0] * x + fit[1], color='red')
		ax.scatter(x, y)
  		plt.ylabel('prediction')
  		plt.xlabel('target')
		plt.title("train %s"%(output_names[i]))
		oneMSE = np.sqrt(np.mean((y_train.T[i] - all_predictions.T[i])**2))
		plt.text(60, .025, 'RMSE: %s , RMSE/mean: %s '%(oneMSE,oneMSE/np.mean(y_train.T[i])))
		plt.savefig("TrainScatter%s_%s.jpg" % (i,output_names[i]))
		plt.close()

	os.chdir("../..")



if DO_TEST:
    print
    print "TEST SET"
    print "Compute predictions"
    predictions_list = []
    start_time = time.time()

    for e, (chunk_data, chunk_length) in enumerate(test_gen):
        print "Chunk %d" % (e + 1)
        xs_chunk = chunk_data

        # need to transpose the chunks to move the 'channels' dimension up
        xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

        print "  load data onto GPU"
        for x_shared, x_chunk in zip(xs_shared, xs_chunk):
            x_shared.set_value(x_chunk)
        num_batches_chunk = int(np.ceil(chunk_length / float(BATCH_SIZE)))

        # make predictions, don't forget to cute off the zeros at the end
        predictions_chunk_list = []
        for b in xrange(num_batches_chunk):
            if b % 1000 == 0:
                print "  batch %d/%d" % (b + 1, num_batches_chunk)

            predictions = compute_output(b)
            predictions_chunk_list.append(predictions)

        predictions_chunk = np.vstack(predictions_chunk_list)
        predictions_chunk = predictions_chunk[:chunk_length] # cut off zeros / padding

        print "  compute average over transforms"
        predictions_chunk_avg = predictions_chunk.reshape(-1, len(augmentation_transforms), 37).mean(1)

        predictions_list.append(predictions_chunk_avg)

        time_since_start = time.time() - start_time
        print "  %s since start" % load_data.hms(time_since_start)

    all_predictions = np.vstack(predictions_list)


    print "Write predictions to %s" % target_path_test
    load_data.save_gz(target_path_test, all_predictions)

    print "Done!"


if DO_PRINT_FILTERS:
    if not os.path.isdir(IMAGE_OUTPUT_PATH): os.mkdir(IMAGE_OUTPUT_PATH)  
    #plt.gray()

    #os.chdir("..")
    print "print filtered"

    print "1st image"

    print(test_gen)
    print(valid_gen) 
    print(BATCH_SIZE)
    chunk_data, chunk_length = test_gen.next()
    print(chunk_length)
    xs_chunk = chunk_data
        # need to transpose the chunks to move the 'channels' dimension up
    xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

    print "  load data onto GPU"
    for x_shared, x_chunk in zip(xs_shared, xs_chunk):
      	   x_shared.set_value(x_chunk)
    num_batches_chunk = int(np.ceil(chunk_length / float(BATCH_SIZE)))

    predictions = compute_output(0)


    l0_output = theano.function([idx], l0r.output(), givens=givens)
    l1_output = theano.function([idx], l1a.output(), givens=givens)
    l3_output = theano.function([idx], l2a.output(), givens=givens)

    os.chdir(IMAGE_OUTPUT_PATH)
#filter of layer 1 , output format should be  (32, 45, 45, 128)
    input_img = l0_output(0)[0]
    if not os.path.isdir("l0real"): os.mkdir("l0real")  
    for i in range(0,3):
      plt.imshow(input_img[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l0real/%s.jpg" % i)
      plt.close()

    input_img = l0_output(0)[1]
    if not os.path.isdir("l0real2"): os.mkdir("l0real2")  
    for i in range(0,3):
      plt.imshow(input_img[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l0real2/%s.jpg" % i)
      plt.close()

    filters1 = l1_output(0).transpose(3,0,1,2)[0]
    if not os.path.isdir("l1real"): os.mkdir("l1real")  
    for i in range(0,32):
      plt.imshow(filters1[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l1real/%s.jpg" % i)
      plt.close()

    filters2 = l1_output(0).transpose(3,0,1,2)[1]
    if not os.path.isdir("l1real2"): os.mkdir("l1real2")  
    for i in range(0,32):
      plt.imshow(filters2[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l1real2/%s.jpg" % i)
      plt.close()


    filters3 = l3_output(0).transpose(3,0,1,2)[0]
    if not os.path.isdir("l2real"): os.mkdir("l2real")  
    for i in range(0,64):
      plt.imshow(filters3[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l2real/%s.jpg" % i)
      plt.close()



    print "2nd image"

   # for e, (chunk_data, chunk_length) in enumerate(test_gen):
   #     if e>0: break
#	xs_chunk = chunk_data
        # need to transpose the chunks to move the 'channels' dimension up
 #   	xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

  #  	print "  load data onto GPU"
   # 	for x_shared, x_chunk in zip(xs_shared, xs_chunk):
    #  	      x_shared.set_value(x_chunk)
    #	num_batches_chunk = int(np.ceil(chunk_length / float(BATCH_SIZE)))

    #	predictions = compute_output(8)

    #l0_output = theano.function([idx], l0r.output(), givens=givens)
    #l1_output = theano.function([idx], l1a.output(), givens=givens)
    #l3_output = theano.function([idx], l2a.output(), givens=givens)

    input_img = l0_output(0)[128/16]
    if not os.path.isdir("l0real_2"): os.mkdir("l0real_2")  
    for i in range(0,3):
      plt.imshow(input_img[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l0real_2/%s.jpg" % i)
      plt.close()
    input_img = l0_output(0)[128/16+1]
    if not os.path.isdir("l0real2_2"): os.mkdir("l0real2_2")  
    for i in range(0,3):
      plt.imshow(input_img[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l0real2_2/%s.jpg" % i)
      plt.close()


#filter of layer 1 , output format should be  (32, 45, 45, 128)
    filters1 = l1_output(0).transpose(3,0,1,2)[128/16]
    if not os.path.isdir("l1real_2"): os.mkdir("l1real_2")  
    for i in range(0,32):
      plt.imshow(filters1[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l1real_2/%s.jpg" % i)
      plt.close()

    filters2 = l1_output(0).transpose(3,0,1,2)[128/16+1]
    if not os.path.isdir("l1real2_2"): os.mkdir("l1real2_2")  
    for i in range(0,32):
      plt.imshow(filters2[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l1real2_2/%s.jpg" % i)
      plt.close()


    filters3 = l3_output(0).transpose(3,0,1,2)[128/16]
    if not os.path.isdir("l2real_2"): os.mkdir("l2real_2")  
    for i in range(0,64):
      plt.imshow(filters3[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l2real_2/%s.jpg" % i)
      plt.close()


    print "3rd image"

   # for e, (chunk_data, chunk_length) in enumerate(test_gen):
   #     if e>0: break
#	xs_chunk = chunk_data
        # need to transpose the chunks to move the 'channels' dimension up
 #   	xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

  #  	print "  load data onto GPU"
   # 	for x_shared, x_chunk in zip(xs_shared, xs_chunk):
    #  	      x_shared.set_value(x_chunk)
    #	num_batches_chunk = int(np.ceil(chunk_length / float(BATCH_SIZE)))

    #	predictions = compute_output(8)

    #l0_output = theano.function([idx], l0r.output(), givens=givens)
    #l1_output = theano.function([idx], l1a.output(), givens=givens)
    #l3_output = theano.function([idx], l2a.output(), givens=givens)

    input_img = l0_output(0)[2*128/16]
    if not os.path.isdir("l0real_3"): os.mkdir("l0real_3")  
    for i in range(0,3):
      plt.imshow(input_img[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.close()
      plt.savefig("l0real_3/%s.jpg" % i)
    input_img = l0_output(0)[2*128/16+1]
    if not os.path.isdir("l0real2_3"): os.mkdir("l0real2_3")  
    for i in range(0,3):
      plt.imshow(input_img[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l0real2_3/%s.jpg" % i)
      plt.close()


#filter of layer 1 , output format should be  (32, 45, 45, 128)
    filters1 = l1_output(0).transpose(3,0,1,2)[2*128/16]
    if not os.path.isdir("l1real_3"): os.mkdir("l1real_3")  
    for i in range(0,32):
      plt.imshow(filters1[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l1real_3/%s.jpg" % i)
      plt.close()

    filters2 = l1_output(0).transpose(3,0,1,2)[2*128/16+1]
    if not os.path.isdir("l1real2_3"): os.mkdir("l1real2_3")  
    for i in range(0,32):
      plt.imshow(filters2[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l1real2_3/%s.jpg" % i)
      plt.close()


    filters3 = l3_output(0).transpose(3,0,1,2)[2*128/16]
    if not os.path.isdir("l2real_3"): os.mkdir("l2real_3")  
    for i in range(0,64):
      plt.imshow(filters3[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l2real_3/%s.jpg" % i)
      plt.close()

    print "print filter"
    print "black input"
   # os.chdir(IMAGE_OUTPUT_PATH)
    start_time = time.time()


    inputBlack=np.zeros((BATCH_SIZE, NUM_INPUT_FEATURES, 69, 69),dtype=theano.config.floatX)
    inputWhite=np.ones((BATCH_SIZE, NUM_INPUT_FEATURES, 69, 69),dtype=theano.config.floatX)

    #black and white are switched!!!!

    for x_shared in xs_shared:
          x_shared.set_value(inputWhite)

  #  whitePrediction=compute_output(0)

  #  with open("blackPrediction.txt", 'w')as f:
#	f.write(" %s " % (whitePrediction))

    l1_output = theano.function([idx], l1a.output(), givens=givens)
    l3_output = theano.function([idx], l2a.output(), givens=givens)

#filter of layer 1 , output format should be  (32, 45, 45, 128)
    filters1 = l1_output(0).transpose(3,0,1,2)[0]
    if not os.path.isdir("l1black"): os.mkdir("l1black")  
    for i in range(0,32):
      plt.imshow(filters1[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l1black/%s.jpg" % i)
      plt.close()

    filters2 = l1_output(0).transpose(3,0,1,2)[1]
    if not os.path.isdir("l1black2"): os.mkdir("l1black2")  
    for i in range(0,32):
      plt.imshow(filters2[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l1black2/%s.jpg" % i)
      plt.close()


    filters3 = l3_output(0).transpose(3,0,1,2)[0]
    if not os.path.isdir("l2black"): os.mkdir("l2black")  
    for i in range(0,64):
      plt.imshow(filters3[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l2black/%s.jpg" % i)
      plt.close()

    print "white input"

    for x_shared in xs_shared:
          x_shared.set_value(inputBlack)

#    blackPrediction=compute_output(0)

 #   with open("whitePrediction.txt", 'w')as f:
#	f.write(" %s " % (blackPrediction))

    l1_output = theano.function([idx], l1a.output(), givens=givens)
    l3_output = theano.function([idx], l2a.output(), givens=givens)

    filters1 = l1_output(0).transpose(3,0,1,2)[0]
    if not os.path.isdir("l1white"): os.mkdir("l1white")  
    for i in range(0,32):
     # if i==0:
     #   print("writing one filter image as txt") 
     #   with open("imageTest.txt", 'w')as f: 
      #    np.savetxt(f,filters1[i])
      plt.imshow(filters1[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l1white/%s.jpg" % i)
      plt.close()


    filters2 = l1_output(0).transpose(3,0,1,2)[1]
    if not os.path.isdir("l1white2"): os.mkdir("l1white2")  
    for i in range(0,32):
      plt.imshow(filters2[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l1white2/%s.jpg" % i)
      plt.close()

    filters3 = l3_output(0).transpose(3,0,1,2)[0]
    if not os.path.isdir("l2white"): os.mkdir("l2white")  
    for i in range(0,64):
      plt.imshow(filters3[i],interpolation='none' , vmin=0.0,vmax=1.0) #Needs to be in row,col order
      plt.colorbar()
      plt.savefig("l2white/%s.jpg" % i)
      plt.close()

print "end"

exit()


    
