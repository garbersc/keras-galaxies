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
from keras.layers import Dense, Activation, MaxPooling1D, Dropout, Input, Convolution2D, MaxoutDense
from keras.layers.core import Lambda, Flatten, Reshape, Permute
from keras.optimizers import SGD, Adam
from keras.engine.topology import Merge
from keras.callbacks import LearningRateScheduler
from keras import initializations
import functools

from keras_extra_layers import kerasCudaConvnetPooling2DLayer, kerasCudaConvnetConv2DLayer, fPermute
from custom_for_keras import kaggle_MultiRotMergeLayer_output, OptimisedDivGalaxyOutput, kaggle_input, kaggle_sliced_accuracy, dense_weight_init_values, rmse, input_generator

class kaggle_winsol(Object):
    def __init__(self,BATCH_SIZES,NUM_INPUT_FEATURES,PART_SIZE,input_sizes,LEARNING_RATE_SCHEDULE,MOMENTUM,**kwargs):
        self.BATCH_SIZES=BATCH_SIZES
        self.NUM_INPUT_FEATURES=NUM_INPUT_FEATURES
        self.input_sizes=input_sizes
        self.PART_SIZE=PART_SIZE
        self.LEARNING_RATE_SCHEDULE=LEARNING_RATE_SCHEDULE
        self.MOMENTUM=MOMENTUM

    def init_model(self):
        print "init model"
        input_tensor = Input(batch_shape=(self.BATCH_SIZE, self.NUM_INPUT_FEATURES, self.input_sizes[0][0], self.input_sizes[0][1]) , dtype='float32', name='input_tensor')
        input_tensor_45 = Input(batch_shape=(self.BATCH_SIZE, self.NUM_INPUT_FEATURES, self.input_sizes[0][0], self.input_sizes[0][1]) , dtype='float32', name='input_tensor_45') 

        input_0 = Lambda(lambda x: x,output_shape=(self.NUM_INPUT_FEATURES, self.input_sizes[0][0], self.input_sizes[0][1]),batch_input_shape=(self.BATCH_SIZE, self.NUM_INPUT_FEATURES, self.input_sizes[0][0], self.input_sizes[0][1]),name='lambda_input_0')
        input_45 = Lambda(lambda x: x,output_shape=(self.NUM_INPUT_FEATURES, self.input_sizes[1][0], self.input_sizes[1][1]),batch_input_shape=(self.BATCH_SIZE, self.NUM_INPUT_FEATURES, self.input_sizes[0][0], self.input_sizes[0][1]),name='lambda_input_45')

        model1 = Sequential()
        model1.add(input_0)

        model2 = Sequential()
        model2.add(input_45)

        model = Sequential()

        N_INPUT_VARIATION = 2 #depends on the kaggle input settings
        
        model.add(Merge([model1, model2], mode=kaggle_input , output_shape=lambda x: ((model1.output_shape[0]+model2.output_shape[0])*2*N_INPUT_VARIATION, self.NUM_INPUT_FEATURES, self.PART_SIZE, self.PART_SIZE) , arguments={'part_size':self.PART_SIZE, 'n_input_var': N_INPUT_VARIATION, 'include_flip':False, 'random_flip':True}  ))

        #needed for the pylearn moduls used by kerasCudaConvnetConv2DLayer and kerasCudaConvnetPooling2DLayer
        model.add(fPermute((1,2,3,0)))

        model.add(kerasCudaConvnetConv2DLayer(n_filters=32, filter_size=6 , untie_biases=True))        model.add(kerasCudaConvnetPooling2DLayer())

        model.add(kerasCudaConvnetConv2DLayer(n_filters=64, filter_size=5 , untie_biases=True))        model.add(kerasCudaConvnetPooling2DLayer())

        model.add(kerasCudaConvnetConv2DLayer(n_filters=128, filter_size=3 , untie_biases=True)) 
        model.add(kerasCudaConvnetConv2DLayer(n_filters=128, filter_size=3,  weights_std=0.1 , untie_biases=True )) 

        model.add(kerasCudaConvnetPooling2DLayer())

        model.add(fPermute((3,0,1,2)))

        model.add(Lambda(function=kaggle_MultiRotMergeLayer_output, output_shape=lambda x : ( x[0]//4//N_INPUT_VARIATION, (x[1]*x[2]*x[3]*4* N_INPUT_VARIATION) ) , arguments={'num_views':N_INPUT_VARIATION}) ) 

        model.add(Dropout(0.5))
        model.add(MaxoutDense(output_dim=2048, nb_feature=2 ,weights = dense_weight_init_values(model.output_shape[-1],2048, nb_feature=2) )) 

        model.add(Dropout(0.5))
        model.add(MaxoutDense(output_dim=2048, nb_feature=2 ,weights = dense_weight_init_values(model.output_shape[-1],2048, nb_feature=2) )) 

        model.add(Dropout(0.5))
        model.add(Dense(output_dim=37, weights = dense_weight_init_values(model.output_shape[-1],37 ,w_std = 0.01 , b_init_val = 0.1 ) ))

        model_seq=model([input_tensor,input_tensor_45])

        CATEGORISED = False #FXME has to be implemented
        
        output_layer_norm = Lambda(function=OptimisedDivGalaxyOutput , output_shape=lambda x: x ,arguments={'normalised':True,'categorised':CATEGORISED})(model_seq)
        output_layer_noNorm = Lambda(function=OptimisedDivGalaxyOutput , output_shape=lambda x: x ,arguments={'normalised':False,'categorised':CATEGORISED})(model_seq)

        model_norm=Model(input=[input_tensor,input_tensor_45],output=output_layer_norm)
        model_norm_metrics = Model(input=[input_tensor,input_tensor_45],output=output_layer_norm)
        model_noNorm=Model(input=[input_tensor,input_tensor_45],output=output_layer_noNorm)

        self.models = {'model_norm' : model_norm, 'model_norm_metrics':model_norm_metrics,'model_noNorm' }

        
    def compile_model(self):
        self.models['model_norm'].compile(loss='mean_squared_error', optimizer=SGD(lr=self.LEARNING_RATE_SCHEDULE[0], momentum=self.MOMENTUM, nesterov=True) )
        self.models['model_noNorm'].compile(loss='mean_squared_error', optimizer=SGD(lr=self.LEARNING_RATE_SCHEDULE[0], momentum=self.MOMENTUM, nesterov=True))

        self.models['model_norm_metrics'].compile(loss='mean_squared_error', optimizer=SGD(lr=self.LEARNING_RATE_SCHEDULE[0], momentum=self.MOMENTUM, nesterov=True), metrics=[rmse, 'categorical_accuracy',kaggle_sliced_accuracy])
