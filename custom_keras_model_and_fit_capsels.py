import numpy as np
import json
import time
from datetime import datetime, timedelta
import functools
from copy import copy

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input,  MaxoutDense
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam
from keras.engine.topology import Merge
from keras.callbacks import LearningRateScheduler

from keras_extra_layers import kerasCudaConvnetPooling2DLayer, fPermute, kerasCudaConvnetConv2DLayer
from custom_for_keras import kaggle_MultiRotMergeLayer_output, OptimisedDivGalaxyOutput, kaggle_input, kaggle_sliced_accuracy, dense_weight_init_values, rmse, lr_function


class kaggle_winsol:
    def __init__(self, BATCH_SIZE, NUM_INPUT_FEATURES, PART_SIZE, input_sizes,
                 LEARNING_RATE_SCHEDULE, MOMENTUM, LOSS_PATH, WEIGHTS_PATH):
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_INPUT_FEATURES = NUM_INPUT_FEATURES
        self.input_sizes = input_sizes
        self.PART_SIZE = PART_SIZE
        self.LEARNING_RATE_SCHEDULE = LEARNING_RATE_SCHEDULE
        self.current_lr = self.LEARNING_RATE_SCHEDULE[0]
        self.MOMENTUM = MOMENTUM
        self.WEIGHTS_PATH = WEIGHTS_PATH
        self.LOSS_PATH = LOSS_PATH
        self.hists = {}
        self.first_loss_save = False
        self.models = {}

    def init_hist_dics(self, model_in):
        for n in model_in:
            self.hists[n] = {}
            try:
                for o in model_in[n].metrics_names:
                    self.hists[n][o] = []
            except AttributeError:
                pass

        return self.hists

    def init_models(self):
        print "init model"
        input_tensor = Input(batch_shape=(self.BATCH_SIZE,
                                          self.NUM_INPUT_FEATURES,
                                          self.input_sizes[0][0],
                                          self.input_sizes[0][1]),
                             dtype='float32', name='input_tensor')
        input_tensor_45 = Input(batch_shape=(self.BATCH_SIZE,
                                             self.NUM_INPUT_FEATURES,
                                             self.input_sizes[0][0],
                                             self.input_sizes[0][1]),
                                dtype='float32', name='input_tensor_45')

        input_0 = Lambda(lambda x: x, output_shape=(self.NUM_INPUT_FEATURES,
                                                    self.input_sizes[0][0],
                                                    self.input_sizes[0][1]),
                         batch_input_shape=(
            self.BATCH_SIZE, self.NUM_INPUT_FEATURES, self.input_sizes[0][0],
                             self.input_sizes[0][1]), name='lambda_input_0')
        input_45 = Lambda(lambda x: x, output_shape=(self.NUM_INPUT_FEATURES,
                                                     self.input_sizes[1][0],
                                                     self.input_sizes[1][1]),
                          batch_input_shape=(
            self.BATCH_SIZE, self.NUM_INPUT_FEATURES, self.input_sizes[0][0],
                              self.input_sizes[0][1]), name='lambda_input_45')

        model1 = Sequential()
        model1.add(input_0)

        model2 = Sequential()
        model2.add(input_45)

        model = Sequential()

        N_INPUT_VARIATION = 2  # depends on the kaggle input settings

        model.add(Merge([model1, model2], mode=kaggle_input,
                        output_shape=lambda x: ((model1.output_shape[0]
                                                 + model2.output_shape[0]) * 2
                                                * N_INPUT_VARIATION,
                                                self.NUM_INPUT_FEATURES,
                                                self.PART_SIZE,
                                                self.PART_SIZE),
                        arguments={'part_size': self.PART_SIZE,
                                   'n_input_var': N_INPUT_VARIATION,
                                   'include_flip': False,
                                   'random_flip': True}))

        # needed for the pylearn moduls used by kerasCudaConvnetConv2DLayer and
        # kerasCudaConvnetPooling2DLayer
        model.add(fPermute((1, 2, 3, 0)))

        model.add(kerasCudaConvnetConv2DLayer(
            n_filters=32, filter_size=6, untie_biases=True))
        model.add(kerasCudaConvnetPooling2DLayer())

        model.add(kerasCudaConvnetConv2DLayer(
            n_filters=64, filter_size=5, untie_biases=True))
        model.add(kerasCudaConvnetPooling2DLayer())

        model.add(kerasCudaConvnetConv2DLayer(
            n_filters=128, filter_size=3, untie_biases=True))
        model.add(kerasCudaConvnetConv2DLayer(n_filters=128,
                                              filter_size=3,  weights_std=0.1,
                                              untie_biases=True))

        model.add(kerasCudaConvnetPooling2DLayer())

        model.add(fPermute((3, 0, 1, 2)))

        model.add(Lambda(function=kaggle_MultiRotMergeLayer_output,
                         output_shape=lambda x: (
                             x[0] // 4 // N_INPUT_VARIATION, (x[1] * x[2]
                                                              * x[3] * 4
                                                              * N_INPUT_VARIATION)),
                         arguments={'num_views': N_INPUT_VARIATION}))

        model.add(Dropout(0.5))
        model.add(MaxoutDense(output_dim=2048, nb_feature=2,
                              weights=dense_weight_init_values(
                                  model.output_shape[-1], 2048, nb_feature=2)))

        model.add(Dropout(0.5))
        model.add(MaxoutDense(output_dim=2048, nb_feature=2,
                              weights=dense_weight_init_values(
                                  model.output_shape[-1], 2048, nb_feature=2)))

        model.add(Dropout(0.5))
        model.add(Dense(output_dim=37, weights=dense_weight_init_values(
            model.output_shape[-1], 37, w_std=0.01, b_init_val=0.1)))

        model_seq = model([input_tensor, input_tensor_45])

        CATEGORISED = False  # FXME has to be implemented

        output_layer_norm = Lambda(function=OptimisedDivGalaxyOutput,
                                   output_shape=lambda x: x,
                                   arguments={'normalised': True,
                                              'categorised': CATEGORISED})(model_seq)
        output_layer_noNorm = Lambda(function=OptimisedDivGalaxyOutput,
                                     output_shape=lambda x: x,
                                     arguments={'normalised': False,
                                                'categorised': CATEGORISED})(model_seq)

        model_norm = Model(
            input=[input_tensor, input_tensor_45], output=output_layer_norm)
        model_norm_metrics = Model(
            input=[input_tensor, input_tensor_45], output=output_layer_norm)
        model_noNorm = Model(
            input=[input_tensor, input_tensor_45], output=output_layer_noNorm)

        self.models = {'model_norm': model_norm,
                       'model_norm_metrics': model_norm_metrics,
                       'model_noNorm': model_noNorm}

        return self.models

    def compile_models(self):
        self.models['model_norm'].compile(loss='mean_squared_error',
                                          optimizer=SGD(
                                              lr=self.LEARNING_RATE_SCHEDULE[0],
                                              momentum=self.MOMENTUM,
                                              nesterov=True))
        self.models['model_noNorm'].compile(loss='mean_squared_error',
                                            optimizer=SGD(
                                                lr=self.LEARNING_RATE_SCHEDULE[0],
                                                momentum=self.MOMENTUM,
                                                nesterov=True))

        self.models['model_norm_metrics'].compile(loss='mean_squared_error',
                                                  optimizer=SGD(
                                                      lr=self.LEARNING_RATE_SCHEDULE[0],
                                                      momentum=self.MOMENTUM,
                                                      nesterov=True),
                                                  metrics=[rmse,
                                                           'categorical_accuracy',
                                                           kaggle_sliced_accuracy])

        self.init_hist_dics(self.models)

        return True

    def print_summary(self, modelname='model_norm'):
        self.models[modelname].summary()
        return True

    def load_weights(self, path, modelname='model_norm'):
        self.models[modelname].load_weights(path)
        with open(path, 'a')as f:
            f.write('#loaded weights from ' + path +
                    ' into  model ' + modelname + '\n')
        return True

    def print_last_hist(self, modelname='model_norm_metrics'):
        for n in self.hists['model_norm_metrics']:
            print "   %s : %.3f" % (
                n, self.hists['model_norm_metrics'][n][-1])
        return True

    def evaluate(self, x, y_valid, batch_size=0,
                 modelname='model_norm_metrics', verbose=1):
        if not batch_size:
            batch_size = self.BATCH_SIZE
        evalHist = self.models[modelname].evaluate(
            x=x, y=y_valid, batch_size=batch_size,
            verbose=verbose)

        for i in range(len(self.models[modelname].metrics_names)):
            self.hists[modelname][self.models[modelname].metrics_names[i]].append(
                evalHist[i])

        if verbose:
            self.print_last_hist()

        return evalHist

    def make_lrs_fct(self):
        return functools.partial(lr_function,
                                 lrs=self.LEARNING_RATE_SCHEDULE)

    def save_hist(self, history, modelname='model_norm'):
        if not self.hists:
            self.init_hist_dics(self.models)
        for k in self.hists[modelname]:
            self.hists[modelname][k] += history[k]

        return True

    def fit_gen(self, modelname, data_generator, validation, samples_per_epoch,
                callbacks='default', nb_epoch=1):  # think about making nb_worker>1 work, problem: generator needs multiple instances
        if callbacks == 'default':
            _callbacks = [LearningRateScheduler(self.make_lrs_fct())]
        else:
            _callbacks = callbacks

        hist = self.models[modelname].fit_generator(data_generator,
                                                    validation_data=validation,
                                                    samples_per_epoch=samples_per_epoch,
                                                    nb_epoch=nb_epoch,
                                                    verbose=1,
                                                    callbacks=_callbacks)

        self.save_hist(hist.history, modelname=modelname)

        return hist

    def save_weights(self, modelname='model_norm', path=''):
        if not path:
            path = self.WEIGHTS_PATH
        self.models[modelname].save_weights(path)
        return path

    def save_loss(self, path='', modelname=''):
        if not path:
            path = self.LOSS_PATH
        with open(path, 'a')as f:
            if not self.first_loss_save:
                f.write("#eval losses and metrics:\n")
                self.first_loss_save = True
            if modelname:
                f.write("#history of model: " + modelname + '\n')
                json.dump(self.hists[modelname], f)
            else:
                f.write("#histories of all models:\n")
                for k in self.models:
                    f.write("#history of model: " + k + '\n')
                    json.dump(self.hists[k], f)
            f.write("\n")

        return True

    def save(self):
        self.save_weights()
        self.save_loss(modelname='model_norm_metrics')
        self.save_loss(modelname='model_norm')
        return True

    def full_fit(self, data_gen, validation, samples_per_epoch,
                 validate_every,
                 nb_epochs, verbose=1, save_at_every_validation=True):
        if verbose:
            timedeltas = []
        epochs_run = 0
        epoch_togo = nb_epochs
        for i in range(nb_epochs / validate_every if not nb_epochs
                       % validate_every else nb_epochs / validate_every + 1):
            if verbose:
                time1 = time.time()
                print ''
                print "epochs run: %s - epochs to go: %s " % (
                    epochs_run, epoch_togo)

            # main fit
            hist = self.models['model_norm'].fit_generator(
                data_gen, validation_data=validation,
                samples_per_epoch=samples_per_epoch,
                nb_epoch=np.min([
                    epoch_togo, validate_every]) + epochs_run,
                initial_epoch=epochs_run, verbose=1,
                callbacks=[LearningRateScheduler(self.make_lrs_fct())])

            self.save_hist(hist.history)

            epoch_togo -= np.min([epoch_togo, validate_every])
            epochs_run += np.min([epoch_togo, validate_every])

            if verbose:
                print ''
                print 'validate:'
            self.evaluate(
                validation[0], y_valid=validation[1], verbose=verbose)

            if verbose:
                timedeltas.append(time.time() - time1)
                if len(timedeltas) > 10:
                    timedeltas = timedeltas[-10:]
                print '\nestimated finish: %s \n' % (
                    datetime.now() + timedelta(
                        seconds=(
                            np.mean(timedeltas)
                            * epoch_togo / validate_every)))

            if save_at_every_validation:
                self.save()
