import numpy as np
import json
import warnings
import time
from datetime import datetime, timedelta
import functools

from keras import backend as T
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam
# TODO will be removed from keras2, can this be achieved with a lambda
# layer now? looks like it:
# https://stackoverflow.com/questions/43160181/keras-merge-layer-warning
from keras.layers import Merge
from keras.callbacks import LearningRateScheduler
from keras.engine.topology import InputLayer
from keras import initializers

from keras_extra_layers import kerasCudaConvnetPooling2DLayer, fPermute, kerasCudaConvnetConv2DLayer, MaxoutDense
from custom_for_keras import kaggle_MultiRotMergeLayer_output, OptimisedDivGalaxyOutput, kaggle_input, sliced_accuracy_mean, sliced_accuracy_std, dense_weight_init_values, rmse, lr_function
from lsuv_init import LSUVinit


'''
This class contains the winning solution model of the kaggle galaxies contest transferred to keras and function to fit it.

'''


class kaggle_winsol:
    '''
   Arguments:
    BATCH_SIZE: image fitted at the same time
    NUM_INPUT_FEATURES: features per pixel, typ. colors
    PART_SIZE: integer pixel heigth and width to which the input is cutted
    input_sizes: list of tupel of int. pixel sizes of the input variations
    LEARNING_RATE_SCHEDULE: learning rate schedule for SGD as epochs: learning rate dictionary
    MOMENTUM: nestrenov momentum
    LOSS_PATH: save path for the loss and validation history
    WEIGHTS_PATH: load/save path of the model weights
    '''

    def __init__(self, BATCH_SIZE, NUM_INPUT_FEATURES, PART_SIZE, input_sizes,
                 include_flip=True,
                 LEARNING_RATE_SCHEDULE=None, MOMENTUM=None, LOSS_PATH='./',
                 WEIGHTS_PATH='./'):
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_INPUT_FEATURES = NUM_INPUT_FEATURES
        self.input_sizes = input_sizes
        self.PART_SIZE = PART_SIZE
        self.LEARNING_RATE_SCHEDULE = LEARNING_RATE_SCHEDULE if LEARNING_RATE_SCHEDULE else [
            0.1]
        self.current_lr = self.LEARNING_RATE_SCHEDULE[0]
        self.MOMENTUM = MOMENTUM
        self.WEIGHTS_PATH = WEIGHTS_PATH
        self.LOSS_PATH = LOSS_PATH
        self.hists = {}
        self.first_loss_save = True
        self.models = {}
        self.reinit_counter = 0
        self.include_flip = include_flip

        self.layer_formats = {'conv_0': 1, 'conv_1': 1, 'conv_2': 1,
                              'conv_3': 1, 'pool_0': 2, 'pool_1': 2,
                              'pool_2': 2, 'conv_out_merge': -1,
                              'maxout_0': 0, 'maxout_1': 0,
                              'dense_output': 0}

    '''
    initialize loss and validation histories

    Arguments:
        model_in: collection of models, if None takes all models in the class

    Returns:
        empty dictionary of empty history dictionaries
    '''

    def _init_hist_dics(self, model_in=None):
        if model_in:
            _model_in = model_in
        else:
            _model_in = self.models

        self.hists = {}

        for n in _model_in:
            self.hists[n] = {}
            try:
                for o in _model_in[n].metrics_names:
                    self.hists[n][o] = []
            except AttributeError:
                pass

        return self.hists

    '''
    compliles all available models
    initilises loss histories
    '''

    def _compile_models(self, postfix=''):
        self.models['model_norm' + postfix].compile(loss='mean_squared_error',
                                                    optimizer=SGD(
                                                        lr=self.LEARNING_RATE_SCHEDULE[0],
                                                        momentum=self.MOMENTUM,
                                                        nesterov=bool(self.MOMENTUM)))
        self.models['model_noNorm' + postfix].compile(loss='mean_squared_error',
                                                      optimizer=SGD(
                                                          lr=self.LEARNING_RATE_SCHEDULE[0],
                                                          momentum=self.MOMENTUM,
                                                          nesterov=bool(self.MOMENTUM)))

        self.models['model_norm_metrics' + postfix].compile(loss='mean_squared_error',
                                                            optimizer=SGD(
                                                                lr=self.LEARNING_RATE_SCHEDULE[0],
                                                                momentum=self.MOMENTUM,
                                                                nesterov=bool(self.MOMENTUM)),
                                                            metrics=[rmse,
                                                                     'categorical_accuracy',
                                                                     sliced_accuracy_mean,
                                                                     sliced_accuracy_std])

        self._init_hist_dics(self.models)

        return True

    '''
    initiates models according to the kaggle galaxies winning solution

    Returns:
    dictinary with the model without normalisation, with normalisation and with normalisation and extra metrics for validation
    '''

    def init_models(self):
        print "init model"
        input_tensor = Input(batch_shape=(self.BATCH_SIZE,
                                          self.NUM_INPUT_FEATURES,
                                          self.input_sizes[0][0],
                                          self.input_sizes[0][1]),
                             dtype='float32', name='input_tensor')

        input_tensor_45 = Input(batch_shape=(self.BATCH_SIZE,
                                             self.NUM_INPUT_FEATURES,
                                             self.input_sizes[1][0],
                                             self.input_sizes[1][1]),
                                dtype='float32', name='input_tensor_45')

        input_lay_0 = InputLayer(batch_input_shape=(
            self.BATCH_SIZE, self.NUM_INPUT_FEATURES, self.input_sizes[0][0],
            self.input_sizes[0][1]),
            name='input_lay_seq_0')

        input_lay_1 = InputLayer(batch_input_shape=(
            self.BATCH_SIZE, self.NUM_INPUT_FEATURES, self.input_sizes[1][0],
            self.input_sizes[1][1]),
            name='input_lay_seq_1')

        model = Sequential(name='main_seq')

        N_INPUT_VARIATION = 2  # depends on the kaggle input settings
        include_flip = self.include_flip

        num_views = N_INPUT_VARIATION * (2 if include_flip else 1)

        model.add(Merge([input_lay_0, input_lay_1], mode=kaggle_input,
                        output_shape=lambda x: ((input_lay_0.output_shape[0]
                                                 + input_lay_1.output_shape[0]) * 2
                                                * N_INPUT_VARIATION,
                                                self.NUM_INPUT_FEATURES,
                                                self.PART_SIZE,
                                                self.PART_SIZE),
                        arguments={'part_size': self.PART_SIZE,
                                   'n_input_var': N_INPUT_VARIATION,
                                   'include_flip': include_flip,
                                   'random_flip': False}, name='input_merge'))

        # needed for the pylearn moduls used by kerasCudaConvnetConv2DLayer and
        # kerasCudaConvnetPooling2DLayer
        model.add(fPermute((1, 2, 3, 0), name='input_perm'))

        model.add(kerasCudaConvnetConv2DLayer(
            n_filters=32, filter_size=6, untie_biases=True, name='conv_0'))
        model.add(kerasCudaConvnetPooling2DLayer(name='pool_0'))

        model.add(kerasCudaConvnetConv2DLayer(
            n_filters=64, filter_size=5, untie_biases=True, name='conv_1'))
        model.add(kerasCudaConvnetPooling2DLayer(name='pool_1'))

        model.add(kerasCudaConvnetConv2DLayer(
            n_filters=128, filter_size=3, untie_biases=True, name='conv_2'))
        model.add(kerasCudaConvnetConv2DLayer(n_filters=128,
                                              filter_size=3,  weights_std=0.1,
                                              untie_biases=True, name='conv_3'))

        model.add(kerasCudaConvnetPooling2DLayer(name='pool_2'))

        model.add(fPermute((3, 0, 1, 2), name='cuda_out_perm'))

        model.add(Lambda(function=kaggle_MultiRotMergeLayer_output,
                         output_shape=lambda x: (
                             x[0] // 4 // N_INPUT_VARIATION, (x[1] * x[2]
                                                              * x[3] * 4
                                                              * num_views)),
                         arguments={'num_views': num_views}, name='conv_out_merge'))

        model.add(Dropout(0.5))
        model.add(MaxoutDense(output_dim=2048, nb_feature=2,
                              weights=dense_weight_init_values(
                                  model.output_shape[-1], 2048, nb_feature=2), name='maxout_0'))

        model.add(Dropout(0.5))
        model.add(MaxoutDense(output_dim=2048, nb_feature=2,
                              weights=dense_weight_init_values(
                                  model.output_shape[-1], 2048, nb_feature=2), name='maxout_1'))

        model.add(Dropout(0.5))
        model.add(Dense(units=37, activation='relu',
                        kernel_initializer=initializers.RandomNormal(
                            stddev=0.01),
                        bias_initializer=initializers.Constant(value=0.1),
                        name='dense_output'))

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
            inputs=[input_tensor, input_tensor_45], outputs=output_layer_norm, name='full_model_norm')
        model_norm_metrics = Model(
            inputs=[input_tensor, input_tensor_45], outputs=output_layer_norm, name='full_model_metrics')
        model_noNorm = Model(
            inputs=[input_tensor, input_tensor_45], outputs=output_layer_noNorm, name='full_model_noNorm')

        self.models = {'model_norm': model_norm,
                       'model_norm_metrics': model_norm_metrics,
                       'model_noNorm': model_noNorm}

        self._compile_models()

        return self.models

    def init_models_ellipse(self, input_shape=3):
        print "init model"
        input_tensor = Input(batch_shape=(self.BATCH_SIZE,
                                          input_shape),
                             dtype='float32', name='input_tensor')

        model = Sequential(name='main_seq')

        model.add(Dropout(0.5, input_shape=(input_shape,)))
        model.add(MaxoutDense(output_dim=2048, nb_feature=2,
                              weights=dense_weight_init_values(
                                  model.output_shape[-1], 2048, nb_feature=2), name='maxout_0'))

        model.add(Dropout(0.5))
        model.add(MaxoutDense(output_dim=2048, nb_feature=2,
                              weights=dense_weight_init_values(
                                  model.output_shape[-1], 2048, nb_feature=2), name='maxout_1'))

        model.add(Dropout(0.5))
        model.add(Dense(units=37, activation='relu',
                        kernel_initializer=initializers.RandomNormal(
                            stddev=0.01),
                        bias_initializer=initializers.Constant(value=0.1),
                        name='dense_output'))

        model_seq = model([input_tensor])

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
            inputs=[input_tensor], outputs=output_layer_norm, name='full_model_norm_ellipse')
        model_norm_metrics = Model(
            inputs=[input_tensor], outputs=output_layer_norm, name='full_model_metrics_ellipse')
        model_noNorm = Model(
            inputs=[input_tensor], outputs=output_layer_noNorm, name='full_model_noNorm_ellipse')

        self.models = {'model_norm_ellipse': model_norm,
                       'model_norm_metrics_ellipse': model_norm_metrics,
                       'model_noNorm_ellipse': model_noNorm}

        self._compile_models(postfix='_ellipse')

        return self.models

    '''
    Arguments:
    modelname: name of the model to be printed
    '''

    def print_summary(self, modelname='model_norm'):
        self.models[modelname].summary()
        return True

    '''
    loads previously saved weights

    Arguments:
    path: path to savefile
    modelname: name of the model for which the weights are loaded, in default the models use all the same weight
    '''

    def load_weights(self, path, modelname='model_norm', postfix=''):
        modelname = modelname + postfix
        self.models[modelname].load_weights(path)
        with open(path, 'a')as f:
            f.write('#loaded weights from ' + path +
                    ' into  model ' + modelname + '\n')
        return True

    '''
    load weights from the winning solution

    Arguments:
    path: path to savefile
    modelname: name of the model for which the weights are loaded, in default the models use all the same weight
    '''

    def getWinSolWeights(self,
                         modelname='model_norm',
                         path="analysis/final/try_convent_gpu1_win_sol_net_on_0p0775_validation.pkl",
                         debug=False):
        analysis = np.load(path)
        try:
            l_weights = analysis['param_values']
        except KeyError, e:
            print 'KeyError %s in the analysis loaded from %s. \n Available keys are: %s' % (
                e, path, analysis.keys()
            )
            raise KeyError(e)
        # w_pairs=[]
        # for i in range(len(l_weights)/2):
        #	w_pairs.append([l_weights[2*i],l_weights[2*i+1]])
        w_kSorted = []
        for i in range(len(l_weights) / 2):
            w_kSorted.append(l_weights[-2 - 2 * i])
            w_kSorted.append(l_weights[-1 - 2 * i])
        w_load_worked = False

        if debug:
            print 'imported:'
            print len(w_kSorted)
            print np.shape(w_kSorted)

        def _load_direct():
            for l in self.models[modelname].layers:
                if debug:
                    print '---'
                if debug:
                    print 'found place'
                    print len(l.get_weights())
                    print np.shape(l.get_weights())
                l_weights = l.get_weights()
                if len(l_weights) == len(w_kSorted):
                    if debug:
                        for i in range(len(l_weights)):
                            print type(l_weights[i])
                            print "load %s into %s" % (np.shape(w_kSorted[i]),
                                                       np.shape(l_weights[i]))
                    try:
                        l.set_weights(w_kSorted)
                        return True
                    except ValueError:
                        print "found matching layer length but no matching weights in direct try"
                        return False

        def _load_maxout():
            for l in self.models[modelname].layers:
                l_weights = l.get_weights()
                if len(l_weights) == len(w_kSorted):
                    for i in range(len(l_weights)):
                        if np.shape(l_weights[i]) != np.shape(w_kSorted[i]):
                            if debug:
                                print "reshaping weights of layer %s" % i
                            shape = np.shape(w_kSorted[i])
                            w_kSorted[i] = np.reshape(w_kSorted[i],
                                                      (2, shape[0] / 2) if len(
                                shape) == 1
                                else (2,) + shape[0:-1]
                                + (shape[-1] / 2,))
                            if debug:
                                print "load %s into %s" % (np.shape(w_kSorted[i]),
                                                           np.shape(l_weights[i]))
                    try:
                        l.set_weights(w_kSorted)
                        return True
                    except ValueError:
                        print "found matching length and tried to reshape weights for maxout layers: did not work"
                        return False
                elif len(l_weights) == len(w_kSorted) + 4:  # import for keras 2
                    j = 0
                    for i, lay_in in enumerate(l.layers):
                        if len(lay_in.get_weights()) == 0:
                            continue
                        if len(lay_in.get_weights()) == 2:
                            w_kern = w_kSorted[j]
                            j += 1
                            w_bias = w_kSorted[j]
                            j += 1
                            if type(lay_in) == MaxoutDense:
                                if debug:
                                    print np.shape(w_kern)
                                    print np.shape(w_bias)
                                w_kern = np.reshape(w_kern, (2, np.shape(w_kern)[
                                    0], np.shape(w_kern)[1] / 2))
                                w_bias = np.reshape(
                                    w_bias, (2, np.shape(w_bias)[0] / 2))
                                if 2 * np.shape(w_kern)[1] == np.shape(
                                        lay_in.get_weights()[0])[1]:
                                    w_kern = np.concatenate(
                                        (w_kern, w_kern), 1)
                                    if debug:
                                        print "concatenated the w_kern two times, imported weigths seem not to have been with maxout dense"
                                # elif debug:
                                #     print 'it did not come to the doubling of the maxout weights'
                                # print '%s != %s' % (2 * np.shape(w_kern[1]))
                            try:
                                lay_in.set_weights([w_kern, w_bias])
                            except ValueError, e:
                                print 'WARNING: Setting weights did not work in keras 2 style!'
                                print " tried to load shapes  %s,%s into %s,%s" % (
                                    np.shape(w_kern), np.shape(w_bias),
                                    np.shape(lay_in.get_weights()[0]),
                                    np.shape(lay_in.get_weights()[1]))
                                print e
                                return False
                    return bool(j)

        w_load_worked = _load_direct()
        if not w_load_worked:
            w_load_worked = _load_maxout()
            if not w_load_worked:
                print "no matching weight length were found"
            else:
                print "reshaped weights from maxout via dense and dropout to real maxout"

        return w_load_worked

    '''
    prints the loss and metric information of a model

    '''

    def print_last_hist(self, modelname='model_norm_metrics', postfix=''):
        modelname += postfix
        print ''
        for n in self.hists[modelname]:
            print "   %s : %.3f" % (
                n, self.hists[modelname][n][-1])
        return True

    '''
    evaluates a model according to true answeres, saves the information in the history
    Arguments:
    x: input sample
    y_valid: true answeres
    batch_size: inputs evaluated at the same time, default uses batch size from class initialisation
    verbose: interger, set to 0 to minimize oputput
    '''

    def evaluate(self, x, y_valid, batch_size=None,
                 modelname='model_norm_metrics', verbose=1, postfix=''):
        modelname = modelname + postfix
        if not batch_size:
            batch_size = self.BATCH_SIZE
        evalHist = self.models[modelname].evaluate(
            x=x, y=y_valid, batch_size=batch_size,
            verbose=verbose)

        for i in range(len(self.models[modelname].metrics_names)):
            self.hists[modelname][self.models[modelname].metrics_names[i]].append(
                evalHist[i])

        if verbose:
            self.print_last_hist(postfix=postfix)

        return evalHist

    def predict(self, x, batch_size=None,
                modelname='model_norm_metrics', verbose=1, postfix=''):
        modelname += postfix
        if not batch_size:
            batch_size = self.BATCH_SIZE
        predictions = self.models[modelname].predict(
            x=x, batch_size=batch_size,
            verbose=verbose)

        return predictions

    def _make_lrs_fct(self):
        return functools.partial(lr_function,
                                 lrs=self.LEARNING_RATE_SCHEDULE)

    def _save_hist(self, history, modelname='model_norm', postfix=''):
        modelname += postfix
        if not self.hists:
            self._init_hist_dics(self.models)
        for k in self.hists[modelname]:
            self.hists[modelname][k] += history[k]

        return True

    '''
    performs the fit

    Arguments:
    modelname: string name of the model to be fittet
    data_generator: generator that yields the input data
    validation: validation set as tuple of validation data and solution
    samples_per_epoch: integer number of input samples per epoch, use this also to not run over the whole set
    callbacks: list of callbacks to be excecuted, default uses learning rate schedule. more information at www.keras.io
    nb_epoch: number of epochs to be fitted
    '''

    def fit_gen(self, modelname, data_generator, validation, samples_per_epoch,
                callbacks='default', nb_epoch=1):  # think about making nb_worker>1 work, problem: generator needs multiple instances
        if callbacks == 'default':
            _callbacks = [LearningRateScheduler(self._make_lrs_fct())]
        else:
            _callbacks = callbacks

        # FIXME think about how to handle the missing samples%batch_size
        # samples
        steps_per_epoch = samples_per_epoch // self.BATCH_SIZE

        hist = self.models[modelname].fit_generator(data_generator,
                                                    validation_data=validation,
                                                    steps_per_epoch=steps_per_epoch,
                                                    epochs=nb_epoch,
                                                    verbose=1,
                                                    callbacks=_callbacks)

        self._save_hist(hist.history, modelname=modelname)

        return hist

    '''
    saves the modelweights as hdf5 file
    Arguments:
    path: the path were the weights are to be saved, if default the WEIGHTS_PATH with which the class was initialised is used
    modelname: name of the model, default allmodels have the same weights
    '''

    def save_weights(self, path='', modelname='model_norm', postfix=''):
        modelname += postfix
        if not path:
            path = self.WEIGHTS_PATH
        self.models[modelname].save_weights(path)
        return path

    '''
    saves the loss and validation metric histories as json strings in a text file
    Arguments:
    path: default uses LOSS_PATH from initialisation
    modelname: default saves history of all models
    '''

    def save_loss(self, path='', modelname='', postfix=''):
        if not path:
            path = self.LOSS_PATH
        if self.first_loss_save:
            with open(path, 'a')as f:
                f.write("#eval losses and metrics:\n")
                if modelname:
                    f.write("#history of model: " + modelname + '\n')
                    json.dump(self.hists[modelname], f)
                else:
                    f.write("#histories of all models:\n")
                    for k in self.models:
                        f.write("#history of model: " + k + '\n')
                        json.dump(self.hists[k], f)
                f.write("\n")
            self.first_loss_save = False
        else:
            if modelname:
                with open(path, "r+") as f:
                    d = f.readlines()
                    f.seek(0)
                    rewrite_next_json = False
                    model_found = False
                    for i in d:
                        if i != "#history of model: " + modelname + '\n':
                            if rewrite_next_json:
                                if i.find("{", 0, 1) != -1:
                                    json.dump(self.hists[modelname], f)
                                    rewrite_next_json = False
                                else:
                                    print 'WARNING: loss history save file is not in the expected stats'
                                    json.dump(self.hists[modelname], f)
                                    rewrite_next_json = False
                                    f.write(i)
                            else:
                                f.write(i)
                        else:
                            f.write(i)
                            model_found = True
                            rewrite_next_json = True
                    if not model_found:
                        f.write("#history of model: " + modelname + '\n')
                        json.dump(self.hists[modelname], f)
                    f.write('\n')
            else:
                for k in self.models:
                    self.save_loss(path=path, modelname=k)

        return True

    def _load_one_loss(self, path, modelname):
        loss_hist = {}
        with open(path, 'r') as f:
            d = (i for i in f.readlines()[::-1])
            for line in d:
                if line.find("{", 0, 1) != -1:
                    loss_hist = json.loads(line)
                if d.next() == "#history of model: " + modelname + '\n':
                    break
            if not loss_hist:
                raise Warning('No model %s was found in %s' %
                              (modelname, path))
        return loss_hist

    def load_loss(self, path='', modelname=''):
        if not path:
            path = self.LOSS_PATH
        if modelname:
            return self._load_one_loss(path, modelname)
        else:
            return [self._load_one_loss(path, name)for name in self.models]

    '''
    performs all saving task
    '''

    def save(self, option_string=None, postfix=''):
        if not option_string:
            self.save_weights(postfix=postfix)
            self.save_loss(modelname='model_norm_metrics' + postfix)
            self.save_loss(modelname='model_norm' + postfix)
        elif option_string == 'interrupt':
            self.save_weights(path=self.WEIGHTS_PATH +
                              '_interrupted.h5', postfix=postfix)
            self.save_loss(path=self.LOSS_PATH + '_interrupted.txt',
                           modelname='model_norm_metrics' + postfix)
            self.save_loss(path=self.LOSS_PATH + '_interrupted.txt',
                           modelname='model_norm' + postfix)
        else:
            print 'WARNING: unknown saving opotion *' + option_string + '*'
            self.save(postfix=postfix)
        return True

    '''
    main fitting function that calls a metrics model in addition to the validation after every epoch

    Arguments:
    data_gen: generator that yields the input samples
    validation: tuple of validation data and solution
    samples_per_epoch: number of samples in the fit data set
    validate_every: number of epochs after which the extra metrics are calculated on the validation sample
    nb_epochs: number of epochs to be fitted
    verbose: set to 0 to minimize output
    save_every_validation: save the losses and metrics after every 'validate_every' epochs, weights are overwritten
    '''

    def full_fit(self, data_gen, validation, samples_per_epoch,
                 validate_every,
                 nb_epochs, verbose=1, save_at_every_validation=True,
                 data_gen_creator=None, postfix=''):
        if verbose:
            timedeltas = []
        epochs_run = 0
        epoch_togo = nb_epochs

        # FIXME think about how to handle the missing samples%batch_size
        # samples
        steps_per_epoch = samples_per_epoch // self.BATCH_SIZE

        for i in range(nb_epochs / validate_every if not nb_epochs
                       % validate_every else nb_epochs / validate_every + 1):
            if verbose:
                time1 = time.time()
                print ''
                print "epochs run: %s - epochs to go: %s " % (
                    epochs_run, epoch_togo)

            # main fit
            def _main_fit(
                    self,
                    data_gen, validation_data=validation,
                    samples_per_epoch=samples_per_epoch,
                    nb_epoch=np.min([
                        epoch_togo, validate_every]) + epochs_run,
                    initial_epoch=epochs_run, verbose=1,
                    callbacks=[LearningRateScheduler(self._make_lrs_fct())],
                    data_gen_creator=data_gen_creator, postfix=postfix):
                try:
                    hist = self.models['model_norm' + postfix].fit_generator(
                        data_gen, validation_data=validation_data,
                        steps_per_epoch=steps_per_epoch,
                        epochs=nb_epoch,
                        initial_epoch=initial_epoch, verbose=verbose,
                        callbacks=callbacks)
                    self._save_hist(hist.history, postfix=postfix)
                except ValueError:
                    warnings.warn(
                        'Value Error in the main fit. Generator will be reinitialised.')
                    print 'saving'
                    self.save(postfix=postfix)
                    if data_gen_creator:
                        _main_fit(self, data_gen=data_gen_creator())
                    else:
                        raise ValueError(
                            'no reinitilizer of the data generator defined')

            _main_fit(self, data_gen)

            epoch_togo -= np.min([epoch_togo, validate_every])
            epochs_run += np.min([epoch_togo, validate_every])

            if verbose:
                print ''
                print 'validate:'

            self.evaluate(
                validation[0], y_valid=validation[1], verbose=verbose, postfix=postfix)

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
                self.save(postfix=postfix)

    def LSUV_init(self, train_batch, batch_size=None, modelname='model_norm', postfix='',
                  sub_modelname='main_seq'):
        modelname = modelname + postfix
        if not batch_size:
            batch_size = self.BATCH_SIZE
        LSUVinit(self.models[modelname].get_layer(sub_modelname),
                 train_batch, batch_size=batch_size)

    def reinit(self, WEIGHTS_PATH=None, LOSS_PATH=None):
        self.reinit_counter += 1
        if WEIGHTS_PATH:
            self.WEIGHTS_PATH = WEIGHTS_PATH
        else:
            self.WEIGHTS_PATH = ((self.WEIGHTS_PATH.split(
                '.', 1)[0] + '_' + str(self.reinit_counter) + '.h5'))

        if LOSS_PATH:
            self.LOSS_PATH = LOSS_PATH
        else:
            self.LOSS_PATH = ((self.LOSS_PATH.split(
                '.', 1)[0] + '_' + str(self.reinit_counter) + '.h5'))

        self.first_loss_save = True

        self.init_models()

        return True

    def get_layer_output(self, layer, input_=None, modelname='model_norm',
                         main_layer='main_seq', prediction_batch_size=1):
        _layer = self.models[modelname].get_layer(main_layer).get_layer(
            layer)

        if not input_:
            input_ = [np.ones(shape=(prediction_batch_size,) + i[1:])
                      for i in self.models[modelname].get_layer(main_layer).input_shape]

        if self.layer_formats[layer] > 0:
            output_layer = fPermute((3, 0, 1, 2))(_layer.output)
            output_layer = Lambda(lambda x: T.reshape(x[0], (
                prediction_batch_size,) + tuple(T.shape(output_layer)[1:])),
                output_shape=lambda input_shape: (
                prediction_batch_size,) + input_shape[1:])(output_layer)
        else:
            try:
                output_layer = _layer.output
            except AttributeError:
                print 'debug infos after Attribute error'
                print layer
                print _layer
                raise AttributeError

        intermediate_layer_model = Model(inputs=self.models[modelname]
                                         .get_layer(main_layer).get_input_at(0),
                                         outputs=output_layer)
        return intermediate_layer_model.predict(input_,
                                                batch_size=prediction_batch_size)

    def get_layer_weights(self, layer,  modelname='model_norm',
                          main_layer='main_seq'):
        if type(layer) == int:
            ret_weights = self.models[modelname].get_layer(
                main_layer).layers[layer].get_weights()
        elif type(layer) == str:
            ret_weights = self.models[modelname].get_layer(main_layer).get_layer(
                layer).get_weights()
        else:
            raise ValueError('layer must be specified as int or string')
        return ret_weights
