import numpy as np
import json
import warnings
import time
from datetime import datetime, timedelta
import functools

import h5py

from keras import backend as T
from keras.models import Model
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler

from keras_extra_layers import fPermute
from custom_for_keras import sliced_accuracy_mean, sliced_accuracy_std, rmse,\
    lr_function
from lsuv_init import LSUVinit


class kaggle_base(object):
    '''
   Arguments:
    BATCH_SIZE: image fitted at the same time
    LEARNING_RATE_SCHEDULE: learning rate schedule for SGD as epochs: learning rate dictionary
    MOMENTUM: nestrenov momentum
    LOSS_PATH: save path for the loss and validation history
    WEIGHTS_PATH: load/save path of the model weights
    '''

    def __init__(self,
                 BATCH_SIZE,
                 LEARNING_RATE_SCHEDULE=None,
                 MOMENTUM=None,
                 LOSS_PATH='./',
                 WEIGHTS_PATH='./',
                 debug=False,
                 ** kwargs):
        self.BATCH_SIZE = BATCH_SIZE
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
        self.debug = debug

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

        self.first_loss_save = {'allsave': self.first_loss_save}
        for k in self.models.keys():
            self.first_loss_save[k] = True

        return self.hists

    '''
    compliles all available models
    initilises loss histories
    '''

    def _compile_models(self,
                        optimizer=None,
                        loss='mean_squared_error',
                        postfix=''):
        if not self.models:
            raise ValueError('Did not find any models to compile')

        if not optimizer:
            optimizer = SGD(
                lr=self.LEARNING_RATE_SCHEDULE[0],
                momentum=self.MOMENTUM,
                nesterov=bool(self.MOMENTUM))
        try:
            self.models['model_norm' + postfix].compile(
                loss=loss,
                optimizer=optimizer)
        except KeyError:
            pass
        try:
            self.models['model_noNorm' + postfix].compile(
                loss=loss,
                optimizer=optimizer)
        except KeyError:
            pass
        try:
            self.models['model_norm_metrics' + postfix].compile(
                loss=loss,
                optimizer=optimizer,
                metrics=[rmse,
                         'categorical_accuracy',
                         sliced_accuracy_mean,
                         sliced_accuracy_std])

        except KeyError:
            pass
        self._init_hist_dics(self.models)

        return True

    '''
    initiates models according to the kaggle galaxies winning solution

    Returns:
    dictinary with the model without normalisation, with normalisation and with normalisation and extra metrics for validation
    '''

    def init_models(self):
        pass

    '''
    Arguments:
    modelname: name of the model to be printed
    '''

    def print_summary(self, modelname='model_norm', postfix=''):
        modelname += postfix
        self.models[modelname].summary()
        return True

    def load_one_layers_weight(self, path, layername_source, layername_this='',
                               modelname='model_norm',
                               sub_modelname='main_seq',
                               postfix=''):
        modelname = modelname + postfix

        if not type(layername_source) == list:
            layername_source = [layername_source]
        if not layername_this:
            layername_this = layername_source
        elif not type(layername_this) == list:
            layername_this == [layername_source]

        file_ = h5py.File(path, 'r')

        for ls, lt in zip(layername_source, layername_this):
            if self.debug:
                print
                print 'loading weights from layer %s to layer %s' % (ls, lt)
            try:
                weight = file_[sub_modelname][ls]
            except KeyError, e:
                print
                print 'KeyError'
                print '\ttried key %s' % sub_modelname
                print '\tpossible keys are: %s' % file_.keys()
                print
                raise KeyError(e)

            if self.debug:
                print 'keys in source weight object %s' % weight.keys()
                print '\t shapes: %s' % [np.shape(weight[n]) for n
                                         in weight.keys()]

            weight = [np.array(weight[n]) for n in weight.keys()]

            # if self.debug:
            #     print '\t %s' % repr(np.shape(weight[0]))
            #     # if ls == 'conv_3':
            #     for w in weight:
            #         print np.shape(w)
            #         print np.shape(w[0])

            self.models[modelname].get_layer(
                sub_modelname).get_layer(lt).set_weights(weight)

        file_.close()
        if self.debug:
            print
        with open(self.LOSS_PATH, 'a')as f:
            f.write('#loaded weights of layer(s) ' + str(layername_source) + '  from '
                    + str(path) + ' into  model ' +
                    str(modelname) + ' into the layer(s) '
                    + str(layername_this) + '\n')
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
        with open(self.LOSS_PATH, 'a')as f:
            f.write('#loaded weights from ' + path +
                    ' into  model ' + modelname + '\n')
        return True
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
            self.hists[modelname][self.models[modelname].metrics_names[i]]\
                .append(evalHist[i])

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
        modelname += postfix
        if not path:
            path = self.LOSS_PATH
        if self.first_loss_save['allsave'] and\
           (not modelname or self.first_loss_save[modelname]):
            with open(path, 'a')as f:
                f.write("#eval losses and metrics:\n")
                if modelname:
                    f.write("#history of model: " + modelname + '\n')
                    json.dump(self.hists[modelname], f)
                    self.first_loss_save[modelname] = False
                else:
                    f.write("#histories of all models:\n")
                    for k in self.models:
                        f.write("#history of model: " + k + '\n')
                        json.dump(self.hists[k], f)
                        print '\n'
                        self.first_loss_save['allsave'] = False
                f.write("\n")
        else:
            if modelname:
                with open(path, "r+") as f:
                    d = f.readlines()[::-1]
                    f.seek(0)
                    # rewrite_next_json = False
                    model_found = False

                    for j, i in enumerate(d):
                        if i == "#history of model: " + modelname + '\n':
                            model_found = True
                            if d[(j - 1)].find("{", 0, 1) != -1:
                                d[(j - 1)] = json.dumps(self.hists[modelname]) + '\n'
                            else:
                                print 'WARNING: loss history save file is not in the expected stats'
                                d = [json.dumps(
                                    self.hists[modelname]) + '\n', i] + d
                            break
                    if not model_found:
                        d = [json.dumps(self.hists[modelname]) + '\n',
                             "#history of model: " + modelname + '\n'] + d
                    d = d[::-1]
                    for i in d:
                        f.write(i)

                    # for i in d:
                    #     if i != "#history of model: " + modelname + '\n':
                    #         if rewrite_next_json:
                    #             if i.find("{", 0, 1) != -1:
                    #                 json.dump(self.hists[modelname], f)
                    #                 rewrite_next_json = False
                    #                 f.write('\n')
                    #             else:
                    #                 print 'WARNING: loss history save file is not in the expected stats'
                    #                 json.dump(self.hists[modelname], f)
                    #                 rewrite_next_json = False
                    #                 f.write(i)
                    #         else:
                    #             f.write(i)
                    #     else:
                    #         f.write(i)
                    #         model_found = True
                    #         rewrite_next_json = True
                    # if not model_found:
                    #     f.write("#history of model: " + modelname + '\n')
                    #     json.dump(self.hists[modelname], f)
                    # f.write('\n')
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
                try:
                    if d.next() == "#history of model: " + modelname + '\n':
                        break
                except StopIteration:
                    break
            if not loss_hist:
                raise Warning('No model %s was found in %s' %
                              (modelname, path))
        return loss_hist

    def load_loss(self, path='', modelname='', postfix=''):
        modelname += postfix
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
                 data_gen_creator=None, postfix='', extracallbacks=None):
        if verbose:
            timedeltas = []
        epochs_run = 0
        epoch_togo = nb_epochs

        # FIXME think about how to handle the missing samples%batch_size
        # samples
        steps_per_epoch = samples_per_epoch // self.BATCH_SIZE

        callbacks_ = [LearningRateScheduler(self._make_lrs_fct())]
        if extracallbacks:
            if not type(extracallbacks) == list:
                extracallbacks = [extracallbacks]
            callbacks_ += extracallbacks

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
                    callbacks=callbacks_,
                    data_gen_creator=data_gen_creator, postfix=postfix):
                try:
                    hist = self.models['model_norm' + postfix].fit_generator(
                        data_gen, validation_data=validation_data,
                        steps_per_epoch=steps_per_epoch,
                        epochs=nb_epoch,
                        initial_epoch=initial_epoch, verbose=verbose,
                        callbacks=callbacks)
                    self._save_hist(hist.history, postfix=postfix)
                except ValueError, e:
                    warnings.warn(
                        'Value Error in the main fit. Generator will be reinitialised. \n %s'
                        % e)
                    print 'saving'
                    self.save(postfix=postfix)
                    if data_gen_creator:
                        _main_fit(self, data_gen=data_gen_creator())
                    else:
                        warnings.warn(
                            'No reinitilizer of the data generator defined')
                        raise ValueError(e)

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

    def LSUV_init(self, train_batch, batch_size=None, modelname='model_norm',
                  postfix='',
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
                         main_layer='main_seq', prediction_batch_size=1,
                         postfix=''):
        modelname += postfix

        _layer = self.models[modelname].get_layer(main_layer).get_layer(
            layer)

        if not input_:
            input_ = [np.ones(shape=(prediction_batch_size,) + i[1:])
                      for i in self.models[modelname].get_layer(main_layer).input_shape]

        if self.layer_formats[layer] > 0 and self.layer_formats[layer] < 4:
            output_layer = fPermute((3, 0, 1, 2))(_layer.output)
            output_layer = Lambda(lambda x: T.reshape(x[0], (
                prediction_batch_size,) + tuple(T.shape(output_layer)[1:])),
                output_shape=lambda input_shape: (
                prediction_batch_size,) + input_shape[1:])(output_layer)
        else:
            try:
                output_layer = _layer.output
            except AttributeError, e:
                print 'debug infos after Attribute error'
                print layer
                print _layer
                raise AttributeError(e)

        if self.layer_formats[layer] == 4:
            def reshape_output(x, BATCH_SIZE=self.BATCH_SIZE):
                input_shape = T.shape(x)
                input_ = x
                new_input_shape = (
                    prediction_batch_size, input_shape[1], input_shape[2] * input_shape[0] / prediction_batch_size, input_shape[3])
                input_ = input_.reshape(new_input_shape)
                return input_

            output_lambda = Lambda(function=reshape_output, output_shape=lambda input_shape: (
                prediction_batch_size, input_shape[1], input_shape[2] * input_shape[0] / prediction_batch_size, input_shape[3]))
            output_layer = output_lambda(output_layer)

        intermediate_layer_model = Model(inputs=self.models[modelname]
                                         .get_layer(main_layer)
                                         .get_input_at(0),
                                         outputs=output_layer)

        return intermediate_layer_model.predict(
            input_, batch_size=prediction_batch_size)

    def get_layer_weights(self, layer,  modelname='model_norm',
                          main_layer='main_seq', postfix=''):
        modelname += postfix
        if type(layer) == int:
            ret_weights = self.models[modelname].get_layer(
                main_layer).layers[layer].get_weights()
        elif type(layer) == str:
            ret_weights = self.models[modelname].get_layer(main_layer)\
                                                .get_layer(layer).get_weights()
        else:
            raise ValueError('layer must be specified as int or string')
        return ret_weights
