from custom_keras_model_base import kaggle_base

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.core import Lambda
# TODO will be removed from keras2, can this be achieved with a lambda
# layer now? looks like it:
# https://stackoverflow.com/questions/43160181/keras-merge-layer-warning
from keras.layers import Merge
from keras.engine.topology import InputLayer
from keras import initializers

from keras_extra_layers import kerasCudaConvnetPooling2DLayer, fPermute, kerasCudaConvnetConv2DLayer, MaxoutDense
from custom_for_keras import kaggle_MultiRotMergeLayer_output, OptimisedDivGalaxyOutput, kaggle_input, dense_weight_init_values


'''
This class contains the winning solution model of the kaggle galaxies contest transferred to keras and function to fit it.

'''


class kaggle_winsol(kaggle_base):
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
                 WEIGHTS_PATH='./',
                 **kwargs):

        self.NUM_INPUT_FEATURES = NUM_INPUT_FEATURES
        self.input_sizes = input_sizes
        self.PART_SIZE = PART_SIZE
        self.include_flip = include_flip

        self.layer_formats = {'conv_0': 1, 'conv_1': 1, 'conv_2': 1,
                              'conv_3': 1, 'pool_0': 2, 'pool_1': 2,
                              'pool_2': 2, 'conv_out_merge': -1,
                              'maxout_0': 0, 'maxout_1': 0,
                              'dense_output': 0}

        super(kaggle_winsol, self).__init__(
            BATCH_SIZE=BATCH_SIZE,
            LEARNING_RATE_SCHEDULE=LEARNING_RATE_SCHEDULE,
            MOMENTUM=MOMENTUM,
            LOSS_PATH=LOSS_PATH,
            WEIGHTS_PATH=WEIGHTS_PATH,
            **kwargs)

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
