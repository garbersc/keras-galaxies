from custom_keras_model_and_fit_capsels import kaggle_winsol

import keras.backend as T
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2DTranspose
from keras.layers.core import Lambda

from keras_extra_layers import kerasCudaConvnetPooling2DLayer, fPermute,\
    kerasCudaConvnetConv2DLayer, DeBias

from keras.optimizers import SGD


class simple_deconv(kaggle_winsol):
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

    def __init__(self, *args,
                 **kwargs):

        super(simple_deconv, self).__init__(
            *args,
            **kwargs)

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
            self.models['model_deconv' + postfix].compile(
                loss=loss,
                optimizer=optimizer)
        except KeyError:
            pass
        try:
            self.models['model_simple' + postfix].compile(
                loss=loss,
                optimizer=optimizer)
        except KeyError:
            pass

        self._init_hist_dics(self.models)

        return True

    def init_models(self, final_units=3):
        print "init model"
        sinput_tensor = Input(batch_shape=(self.BATCH_SIZE,
                                           self.NUM_INPUT_FEATURES,
                                           45,
                                           45),
                              dtype='float32', name='input_tensor')

        '''
        simple conv model with input layer and conv layer
        '''
        smodel = Sequential(name='simple_mod')

        smodel.add(fPermute(
            (1, 2, 3, 0), batch_input_shape=(self.BATCH_SIZE, self.NUM_INPUT_FEATURES, 45, 45),  name='sinput_perm'))

        smodel.add(kerasCudaConvnetConv2DLayer(
            n_filters=32, filter_size=6, untie_biases=True, name='sconv_0'))

        smodel_seq = smodel(sinput_tensor)

        output_layer_smodel = Lambda(function=lambda x: x,
                                     output_shape=lambda x: x,
                                     )(smodel_seq)

        '''
        deconvolution model
        '''

        deconv_perm_layer = fPermute((3, 0, 1, 2), name='deconv_out_perm')

        deconv_perm_tensor = deconv_perm_layer(
            smodel.get_layer('sconv_0').get_output_at(0))

        debias_layer = DeBias(nFilters=32, name='debias_layer')(
            deconv_perm_tensor)

        deconv_layer = Conv2DTranspose(filters=3, kernel_size=6,
                                       strides=(1, 1),
                                       use_bias=False,
                                       name='deconv_layer',
                                       )(debias_layer)

        def reshape_output(x, BATCH_SIZE=self.BATCH_SIZE):
            input_shape = T.shape(x)
            input_ = x
            new_input_shape = (
                BATCH_SIZE, input_shape[1], input_shape[2] * input_shape[0] / BATCH_SIZE, input_shape[3])
            input_ = input_.reshape(new_input_shape)
            return input_

        output_layer_deconv = Lambda(function=reshape_output, output_shape=lambda input_shape: (
            self.BATCH_SIZE, input_shape[1], input_shape[2] * input_shape[0] / self.BATCH_SIZE, input_shape[3]))(deconv_layer)

        model_deconv = Model(inputs=smodel.get_input_at(0), outputs=output_layer_deconv,
                             name='deconv_1')
        model_simple = Model(sinput_tensor,
                             outputs=output_layer_smodel, name='simple_model')

        self.models = {'model_deconv': model_deconv,
                       'model_simple': model_simple
                       }

        self._compile_models(loss='categorical_crossentropy')

        return self.models
