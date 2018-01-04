from custom_keras_model_and_fit_capsels import kaggle_winsol

import keras.backend as T
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2DTranspose, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda

from keras_extra_layers import fPermute, Bias, DeBias, DePool

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

        # TODO: extraction of the weights and output of the keras conv layer is
        # not checked yet
        use_keras_conv = True
        if 'use_keras_conv' in kwargs:
            use_keras_conv = kwargs.pop('use_keras_conv')

        super(simple_deconv, self).__init__(
            *args,
            **kwargs)

        self.use_keras_conv = use_keras_conv
        self.layer_formats = {'conv_0': 1, 'conv_1': 1, 'conv_2': 1,
                              'conv_3': 1, 'pool_0': 2, 'pool_1': 2,
                              'pool_2': 2, 'conv_out_merge': -1,
                              'dense_output': 0}
        super(simple_deconv, self).__init__(
            *args,
            **kwargs)

        self.use_keras_conv = use_keras_conv
        self.layer_formats = {'conv_0': 1, 'conv_1': 1, 'conv_2': 1,
                              'conv_3': 1, 'pool_0': 2, 'pool_1': 2,
                              'pool_2': 2, 'conv_out_merge': -1,
                              'dense_output': 0}
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
            self.models['model_simple' + postfix].compile(
                loss=loss,
                optimizer=optimizer)
        except KeyError:
            pass
        try:
            self.models['model_deconv' + postfix].compile(
                loss=loss,
                optimizer=optimizer)
        except KeyError:
            pass

        self._init_hist_dics(self.models)

        return True

    def init_models(self, final_units=3, n_maxout_layers=0,
                    loss='categorical_crossentropy',
                    optimizer=None,
                    extra_metrics=[],
                    freeze_conv=False,
                    cut_out_conv=(False, False, False, False),
                    conv_filters_n=(32, 64, 128, 128),
                    use_dropout=True,
                    final_activation='softmax'):

        if not (type(freeze_conv) in (tuple, list)):
            freeze_conv = bool(freeze_conv)
            freeze_conv = (freeze_conv, freeze_conv, freeze_conv, freeze_conv)
        elif len(freeze_conv) != 4:
            raise ValueError(
                'Wrong number of freeze variables for the conv layers. Expected four  or bool, got %i' % len(freeze_conv))

        print "init model"
        sinput_tensor = Input(batch_shape=(self.BATCH_SIZE,
                                           self.NUM_INPUT_FEATURES,
                                           self.input_sizes[1][0],
                                           self.input_sizes[1][1]
                                           ),
                              dtype='float32', name='input_tensor')

        '''
        emulation of the main NN without the merge layer
        parameters and weights are set the same
        '''
        model = Sequential(name='main_seq')

        model.add(Conv2D(filters=conv_filters_n[0], kernel_size=6, name='conv_0',
                         trainable=not freeze_conv[0], use_bias=False, batch_input_shape=(16, self.NUM_INPUT_FEATURES, 45, 45)))

        model.add(Bias(
            conv_filters_n[0], name='conv_0_bias'))

        model.add(MaxPooling2D(name='pool_0'))

        model.add(Conv2D(filters=conv_filters_n[1], kernel_size=5,
                         name='conv_1',
                         trainable=not freeze_conv[1], use_bias=False))
        model.add(Bias(conv_filters_n[1], name='conv_1_bias'))

        '''
        deconvolution model
        '''
        deconv_perm_layer = fPermute((0, 1, 2, 3), name='deconv_out_perm')

        deconv_perm_tensor = deconv_perm_layer(
            model.get_layer('conv_1_bias').get_output_at(0))

        debias_layer_1 = DeBias(nFilters=conv_filters_n[0], name='debias_layer_1')(
            deconv_perm_tensor)

        deconv_layer_1 = Conv2DTranspose(filters=conv_filters_n[0], kernel_size=5,
                                         strides=(1, 1),
                                         use_bias=False,
                                         name='deconv_layer_1'
                                         )(debias_layer_1)

        depool_0 = DePool(model=model, pool_layer_origin=['pool_0'], name='depool_layer_0')(
            deconv_layer_1)

        debias_layer_0 = DeBias(nFilters=32, name='debias_layer_0')(
            depool_0)

        deconv_layer_0 = Conv2DTranspose(filters=3, kernel_size=6,
                                         strides=(1, 1),
                                         use_bias=False,
                                         name='deconv_layer_0'
                                         )(debias_layer_0)

        smodel_seq = model(sinput_tensor)

        output_layer_smodel = Lambda(function=lambda x: x,
                                     output_shape=lambda x: x,
                                     )(smodel_seq)

        def reshape_output(x, BATCH_SIZE=self.BATCH_SIZE):
            input_shape = T.shape(x)
            input_ = x
            # new_input_shape = (
            # BATCH_SIZE, input_shape[1], input_shape[2] * input_shape[0] /
            # BATCH_SIZE, input_shape[3])
            new_input_shape = (
                input_shape[0], input_shape[1], input_shape[2], input_shape[3])
            input_ = input_.reshape(new_input_shape)
            return input_

        output_layer_deconv = Lambda(function=reshape_output, output_shape=lambda input_shape: (
            self.BATCH_SIZE, input_shape[1], input_shape[2] * input_shape[0] /
            self.BATCH_SIZE, input_shape[3]))(deconv_layer_0)

        model_simple = Model(sinput_tensor,
                             outputs=output_layer_smodel, name='model_simple')

        model_deconv = Model(inputs=model.get_input_at(0), outputs=output_layer_deconv,
                             name='model_deconv')

        self.models = {'model_norm': model_simple,
                       'model_deconv': model_deconv,
                       }

        self._compile_models(loss='categorical_crossentropy')

        return self.models
