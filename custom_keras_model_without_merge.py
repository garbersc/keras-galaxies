from custom_keras_model_and_fit_capsels import kaggle_winsol

import keras.backend as T
import keras.initializers as init
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.core import Lambda
# TODO will be removed from keras2, can this be achieved with a lambda
# layer now? looks like it:
# https://stackoverflow.com/questions/43160181/keras-merge-layer-warning
from keras.layers import Merge
from keras.engine.topology import InputLayer
from keras import initializers

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

from keras_extra_layers import MaxoutDense, Bias, DeBias, fPermute
from custom_for_keras import kaggle_MultiRotMergeLayer_output,  kaggle_input,\
    dense_weight_init_values

from keras.optimizers import SGD
from custom_for_keras import rmse


class kaggle_x_cat_x_maxout(kaggle_winsol):
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

        super(kaggle_x_cat_x_maxout, self).__init__(
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
                        postfix='', extra_metrics=[]):
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
                         'mean_squared_error',
                         'categorical_crossentropy'
                         ] + extra_metrics)

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
        if not self.use_keras_conv:
            from keras_extra_layers import kerasCudaConvnetPooling2DLayer,\
                kerasCudaConvnetConv2DLayer

        if not (type(freeze_conv) in (tuple, list)):
            freeze_conv = bool(freeze_conv)
            freeze_conv = (freeze_conv, freeze_conv, freeze_conv, freeze_conv)
        elif len(freeze_conv) != 4:
            raise ValueError(
                'Wrong number of freeze variables for the conv layers. Expected four  or bool, got %i' % len(freeze_conv))

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
                                                 + input_lay_1.output_shape[0])
                                                * num_views
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
        if not self.use_keras_conv:
            model.add(fPermute((1, 2, 3, 0), name='input_perm'))

        if not cut_out_conv[0]:
            if self.use_keras_conv:
                model.add(Conv2D(filters=conv_filters_n[0], kernel_size=6, name='conv_0',
                                 trainable=not freeze_conv[0], use_bias=False))

                model.add(Bias(
                    conv_filters_n[0], name='conv_0_bias'))
                self.layer_formats['conv_0_bias'] = 3
            else:
                model.add(kerasCudaConvnetConv2DLayer(
                    n_filters=conv_filters_n[0], filter_size=6, untie_biases=True, name='conv_0',
                    trainable=not freeze_conv[0]))
        else:
            del self.layer_formats['conv_0']

        if self.use_keras_conv:
            model.add(MaxPooling2D(name='pool_0'))
        else:
            model.add(kerasCudaConvnetPooling2DLayer(name='pool_0'))

        if not cut_out_conv[1]:
            if self.use_keras_conv:
                model.add(Conv2D(filters=conv_filters_n[1], kernel_size=5,
                                 name='conv_1',
                                 trainable=not freeze_conv[1], use_bias=False))
                model.add(Bias(conv_filters_n[1], name='conv_1_bias'))
                self.layer_formats['conv_1_bias'] = 3
            else:
                model.add(kerasCudaConvnetConv2DLayer(
                    n_filters=conv_filters_n[1], filter_size=5,
                    untie_biases=True, name='conv_1',
                    trainable=not freeze_conv[1]))
        else:
            del self.layer_formats['conv_1']

        if self.use_keras_conv:
            model.add(MaxPooling2D(name='pool_1'))
        else:
            model.add(kerasCudaConvnetPooling2DLayer(name='pool_1'))

        if not cut_out_conv[2]:
            if self.use_keras_conv:
                model.add(Conv2D(filters=conv_filters_n[2], kernel_size=3, name='conv_2',
                                 trainable=not freeze_conv[2], use_bias=False))
                model.add(Bias(conv_filters_n[2], name='conv_2_bias'))
                self.layer_formats['conv_2_bias'] = 3
            else:
                model.add(kerasCudaConvnetConv2DLayer(
                    n_filters=conv_filters_n[2], filter_size=3,
                    untie_biases=True, name='conv_2',
                    trainable=not freeze_conv[2]))
        else:
            del self.layer_formats['conv_2']

        if not cut_out_conv[3]:
            if self.use_keras_conv:
                model.add(Conv2D(filters=conv_filters_n[3], kernel_size=3,
                                 name='conv_3',
                                 trainable=not freeze_conv[3], use_bias=False))
                model.add(Bias(conv_filters_n[3], name='conv_3_bias'))
                self.layer_formats['conv_3_bias'] = 3
            else:
                model.add(kerasCudaConvnetConv2DLayer(n_filters=conv_filters_n[3],
                                                      filter_size=3,
                                                      weights_std=0.1,
                                                      untie_biases=True,
                                                      name='conv_3',
                                                      trainable=not freeze_conv[3]))
        else:
            del self.layer_formats['conv_3']

        if self.use_keras_conv:
            model.add(MaxPooling2D(name='pool_2'))
        else:
            model.add(kerasCudaConvnetPooling2DLayer(name='pool_2'))

        if not self.use_keras_conv:
            model.add(fPermute((3, 0, 1, 2), name='cuda_out_perm'))

        model.add(Lambda(function=kaggle_MultiRotMergeLayer_output,
                         output_shape=lambda x: (
                             x[0] // 4 // num_views, (x[1] * x[2]
                                                      * x[3] * 4
                                                      * num_views)),
                         arguments={'n_input_var': N_INPUT_VARIATION,
                                    'num_views': num_views},
                         name='conv_out_merge'))

        for i in range(n_maxout_layers) if n_maxout_layers > 0 else []:
            mo_name = 'maxout_%s' % str(i)
            self.layer_formats[mo_name] = 0
            if use_dropout:
                model.add(Dropout(0.5))
            model.add(MaxoutDense(output_dim=2048, nb_feature=2,
                                  weights=dense_weight_init_values(
                                      model.output_shape[-1], 2048,
                                      nb_feature=2),
                                  name=mo_name))

        if use_dropout:
            model.add(Dropout(0.5))
        model.add(Dense(units=final_units, activation=final_activation,
                        kernel_initializer=initializers.RandomNormal(
                            stddev=0.01),
                        bias_initializer=initializers.Constant(value=0.1),
                        name='dense_output'))

        model_seq = model([input_tensor, input_tensor_45])

        # schould not matter due to softmax activation in last layer
        output_layer_norm = Lambda(function=lambda x: (x - T.min(x)) / (T.max(x) - T.min(x)),
                                   output_shape=lambda x: x,
                                   )(model_seq)

        output_layer_noNorm = Lambda(function=lambda x: x,
                                     output_shape=lambda x: x,
                                     )(model_seq)

        deconv_perm_layer = fPermute((0, 1, 2, 3), name='deconv_out_perm')

        deconv_perm_tensor = deconv_perm_layer(
            model.get_layer('conv_1').get_output_at(0))

        debias_layer_1 = DeBias(nFilters=32, name='debias_layer_1')(
            deconv_perm_tensor)

        deconv_layer_1 = Conv2DTranspose(filters=3, kernel_size=6,
                                         strides=(1, 1),
                                         use_bias=False,
                                         name='deconv_layer_1'
                                         )(debias_layer_1)

        debias_layer_0 = DeBias(nFilters=32, name='debias_layer_0')(
            deconv_layer_1)

        deconv_layer_0 = Conv2DTranspose(filters=3, kernel_size=6,
                                         strides=(1, 1),
                                         use_bias=False,
                                         name='deconv_layer_0'
                                         )(debias_layer_0)

        # def reshape_output(x, BATCH_SIZE=self.BATCH_SIZE):
        #     input_shape = T.shape(x)
        #     input_ = x
        #     new_input_shape = (
        #         BATCH_SIZE, input_shape[1], input_shape[2] * input_shape[0] / BATCH_SIZE, input_shape[3])
        #     input_ = input_.reshape(new_input_shape)
        #     return input_

        def reshape_output(x, BATCH_SIZE=self.BATCH_SIZE):
            input_shape = T.shape(x)
            input_ = x
            new_input_shape = (
                BATCH_SIZE, input_shape[1], input_shape[2] * input_shape[0] / BATCH_SIZE, input_shape[3])
            input_ = input_.reshape(new_input_shape)
            return input_

        # output_layer_deconv = Lambda(function=reshape_output, output_shape=lambda input_shape: (
        #     self.BATCH_SIZE, input_shape[1], input_shape[2] * input_shape[0] /
        #     self.BATCH_SIZE, input_shape[3]))(deconv_layer)

        output_layer_deconv = Lambda(
            function=reshape_output, output_shape=lambda x: x,)(deconv_layer_0)

        model_norm = Model(
            inputs=[input_tensor, input_tensor_45], outputs=output_layer_norm, name='full_model_norm')
        model_norm_metrics = Model(
            inputs=[input_tensor, input_tensor_45], outputs=output_layer_norm, name='full_model_metrics')
        model_noNorm = Model(
            inputs=[input_tensor, input_tensor_45], outputs=output_layer_noNorm, name='full_model_noNorm')
        model_deconv = Model(inputs=model.get_input_at(0), outputs=output_layer_deconv,
                             name='model_deconv')

        self.models = {'model_norm': model_norm,
                       'model_norm_metrics': model_norm_metrics,
                       'model_noNorm': model_noNorm,
                       'model_deconv': model_deconv,
                       }

        self._compile_models(
            loss=loss, extra_metrics=extra_metrics, optimizer=optimizer)

        return self.models
