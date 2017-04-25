from custom_keras_model_and_fit_capsels import kaggle_winsol

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
from custom_for_keras import kaggle_MultiRotMergeLayer_output,  kaggle_input, dense_weight_init_values

from keras.optimizers import SGD
from custom_for_keras import rmse

'''
This class contains the winning solution model of the kaggle galaxies contest transferred to keras and function to fit it.

'''


class kaggle_x_cat(kaggle_winsol):
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

        super(kaggle_x_cat, self).__init__(
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
                         ])

        except KeyError:
            pass
        self._init_hist_dics(self.models)

        return True

    def init_models(self, final_units=3):
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
                                                * 2
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
        model.add(Dense(units=final_units, activation='relu',
                        kernel_initializer=initializers.RandomNormal(
                            stddev=0.01),
                        bias_initializer=initializers.Constant(value=0.1),
                        name='dense_output'))

        model_seq = model([input_tensor, input_tensor_45])

        model_norm = Model(
            inputs=[input_tensor, input_tensor_45], outputs=model_seq,
            name='full_model_norm')
        model_norm_metrics = Model(
            inputs=[input_tensor, input_tensor_45], outputs=model_seq,
            name='full_model_metrics')

        self.models = {'model_norm': model_norm,
                       'model_norm_metrics': model_norm_metrics}

        self._compile_models(loss='categorical_crossentropy')

        return self.models
