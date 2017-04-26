from custom_keras_model_base import kaggle_base

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.core import Lambda
from keras import initializers

from keras_extra_layers import MaxoutDense
from custom_for_keras import OptimisedDivGalaxyOutput, dense_weight_init_values


class kaggle_ellipse_fit(kaggle_base):
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

    def __init__(self,
                 BATCH_SIZE,
                 LEARNING_RATE_SCHEDULE=None,
                 MOMENTUM=None,
                 LOSS_PATH='./',
                 WEIGHTS_PATH='./',
                 **kwargs):

        self.layer_formats = {'maxout_0': 0, 'maxout_1': 0,
                              'dense_output': 0}

        super(kaggle_ellipse_fit, self).__init__(
            BATCH_SIZE,
            LEARNING_RATE_SCHEDULE=LEARNING_RATE_SCHEDULE,
            MOMENTUM=MOMENTUM,
            LOSS_PATH=LOSS_PATH,
            WEIGHTS_PATH=WEIGHTS_PATH,
            **kwargs)

    '''
    compliles all available models
    initilises loss histories
    '''

    def _compile_models(self, postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self)._compile_models(postfix=postfix)

    '''
    initiates models according to the kaggle galaxies winning solution

    Returns:
    dictinary with the model without normalisation, with normalisation and with normalisation and extra metrics for validation
    '''

    def init_models(self, input_shape=3):
        print "init model"
        input_tensor = Input(batch_shape=(self.BATCH_SIZE,
                                          input_shape),
                             dtype='float32', name='input_tensor')

        model = Sequential(name='main_seq')

        model.add(Dropout(0.5, input_shape=(input_shape,)))
        model.add(MaxoutDense(output_dim=2048, nb_feature=2,
                              weights=dense_weight_init_values(
                                  model.output_shape[-1], 2048, nb_feature=2),
                              name='maxout_0'))

        model.add(Dropout(0.5))
        model.add(MaxoutDense(output_dim=2048, nb_feature=2,
                              weights=dense_weight_init_values(
                                  model.output_shape[-1], 2048, nb_feature=2),
                              name='maxout_1'))

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
                                              'categorised': CATEGORISED})(
                                                  model_seq)
        output_layer_noNorm = Lambda(function=OptimisedDivGalaxyOutput,
                                     output_shape=lambda x: x,
                                     arguments={'normalised': False,
                                                'categorised': CATEGORISED})(
                                                    model_seq)

        model_norm = Model(
            inputs=[input_tensor], outputs=output_layer_norm,
            name='full_model_norm_ellipse')
        model_norm_metrics = Model(
            inputs=[input_tensor], outputs=output_layer_norm,
            name='full_model_metrics_ellipse')
        model_noNorm = Model(
            inputs=[input_tensor], outputs=output_layer_noNorm,
            name='full_model_noNorm_ellipse')

        self.models = {'model_norm_ellipse': model_norm,
                       'model_norm_metrics_ellipse': model_norm_metrics,
                       'model_noNorm_ellipse': model_noNorm}

        self._compile_models(postfix='_ellipse')

        return self.models

    '''
    Arguments:
    modelname: name of the model to be printed
    '''

    def print_summary(self, modelname='model_norm', postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).print_summary(modelname=modelname,
                                                             postfix=postfix)
    '''
    loads previously saved weights

    Arguments:
    path: path to savefile
    modelname: name of the model for which the weights are loaded, in default the models use all the same weight
    '''

    def load_weights(self, path, modelname='model_norm', postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).load_weights(
            path=path, modelname=modelname, postfix=postfix)

    '''
    prints the loss and metric information of a model

    '''

    def print_last_hist(self, modelname='model_norm_metrics', postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).print_last_hist(
            modelname=modelname, postfix=postfix)

    '''
    evaluates a model according to true answeres, saves the information in the history
    Arguments:
    x: input sample
    y_valid: true answeres
    batch_size: inputs evaluated at the same time, default uses batch size from class initialisation
    verbose: interger, set to 0 to minimize oputput
    '''

    def evaluate(self, x, y_valid, batch_size=None,
                 modelname='model_norm_metrics', verbose=1, postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).evaluate(x, y_valid,
                                                        batch_size=batch_size,
                                                        modelname=modelname,
                                                        verbose=verbose,
                                                        postfix=postfix)

    def predict(self, x, batch_size=None,
                modelname='model_norm_metrics', verbose=1, postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).predict(x, batch_size,
                                                       modelname=modelname,
                                                       verbose=verbose,
                                                       postfix=postfix)

    def _save_hist(self, history, modelname='model_norm', postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self)._save_hist(history=history,
                                                          modelname=modelname,
                                                          postfix=postfix)

    '''
    saves the modelweights as hdf5 file
    Arguments:
    path: the path were the weights are to be saved, if default the WEIGHTS_PATH with which the class was initialised is used
    modelname: name of the model, default allmodels have the same weights
    '''

    def save_weights(self, path='', modelname='model_norm',
                     postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).save_weights(
            path=path,
            modelname=modelname,
            postfix=postfix)

    '''
    saves the loss and validation metric histories as json strings in a text file
    Arguments:
    path: default uses LOSS_PATH from initialisation
    modelname: default saves history of all models
    '''

    def save_loss(self, path='', modelname='', postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).save_loss(
            path=path,
            modelname=modelname,
            postfix=postfix)

    def load_loss(self, path='', modelname='', postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).load_loss(path=path,
                                                         modelname=modelname,
                                                         postfix=postfix)
    '''
    performs all saving task
    '''

    def save(self, option_string=None, postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).save(
            option_string=option_string,
            postfix=postfix)

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
                 data_gen_creator=None, postfix='_ellipse'):
        super(kaggle_ellipse_fit, self).full_fit(
            data_gen,
            validation,
            samples_per_epoch,
            validate_every,
            nb_epochs,
            verbose=verbose,
            save_at_every_validation=save_at_every_validation,
            data_gen_creator=data_gen_creator,
            postfix=postfix)

    def LSUV_init(self, train_batch, batch_size=None, modelname='model_norm',
                  postfix='_ellipse',
                  sub_modelname='main_seq'):
        super(kaggle_ellipse_fit, self).LSUV_init(train_batch,
                                                  batch_size=batch_size,
                                                  modelname=modelname,
                                                  postfix=postfix,
                                                  sub_modelname=sub_modelname)

    def get_layer_output(self, layer, input_=None, modelname='model_norm',
                         main_layer='main_seq', prediction_batch_size=1,
                         postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).get_layer_output(
            layer=layer,
            input_=input_,
            modelname=modelname,
            main_layer=main_layer,
            prediction_batch_size=prediction_batch_size,
            postfix=postfix)

    def get_layer_weights(self, layer,  modelname='model_norm',
                          main_layer='main_seq', postfix='_ellipse'):
        return super(kaggle_ellipse_fit, self).get_layer_weights(
            layer=layer,
            modelname=modelname,
            main_layer=main_layer,
            postfix=postfix)
