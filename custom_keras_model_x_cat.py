from custom_keras_model_x_cat_x_maxout import kaggle_x_cat_x_maxout

import h5py
import numpy as np


class kaggle_x_cat(kaggle_x_cat_x_maxout):
    def __init__(self, *args,
                 **kwargs):

        super(kaggle_x_cat, self).__init__(
            *args,
            **kwargs)

    def init_models(self, final_units=3, loss='categorical_crossentropy',
                    **kwargs):
        if 'conv_filters_n' in kwargs.keys():
            self.conv_filters_n = kwargs['conv_filters_n']
        return super(kaggle_x_cat, self).init_models(final_units=final_units,
                                                     loss=loss,
                                                     n_maxout_layers=2,
                                                     **kwargs)

    def load_one_layers_weight(self, path, layername_source, layername_this='',
                               modelname='model_norm',
                               sub_modelname='main_seq',
                               postfix='', used_conv_layers=None):
        if (not used_conv_layers or layername_source not in used_conv_layers)\
           and layername_source.find('maxout_0') < 0:
            return super(kaggle_x_cat, self).load_one_layers_weight(
                path,
                layername_source,
                layername_this='',
                modelname='model_norm',
                sub_modelname='main_seq',
                postfix='')
        elif not used_conv_layers:
            used_conv_layers = {'conv_3': range(self.conv_filters_n[3])}

        # debug!!!
        print type(used_conv_layers)
        print type(self.conv_filters_n)
        print used_conv_layers
        print self.conv_filters_n

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

            try:
                if ls.find('conv') >= 0:
                    weight = [np.array([np.transpose(weight[ls + '_W'],
                                                     (3, 1, 2, 0))[i]
                                        for i in used_conv_layers[ls]])
                              .transpose(3, 1, 2, 0),
                              np.array([weight[ls + '_b'][i]
                                        for i in used_conv_layers[ls]])]
                    if ls.find('0') < 0 and ls.find('conv') >= 0:
                        for x in ls:
                            if x.isdigit():
                                conv_id = int(x)
                                break
                        weight = [np.array([weight[0][i]
                                            for i in
                                            used_conv_layers['conv_'
                                                             + str(conv_id - 1)]]),
                                  weight[1]]
                elif ls.find('maxout_0') >= 0:
                    w_init_shape = np.shape(weight.values()[0])
                    weight_stripe = w_init_shape[1] / 128
                    weight_kernel = np.reshape(
                        weight.values()[0],
                        (w_init_shape[0], 128,
                         weight_stripe, w_init_shape[-1]))\
                        .transpose(1, 0, 2, 3)
                    weight_kernel = np.array([weight_kernel[i] for i in
                                              used_conv_layers['conv_3']])\
                                      .transpose(1, 0, 2, 3).reshape((
                                          w_init_shape[0],
                                          len(used_conv_layers['conv_3']) *
                                          weight_stripe,
                                          w_init_shape[-1]))
                    weight = [weight_kernel, weight.values()[1]]
                else:
                    raise TypeError(
                        'Layer ' + ls + ' is here not awaited!')

                print ls
                print np.shape(weight[0])
            except:
                if type(weight) == 'dict':
                    for n in weight:
                        print n
                        print np.shape(weight[n])
                else:
                    for w in weight:
                        print np.shape(w)
                print used_conv_layers
                # print layername_source
                # print used_conv_layers[layername_source[0]]
                # for i in used_conv_layers[layername_source[0]]:
                #     print i
                raise

            try:
                self.models[modelname].get_layer(
                    sub_modelname).get_layer(lt).set_weights(weight)
            except ValueError, e:
                print 'source ' + ls
                for w in weight:
                    print np.shape(w)
                print 'target ' + lt
                for w in self.models[modelname].get_layer(
                        sub_modelname).get_layer(lt).get_weights():
                    print np.shape(w)
                print
                print self.models[modelname].get_layer(
                    sub_modelname).get_layer(lt).get_config()
                print
                raise ValueError(e)

        file_.close()
        if self.debug:
            print
        with open(self.LOSS_PATH, 'a')as f:
            f.write('#loaded weights of layer(s) ' + str(layername_source) + '  from '
                    + str(path) + ' into  model ' +
                    str(modelname) + ' into the layer(s) '
                    + str(layername_this) + '\n')
        return True

    def load_weights(self, path, modelname='model_norm', postfix='', **kwargs):
        if 'used_conv_layers' in kwargs:
            self.used_conv_layers = kwargs.pop('used_conv_layers')
            for sub_model in self.models[modelname + postfix].layers:
                if 'layers' in vars(sub_model):
                    for layer in sub_model.layers:
                        if 'layer_formats' in vars(self)\
                            and layer.name in self.layer_formats\
                            and (self.layer_formats[layer.name] == 0
                                 or self.layer_formats[layer.name] == 1):
                            self.load_one_layers_weight(
                                path=path,
                                layername_source=layer.name,
                                modelname=modelname,
                                sub_modelname=sub_model.name,
                                postfix=postfix,
                                used_conv_layers=self.used_conv_layers)
            return_ = True
        else:
            return_ = super(kaggle_x_cat, self).load_weights(path=path,
                                                             modelname=modelname,
                                                             postfix=postfix,
                                                             **kwargs)
        return return_
