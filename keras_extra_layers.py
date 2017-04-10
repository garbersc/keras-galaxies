'''
Author: Christoph Garbers
keras layers that are needed if no CuDNN speed up is available
'''
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import functools
import copy

from keras import initializers
from keras import activations

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs

# no mask implementation

'''
build(input_shape): this is where you will define your weights. This method must set self.built = True, which can be done by calling super([Layer], self).build().

call(x): this is where the layer's logic lives. Unless you want your layer to support masking, you only have to care about the first argument passed to call: the input tensor.

get_output_shape_for(input_shape): in case your layer modifies the shape of its input, you should specify here the shape transformation logic. This allows Keras to do automatic shape inference.

'''


class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=(input_shape[1], self.output_dim),
                                 initializer='random_uniform',
                                 trainable=True)
        super(MyLayer, self).build()  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


# not neede anymore, available in keras 2
def constant(shape, scale=1., name=None):
    constant = scale
    for i in shape[::-1]:
        try:
            constant = [constant] * i
        except:
            print("exception in constant init! i is ",
                  i, " the shape is ", shape)
            exit()
    return K.variable(constant)


class fPermute(Layer):
    def __init__(self, dims, **kwargs):
        self.dims = tuple(dims)
        super(fPermute, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        input_shape = list(input_shape)
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            target_dim = input_shape[dim]
            output_shape[i] = target_dim
        return tuple(output_shape)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        return K.permute_dimensions(x, self.dims)

    def get_config(self):
        config = {'dims': self.dims}
        base_config = super(fPermute, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class kerasCudaConvnetPooling2DLayer(Layer):
    def __init__(self, pool_size=2, stride=None, **kwargs):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        # self.input_layer = input_layer
        self.params = []
        self.bias_params = []
        # self.mb_size = self.input_layer.mb_size

        self.pool_op = MaxPool(ds=self.pool_size, stride=self.stride)

        super(kerasCudaConvnetPooling2DLayer, self).__init__(**kwargs)

    # def build(self, input_shape):
    # super(kerasCudaConvnetPooling2DLayer, self).build(input_shape)  # Be
    # sure to call this somewhere!

    def call(self, x, mask=None):
        contiguous_input = gpu_contiguous(x)
        return self.pool_op(contiguous_input)

    def get_output_shape_for(self, input_shape):
        l, w, h, m_b = input_shape

        new_w = int(
            np.ceil(float(w - self.pool_size + self.stride) / self.stride))
        new_h = int(
            np.ceil(float(h - self.pool_size + self.stride) / self.stride))

        return (l, new_w, new_h, m_b)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


class kerasCudaConvnetConv2DLayer(Layer):
    def __init__(self, n_filters, filter_size, weights_std=0.01,
                 init_bias_value=0.1, stride=1, activation='relu',
                 partial_sum=None, pad=0, untie_biases=False,
                 # check the keyword arguments if nopt on default values
                 initW='truncated_normal', initB='constant',
                 initial_weights=None, W_regularizer=None, W_constraint=None,
                 b_regularizer=None, b_constraint=None, **kwargs):
        """
        Only the valid border mode is supported.

        n_filters should be a multiple of 16
        """

        self.initW = initializers.get(
            {'class_name': initW, 'config': {'stddev': weights_std}})
        self.initB = initializers.get({'class_name': initB,
                                       'config': {'value': init_bias_value}})
        self.initial_weights = initial_weights
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.weights_std = np.float32(weights_std)
        self.init_bias_value = np.float32(init_bias_value)
        self.stride = stride
        self.nonlinearity = activations.get(activation)
        self.partial_sum = partial_sum
        self.pad = pad
        self.untie_biases = untie_biases
        self.W_regularizer = W_regularizer
        self.W_constraint = W_constraint
        self.b_regularizer = b_regularizer
        self.b_constraint = b_constraint

        self.filter_acts_op = FilterActs(
            stride=self.stride, partial_sum=self.partial_sum, pad=self.pad)
        super(kerasCudaConvnetConv2DLayer, self).__init__(**kwargs)
    '''
    def reset_params(self):
        self.W.set_value(np.random.randn(
            *self.filter_shape).astype(np.float32) * self.weights_std)

        if self.untie_biases:
            self.b.set_value(np.ones(self.get_output_shape()[:3]).astype(
                np.float32) * self.init_bias_value)
        else:
            self.b.set_value(np.ones(self.n_filters).astype(
                np.float32) * self.init_bias_value)
    '''

    def build(self, input_shape):
        if K.image_data_format() != 'channels_first':
            print "maybe wrong dim ordering in custom conv layer, ordering is %s, data format is" % (K.image_dim_ordering(), K.image_data_format())

        self.filter_shape = (
            input_shape[0], self.filter_size, self.filter_size, self.n_filters)

        self.W = self.add_weight(self.filter_shape,
                                 initializer=self.initW,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 trainable=True)
        if self.untie_biases:
            self.b = self.add_weight((self.get_output_shape_for(input_shape)[:3]),
                                     initializer=self.initB,
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint,
                                     trainable=True)
        else:
            self.b = self.add_weight((self.n_filters,),
                                     initializer=self.initB,
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint,
                                     trainable=True)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

        super(kerasCudaConvnetConv2DLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        input_ = x

        contiguous_input = gpu_contiguous(input_)
        contiguous_filters = gpu_contiguous(self.W)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)

        if self.untie_biases:
            conved += self.b.dimshuffle(0, 1, 2, 'x')
        else:
            conved += self.b.dimshuffle(0, 'x', 'x', 'x')

        return self.nonlinearity(conved)

    def get_output_shape_for(self, input_shape):
        l, w, h, m_b = input_shape
        output_width = (w + 2 * self.pad - self.filter_size +
                        self.stride) // self.stride
        output_height = (h + 2 * self.pad - self.filter_size +
                         self.stride) // self.stride
        output_shape = (self.n_filters, output_width, output_height, m_b)
        return output_shape

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)
