'''
Author: Christoph Garbers
keras layers that are needed if no CuDNN speed up is available
and layers that fell out of favor in keras2
'''
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
import numpy as np
import copy
import warnings

from keras.layers import Conv2DTranspose
from keras import initializers
from keras import activations
from keras import regularizers
from keras import constraints

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


class DeConv(Conv2DTranspose):
    # Adds a scaling of the edges to the Conv2DTranspose layer to avoid
    # artifacts in the stride=(1,1) case
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        # if type(kernel_size) == 'int':
        #     self.kernel_size = (kernel_size, kernel_size)
        # else:
        #     self.kernel_size = kernel_size
        if strides != (1, 1) or padding != 'valid':
            warnings.warn(
                'Layer DeConv was not build for this stride and/or padding option!')
        super(DeConv, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(DeConv, self).build(input_shape)

        shape = super(DeConv, self).compute_output_shape(input_shape)

        a = np.zeros(shape)
        for i in range(shape[2]):
            for j in range(shape[3]):
                a[:, :, i, j] = float(np.prod(self.kernel_size))\
                    / min(float(i + 1), self.kernel_size[0])\
                    / min(float(j + 1), self.kernel_size[1])

        self.edge_scale = K.variable(value=a)

    def call(self, inputs):
        outputs = super(DeConv, self).call(inputs)

        outputs = outputs * self.edge_scale

        return outputs

# Untied Bias Layer. Can be used instead of Activation.


class Bias(Layer):
    def __init__(self, nFilters, **kwargs):
        self.nFilters = nFilters
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.b = self.add_weight(shape=(input_shape[1:]),
                                 initializer='constant',
                                 trainable=True,
                                 name='{}_b'.format(self.name))
        self.built = True
        # Be sure to call this somewhere!
        super(Bias, self).build(input_shape)

    def call(self, x, mask=None):
        output = x
        output += self.b.dimshuffle('x', 0, 1, 2)
        return output

    def get_output_shape_for(self, input_shape):
        return input_shape

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


class DeBias(Bias):
    def __init__(self, nFilters, **kwargs):
        super(DeBias, self).__init__(nFilters, **kwargs)

    def call(self, x, mask=None):
        output = x
        output -= self.b.dimshuffle('x', 0, 1, 2)
        return output


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


class DePool(Layer):
    def __init__(self,  model,
                 pool_layer_origin=['pool_0'], stride=(2, 2),
                 **kwargs):

        self.stride = stride
        self.model = model
        self.pool_layer_origin = pool_layer_origin
        super(DePool, self).__init__(**kwargs)

    def _get_pool_flags(self, pool):
        # permutation needed if the layer is in the 'normal' not the pylearn
        # order, maybe make a switch for that and the channel order
        input_ = K.permute_dimensions(pool.get_input_at(0), (3, 0, 1, 2))
        pooled = K.permute_dimensions(pool.get_output_at(0), (3, 0, 1, 2))

        pooled = K.repeat_elements(pooled, self.stride[0], axis=-2)
        pooled = K.repeat_elements(pooled, self.stride[1], axis=-1)

        print 'shapes before k.equal %s \t %s' % (K.int_shape(input_),
                                                  K.int_shape(
            pooled))

        return K.equal(input_, pooled)

    def call(self, x):
        pool = self.model
        for name in self.pool_layer_origin:
            pool = pool.get_layer(name)
        flags = self._get_pool_flags(pool)

        x_up = K.repeat_elements(x, self.stride[0], axis=-2)
        x_up = K.repeat_elements(x_up, self.stride[1], axis=-1)

        print 'shapes before * %s ' % str(K.int_shape(x_up))

        x_up = x_up * K.cast(flags, 'float32')

        return x_up

    def get_output_shape_for(self, input_shape):
        m_b,  l, w, h = input_shape

        return (m_b, l, self.stride[0] * w, self.stride[1] * h)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


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


class kerasCudaConvnetDeconv2DLayer(Layer):
    def __init__(self, n_filters, filter_size, weights_std=0.01,
                 init_bias_value=0.1, stride=1, activation='relu',
                 partial_sum=None, pad=0, untie_biases=False,
                 initW='truncated_normal', initB='constant',
                 initial_weights=None, W_regularizer=None, W_constraint=None,
                 b_regularizer=None, b_constraint=None, **kwargs):
        """
        Transposed convolution for the pylearn implementation

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

        if stride != 1 or pad != 0:
            warnings.warn(
                'Layer DeConv was not build for this stride and/or padding option!')

        self.filter_acts_op = FilterActs(
            stride=self.stride, partial_sum=self.partial_sum, pad=self.pad)
        super(kerasCudaConvnetDeconv2DLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_shape_ = input_shape
        self.filter_shape = (
            self.n_filters, self.filter_size, self.filter_size, input_shape[0])

        self.W = self.add_weight(self.filter_shape,
                                 initializer=self.initW,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 trainable=True)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

        super(kerasCudaConvnetDeconv2DLayer, self).build(
            input_shape)

    def call(self, x, mask=None):
        input_ = x
        output_ = K.zeros(self.compute_output_shape(self.input_shape_))
        for w in range(self.input_shape_[1]):
            for h in range(self.input_shape_[2]):
                # w_extended = np.zeros(
                #     self.compute_output_shape(self.input_shape_))
                # w_extended[:, w: w + self.filter_size,
                #             h: h + self.filter_size, :] = w_extended[
                #                 :, w: w + self.filter_size,
                #                 h: h + self.filter_size, :] + K.eval(self.W)
                # w_extended = K.variable(w_extended)
                # output_step = K.dot(w_extended, input_[:, w, h, :])
                output_[:, w: w + self.filter_size, h: h +
                        self.filter_size, :]
                # output_ += output_step
        return output_

    def get_output_shape_for(self, input_shape):
        l, w, h, m_b = input_shape
        output_width = (w + 2 * self.pad + self.filter_size -
                        self.stride) // self.stride

        output_height = (h + 2 * self.pad + self.filter_size -
                         self.stride) // self.stride
        output_shape = (self.n_filters, output_width, output_height, m_b)
        return output_shape

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

    def build(self, input_shape):
        if K.image_data_format() != 'channels_first':
            warnings.warn(
                "maybe wrong dim ordering in custom conv layer, ordering is %s, data format is" % (
                    K.image_dim_ordering(), K.image_data_format()))

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


# from keras1:


class MaxoutDense(Layer):
    """A dense maxout layer.
    A `MaxoutDense` layer takes the element-wise maximum of
    `nb_feature` `Dense(input_dim, output_dim)` linear layers.
    This allows the layer to learn a convex,
    piecewise linear activation function over the inputs.
    Note that this is a *linear* layer;
    if you wish to apply activation function
    (you shouldn't need to --they are universal function approximators),
    an `Activation` layer must be added after.
    # Arguments
        output_dim: int > 0.
        nb_feature: number of Dense layers to use internally.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    # References
        - [Maxout Networks](http://arxiv.org/abs/1302.4389)
    """

    def __init__(self, output_dim,
                 nb_feature=4,
                 init='glorot_uniform',
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 **kwargs):
        self.output_dim = output_dim
        self.nb_feature = nb_feature
        self.init = initializers.get(init)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MaxoutDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    shape=(None, input_dim))

        self.W = self.add_weight((self.nb_feature, input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.nb_feature, self.output_dim,),
                                     initializer='zero',
                                     name='b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def call(self, x):
        # no activation, this layer is only linear.
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        output = K.max(output, axis=1)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': initializers.serialize(self.init),
                  'nb_feature': self.nb_feature,
                  'W_regularizer': regularizers.serialize(self.W_regularizer),
                  'b_regularizer': regularizers.serialize(self.b_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'W_constraint': constraints.serialize(self.W_constraint),
                  'b_constraint': constraints.serialize(self.b_constraint),
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(MaxoutDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
