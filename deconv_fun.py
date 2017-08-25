import numpy as np
import keras.backend as K
import theano.tensor as T
import theano


# def deconv_fun(input_, input_shape_,  weights_, output_shape_, filter_size_):
#     output_ = K.zeros(output_shape_)
#     for w in range(input_shape_[1]):
#         for h in range(input_shape_[2]):
#             weights_ = K.variable(weights_)
#             try:
#                 output_step = K.dot(weights_, input_[
#                     :, w: w + filter_size_, h: h + filter_size_, :])
#                 output_ = output_ + output_step
#             except (ValueError, theano.gof.fg.MissingInputError):
#                 print 'Value Error or Missing Input occured'
#     return output_


def deconv_fun_output_shape(x):
    return x
