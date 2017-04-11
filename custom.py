"""
Custom stuff that is specific to the galaxy contest
"""

import theano
import theano.tensor as T
import numpy as np
from consider_constant import consider_constant


def taylorLog3(z):
    x = z - 1
    return x - (x**2) / 2 + (x**3) / 3


def logloss(act, pred):
    """ Vectorised computation of logloss """

    # cap in official Kaggle implementation,
    # per forums/t/1576/r-code-for-logloss
    # epsilon = 1e-15 #good for float64, too small for float32
    epsilon = 1e-7

    #pred = np.maximum(epsilon,pred)
    #pred = np.minimum(1-epsilon,pred)

    #pred = T.maximum(epsilon,pred)
    #pred = T.minimum(1-epsilon,pred)

    pred = T.clip(pred, epsilon, 1 - epsilon)

    # compute logloss function (vectorised)
    # ll = T.mean(   act*T.log(pred) +
    #            np.subtract(1,act)*T.log(T.sub(1,pred)))

    #ll = T.mean( act*taylorLog3(pred) + (1-act)*taylorLog3(1-pred)  )
    ll = T.mean(act * T.log(pred) + (1 - act) * T.log(1 - pred))
    ll = ll * -1.0

    return ll


def clip_01(x):
    # custom nonlinearity that is linear between [0,1] and clips to the
    # boundaries outside of this interval.
    return T.clip(x, 0, 1)


def tc_exp(x, t):
    """
    A version of the exponential that returns 0 below a certain threshold.
    """
    return T.maximum(T.exp(x + np.log(1 + t)) - t, 0)


def tc_softmax(x, t):
    x_c = x - T.max(x, axis=1, keepdims=True)
    x_e = tc_exp(x_c, t)
    return x_e / T.sum(x_e, axis=1, keepdims=True)


class GalaxyOutputLayer(object):
    """
    This layer expects the layer before to have 37 linear outputs. These are grouped per question and then passed through a softmax each,
    to encode for the fact that the probabilities of all the answers to a question should sum to one.

    Then, these probabilities are re-weighted as described in the competition info, and the MSE of the re-weighted probabilities is the loss function.
    """

    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.target_var = T.matrix()  # variable for the labels

    def targets(self, *args, **kwargs):
        q1 = self.target_var[:, 0:3]  # 1.1 - 1.3
        q2 = self.target_var[:, 3:5]  # 2.1 - 2.2
        q3 = self.target_var[:, 5:7]  # 3.1 - 3.2
        q4 = self.target_var[:, 7:9]  # 4.1 - 4.2
        q5 = self.target_var[:, 9:13]  # 5.1 - 5.4
        q6 = self.target_var[:, 13:15]  # 6.1 - 6.2
        q7 = self.target_var[:, 15:18]  # 7.1 - 7.3
        q8 = self.target_var[:, 18:25]  # 8.1 - 8.7
        q9 = self.target_var[:, 25:28]  # 9.1 - 9.3
        q10 = self.target_var[:, 28:31]  # 10.1 - 10.3
        q11 = self.target_var[:, 31:37]  # 11.1 - 11.6

        return q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11

    def answer_probabilities(self, *args, **kwargs):
        """
        apply softmax functions to the answer groups for each question.
        """
        input = self.input_layer.output(*args, **kwargs)
        q1 = T.nnet.softmax(input[:, 0:3])  # 1.1 - 1.3
        q2 = T.nnet.softmax(input[:, 3:5])  # 2.1 - 2.2
        q3 = T.nnet.softmax(input[:, 5:7])  # 3.1 - 3.2
        q4 = T.nnet.softmax(input[:, 7:9])  # 4.1 - 4.2
        q5 = T.nnet.softmax(input[:, 9:13])  # 5.1 - 5.4
        q6 = T.nnet.softmax(input[:, 13:15])  # 6.1 - 6.2
        q7 = T.nnet.softmax(input[:, 15:18])  # 7.1 - 7.3
        q8 = T.nnet.softmax(input[:, 18:25])  # 8.1 - 8.7
        q9 = T.nnet.softmax(input[:, 25:28])  # 9.1 - 9.3
        q10 = T.nnet.softmax(input[:, 28:31])  # 10.1 - 10.3
        q11 = T.nnet.softmax(input[:, 31:37])  # 11.1 - 11.6

        return q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11

    def weighted_answer_probabilities(self, weight_with_targets=False, *args, **kwargs):
        answer_probabilities = self.answer_probabilities(*args, **kwargs)
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = answer_probabilities

        # weighting factors
        if weight_with_targets:
            t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = self.targets(
                *args, **kwargs)
            w1 = 1
            w2 = t1[:, 1]
            w3 = t2[:, 1]
            w4 = w3
            w7 = t1[:, 0]
            w9 = t2[:, 0]
            w10 = t4[:, 0]
            w11 = w10
            w5 = w4
            w6 = 1
            w8 = t6[:, 0]
        else:
            w1 = 1
            w2 = q1[:, 1]  # question 1 answer 2
            w3 = q2[:, 1] * w2  # question 2 answer 2 * w2
            w4 = w3
            w7 = q1[:, 0]  # question 1 answer 1
            w9 = q2[:, 0] * w2  # question 2 answer 1 * w2
            w10 = q4[:, 0] * w4  # question 4 answer 1 * w4
            w11 = w10
            w5 = w4  # THIS WAS WRONG BEFORE
            # THIS SHOULD TECHNICALLY BE w5 + w7 + w9, but as explained on the
            # forums, there was a mistake generating the dataset.
            w6 = 1
            # see
            # http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/forums/t/6706/is-question-6-also-answered-for-stars-artifacts-answer-1-3
            # for more info
            w8 = q6[:, 0] * w6  # question 6 answer 1 * w6

        # weighted answers
        wq1 = q1 * w1
        wq2 = q2 * w2.dimshuffle(0, 'x')
        wq3 = q3 * w3.dimshuffle(0, 'x')
        wq4 = q4 * w4.dimshuffle(0, 'x')
        wq5 = q5 * w5.dimshuffle(0, 'x')
        wq6 = q6 * w6  # w6.dimshuffle(0, 'x')
        wq7 = q7 * w7.dimshuffle(0, 'x')
        wq8 = q8 * w8.dimshuffle(0, 'x')
        wq9 = q9 * w9.dimshuffle(0, 'x')
        wq10 = q10 * w10.dimshuffle(0, 'x')
        wq11 = q11 * w11.dimshuffle(0, 'x')

        return wq1, wq2, wq3, wq4, wq5, wq6, wq7, wq8, wq9, wq10, wq11

    def error(self, *args, **kwargs):
        predictions = self.predictions(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
        return error

    def predictions(self, *args, **kwargs):
        # concatenate all the columns together.
        return T.concatenate(self.weighted_answer_probabilities(*args, **kwargs), axis=1)
        # This might not be the best way to do this since we're summing everything afterwards.
        # Might be better to just write all of it as a sum straight away.


class ThresholdedGalaxyOutputLayer(object):
    """
    This layer expects the layer before to have 37 linear outputs. These are grouped per question and then passed through a softmax each,
    to encode for the fact that the probabilities of all the answers to a question should sum to one.

    The softmax function used is a special version with a threshold, such that it can return hard 0s and 1s for certain values.

    Then, these probabilities are re-weighted as described in the competition info, and the MSE of the re-weighted probabilities is the loss function.
    """

    def __init__(self, input_layer, threshold):
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.threshold = threshold
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.target_var = T.matrix()  # variable for the labels

    def answer_probabilities(self, *args, **kwargs):
        """
        apply softmax functions to the answer groups for each question.
        """
        input = self.input_layer.output(*args, **kwargs)
        q1 = tc_softmax(input[:, 0:3], self.threshold)  # 1.1 - 1.3
        q2 = tc_softmax(input[:, 3:5], self.threshold)  # 2.1 - 2.2
        q3 = tc_softmax(input[:, 5:7], self.threshold)  # 3.1 - 3.2
        q4 = tc_softmax(input[:, 7:9], self.threshold)  # 4.1 - 4.2
        q5 = tc_softmax(input[:, 9:13], self.threshold)  # 5.1 - 5.4
        q6 = tc_softmax(input[:, 13:15], self.threshold)  # 6.1 - 6.2
        q7 = tc_softmax(input[:, 15:18], self.threshold)  # 7.1 - 7.3
        q8 = tc_softmax(input[:, 18:25], self.threshold)  # 8.1 - 8.7
        q9 = tc_softmax(input[:, 25:28], self.threshold)  # 9.1 - 9.3
        q10 = tc_softmax(input[:, 28:31], self.threshold)  # 10.1 - 10.3
        q11 = tc_softmax(input[:, 31:37], self.threshold)  # 11.1 - 11.6

        return q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11

    def weighted_answer_probabilities(self, *args, **kwargs):
        answer_probabilities = self.answer_probabilities(*args, **kwargs)
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = answer_probabilities

        # weighting factors
        w1 = 1
        w2 = q1[:, 1]  # question 1 answer 2
        w3 = q2[:, 1] * w2  # question 2 answer 2 * w2
        w4 = w3
        w7 = q1[:, 0]  # question 1 answer 1
        w9 = q2[:, 0] * w2  # question 2 answer 1 * w2
        w10 = q4[:, 0] * w4  # question 4 answer 1 * w4
        w11 = w10
        w5 = w4  # THIS WAS WRONG BEFORE
        # THIS SHOULD TECHNICALLY BE w5 + w7 + w9, but as explained on the
        # forums, there was a mistake generating the dataset.
        w6 = 1
        # see
        # http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/forums/t/6706/is-question-6-also-answered-for-stars-artifacts-answer-1-3
        # for more info
        w8 = q6[:, 0] * w6  # question 6 answer 1 * w6

        # weighted answers
        wq1 = q1 * w1
        wq2 = q2 * w2.dimshuffle(0, 'x')
        wq3 = q3 * w3.dimshuffle(0, 'x')
        wq4 = q4 * w4.dimshuffle(0, 'x')
        wq5 = q5 * w5.dimshuffle(0, 'x')
        wq6 = q6 * w6  # w6.dimshuffle(0, 'x')
        wq7 = q7 * w7.dimshuffle(0, 'x')
        wq8 = q8 * w8.dimshuffle(0, 'x')
        wq9 = q9 * w9.dimshuffle(0, 'x')
        wq10 = q10 * w10.dimshuffle(0, 'x')
        wq11 = q11 * w11.dimshuffle(0, 'x')

        return wq1, wq2, wq3, wq4, wq5, wq6, wq7, wq8, wq9, wq10, wq11

    def error(self, *args, **kwargs):
        predictions = self.predictions(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
        return error

    def predictions(self, *args, **kwargs):
        # concatenate all the columns together.
        return T.concatenate(self.weighted_answer_probabilities(*args, **kwargs), axis=1)
        # This might not be the best way to do this since we're summing everything afterwards.
        # Might be better to just write all of it as a sum straight away.


class DivisiveGalaxyOutputLayer(object):
    """
    This layer expects the layer before to have 37 linear outputs. These are grouped per question, clipped, and then normalised by dividing by the sum,
    to encode for the fact that the probabilities of all the answers to a question should sum to one.

    Then, these probabilities are re-weighted as described in the competition info, and the MSE of the re-weighted probabilities is the loss function.
    """

    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.target_var = T.matrix()  # variable for the labels

    def answer_probabilities(self, *args, **kwargs):
        """
        normalise the answer groups for each question.
        """
        input = self.input_layer.output(*args, **kwargs)
        # T.clip(input, 0, 1) # T.maximum(input, 0)
        input_clipped = T.maximum(input, 0)

        q1 = input_clipped[:, 0:3]  # 1.1 - 1.3
        q2 = input_clipped[:, 3:5]  # 2.1 - 2.2
        q3 = input_clipped[:, 5:7]  # 3.1 - 3.2
        q4 = input_clipped[:, 7:9]  # 4.1 - 4.2
        q5 = input_clipped[:, 9:13]  # 5.1 - 5.4
        q6 = input_clipped[:, 13:15]  # 6.1 - 6.2
        q7 = input_clipped[:, 15:18]  # 7.1 - 7.3
        q8 = input_clipped[:, 18:25]  # 8.1 - 8.7
        q9 = input_clipped[:, 25:28]  # 9.1 - 9.3
        q10 = input_clipped[:, 28:31]  # 10.1 - 10.3
        q11 = input_clipped[:, 31:37]  # 11.1 - 11.6

        # what if the sums are 0?
        # adding a very small constant works, but then the probabilities don't sum to 1 anymore.
        # is there a better way?

        q1 = q1 / (q1.sum(1, keepdims=True) + 1e-12)
        q2 = q2 / (q2.sum(1, keepdims=True) + 1e-12)
        q3 = q3 / (q3.sum(1, keepdims=True) + 1e-12)
        q4 = q4 / (q4.sum(1, keepdims=True) + 1e-12)
        q5 = q5 / (q5.sum(1, keepdims=True) + 1e-12)
        q6 = q6 / (q6.sum(1, keepdims=True) + 1e-12)
        q7 = q7 / (q7.sum(1, keepdims=True) + 1e-12)
        q8 = q8 / (q8.sum(1, keepdims=True) + 1e-12)
        q9 = q9 / (q9.sum(1, keepdims=True) + 1e-12)
        q10 = q10 / (q10.sum(1, keepdims=True) + 1e-12)
        q11 = q11 / (q11.sum(1, keepdims=True) + 1e-12)

        return q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11

    def weighted_answer_probabilities(self, *args, **kwargs):
        answer_probabilities = self.answer_probabilities(*args, **kwargs)
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = answer_probabilities

        # weighting factors
        w1 = 1
        w2 = q1[:, 1]  # question 1 answer 2
        w3 = q2[:, 1] * w2  # question 2 answer 2 * w2
        w4 = w3
        w7 = q1[:, 0]  # question 1 answer 1
        w9 = q2[:, 0] * w2  # question 2 answer 1 * w2
        w10 = q4[:, 0] * w4  # question 4 answer 1 * w4
        w11 = w10
        w5 = w4  # THIS WAS WRONG BEFORE
        # THIS SHOULD TECHNICALLY BE w5 + w7 + w9, but as explained on the
        # forums, there was a mistake generating the dataset.
        w6 = 1
        # see
        # http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/forums/t/6706/is-question-6-also-answered-for-stars-artifacts-answer-1-3
        # for more info
        w8 = q6[:, 0] * w6  # question 6 answer 1 * w6

        # weighted answers
        wq1 = q1 * w1
        wq2 = q2 * w2.dimshuffle(0, 'x')
        wq3 = q3 * w3.dimshuffle(0, 'x')
        wq4 = q4 * w4.dimshuffle(0, 'x')
        wq5 = q5 * w5.dimshuffle(0, 'x')
        wq6 = q6 * w6  # w6.dimshuffle(0, 'x')
        wq7 = q7 * w7.dimshuffle(0, 'x')
        wq8 = q8 * w8.dimshuffle(0, 'x')
        wq9 = q9 * w9.dimshuffle(0, 'x')
        wq10 = q10 * w10.dimshuffle(0, 'x')
        wq11 = q11 * w11.dimshuffle(0, 'x')

        return wq1, wq2, wq3, wq4, wq5, wq6, wq7, wq8, wq9, wq10, wq11

    def error(self, *args, **kwargs):
        predictions = self.predictions(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
        return error

    def predictions(self, *args, **kwargs):
        # concatenate all the columns together.
        return T.concatenate(self.weighted_answer_probabilities(*args, **kwargs), axis=1)
        # This might not be the best way to do this since we're summing everything afterwards.
        # Might be better to just write all of it as a sum straight away.


class SquaredGalaxyOutputLayer(object):
    """
    This layer expects the layer before to have 37 linear outputs. These are grouped per question, rectified, squared and then normalised by dividing by the sum,
    to encode for the fact that the probabilities of all the answers to a question should sum to one.

    Then, these probabilities are re-weighted as described in the competition info, and the MSE of the re-weighted probabilities is the loss function.
    """

    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.target_var = T.matrix()  # variable for the labels

    def answer_probabilities(self, *args, **kwargs):
        """
        normalise the answer groups for each question.
        """
        input = self.input_layer.output(*args, **kwargs)
        input_rectified = T.maximum(input, 0)
        input_squared = input_rectified ** 2

        q1 = input_squared[:, 0:3]  # 1.1 - 1.3
        q2 = input_squared[:, 3:5]  # 2.1 - 2.2
        q3 = input_squared[:, 5:7]  # 3.1 - 3.2
        q4 = input_squared[:, 7:9]  # 4.1 - 4.2
        q5 = input_squared[:, 9:13]  # 5.1 - 5.4
        q6 = input_squared[:, 13:15]  # 6.1 - 6.2
        q7 = input_squared[:, 15:18]  # 7.1 - 7.3
        q8 = input_squared[:, 18:25]  # 8.1 - 8.7
        q9 = input_squared[:, 25:28]  # 9.1 - 9.3
        q10 = input_squared[:, 28:31]  # 10.1 - 10.3
        q11 = input_squared[:, 31:37]  # 11.1 - 11.6

        # what if the sums are 0?
        # adding a very small constant works, but then the probabilities don't sum to 1 anymore.
        # is there a better way?

        q1 = q1 / (q1.sum(1, keepdims=True) + 1e-12)
        q2 = q2 / (q2.sum(1, keepdims=True) + 1e-12)
        q3 = q3 / (q3.sum(1, keepdims=True) + 1e-12)
        q4 = q4 / (q4.sum(1, keepdims=True) + 1e-12)
        q5 = q5 / (q5.sum(1, keepdims=True) + 1e-12)
        q6 = q6 / (q6.sum(1, keepdims=True) + 1e-12)
        q7 = q7 / (q7.sum(1, keepdims=True) + 1e-12)
        q8 = q8 / (q8.sum(1, keepdims=True) + 1e-12)
        q9 = q9 / (q9.sum(1, keepdims=True) + 1e-12)
        q10 = q10 / (q10.sum(1, keepdims=True) + 1e-12)
        q11 = q11 / (q11.sum(1, keepdims=True) + 1e-12)

        return q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11

    def weighted_answer_probabilities(self, *args, **kwargs):
        answer_probabilities = self.answer_probabilities(*args, **kwargs)
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = answer_probabilities

        # weighting factors
        w1 = 1
        w2 = q1[:, 1]  # question 1 answer 2
        w3 = q2[:, 1] * w2  # question 2 answer 2 * w2
        w4 = w3
        w7 = q1[:, 0]  # question 1 answer 1
        w9 = q2[:, 0] * w2  # question 2 answer 1 * w2
        w10 = q4[:, 0] * w4  # question 4 answer 1 * w4
        w11 = w10
        w5 = w4  # THIS WAS WRONG BEFORE
        # THIS SHOULD TECHNICALLY BE w5 + w7 + w9, but as explained on the
        # forums, there was a mistake generating the dataset.
        w6 = 1
        # see
        # http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/forums/t/6706/is-question-6-also-answered-for-stars-artifacts-answer-1-3
        # for more info
        w8 = q6[:, 0] * w6  # question 6 answer 1 * w6

        # weighted answers
        wq1 = q1 * w1
        wq2 = q2 * w2.dimshuffle(0, 'x')
        wq3 = q3 * w3.dimshuffle(0, 'x')
        wq4 = q4 * w4.dimshuffle(0, 'x')
        wq5 = q5 * w5.dimshuffle(0, 'x')
        wq6 = q6 * w6  # w6.dimshuffle(0, 'x')
        wq7 = q7 * w7.dimshuffle(0, 'x')
        wq8 = q8 * w8.dimshuffle(0, 'x')
        wq9 = q9 * w9.dimshuffle(0, 'x')
        wq10 = q10 * w10.dimshuffle(0, 'x')
        wq11 = q11 * w11.dimshuffle(0, 'x')

        return wq1, wq2, wq3, wq4, wq5, wq6, wq7, wq8, wq9, wq10, wq11

    def error(self, *args, **kwargs):
        predictions = self.predictions(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
        return error

    def predictions(self, *args, **kwargs):
        # concatenate all the columns together.
        return T.concatenate(self.weighted_answer_probabilities(*args, **kwargs), axis=1)
        # This might not be the best way to do this since we're summing everything afterwards.
        # Might be better to just write all of it as a sum straight away.


class ClippedGalaxyOutputLayer(object):
    """
    This layer expects the layer before to have 37 linear outputs. These are grouped per question, clipped, but NOT normalised, because it seems
    like this might be impeding the learning.

    Then, these probabilities are re-weighted as described in the competition info, and the MSE of the re-weighted probabilities is the loss function.
    """

    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.target_var = T.matrix()  # variable for the labels

    def answer_probabilities(self, *args, **kwargs):
        """
        normalise the answer groups for each question.
        """
        input = self.input_layer.output(*args, **kwargs)
        input_clipped = T.clip(input, 0, 1)  # T.maximum(input, 0)

        q1 = input_clipped[:, 0:3]  # 1.1 - 1.3
        q2 = input_clipped[:, 3:5]  # 2.1 - 2.2
        q3 = input_clipped[:, 5:7]  # 3.1 - 3.2
        q4 = input_clipped[:, 7:9]  # 4.1 - 4.2
        q5 = input_clipped[:, 9:13]  # 5.1 - 5.4
        q6 = input_clipped[:, 13:15]  # 6.1 - 6.2
        q7 = input_clipped[:, 15:18]  # 7.1 - 7.3
        q8 = input_clipped[:, 18:25]  # 8.1 - 8.7
        q9 = input_clipped[:, 25:28]  # 9.1 - 9.3
        q10 = input_clipped[:, 28:31]  # 10.1 - 10.3
        q11 = input_clipped[:, 31:37]  # 11.1 - 11.6

        return q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11

    def targets(self, *args, **kwargs):
        q1 = self.target_var[:, 0:3]  # 1.1 - 1.3
        q2 = self.target_var[:, 3:5]  # 2.1 - 2.2
        q3 = self.target_var[:, 5:7]  # 3.1 - 3.2
        q4 = self.target_var[:, 7:9]  # 4.1 - 4.2
        q5 = self.target_var[:, 9:13]  # 5.1 - 5.4
        q6 = self.target_var[:, 13:15]  # 6.1 - 6.2
        q7 = self.target_var[:, 15:18]  # 7.1 - 7.3
        q8 = self.target_var[:, 18:25]  # 8.1 - 8.7
        q9 = self.target_var[:, 25:28]  # 9.1 - 9.3
        q10 = self.target_var[:, 28:31]  # 10.1 - 10.3
        q11 = self.target_var[:, 31:37]  # 11.1 - 11.6

        return q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11

    def weighted_answer_probabilities(self, weight_with_targets=False, *args, **kwargs):
        answer_probabilities = self.answer_probabilities(*args, **kwargs)
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = answer_probabilities

        # weighting factors
        if weight_with_targets:
            w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11 = self.question_weights(
                self.targets(*args, **kwargs))
        else:
            w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11 = self.question_weights(
                answer_probabilities)

        # weighted answers
        wq1 = q1 * w1
        wq2 = q2 * w2.dimshuffle(0, 'x')
        wq3 = q3 * w3.dimshuffle(0, 'x')
        wq4 = q4 * w4.dimshuffle(0, 'x')
        wq5 = q5 * w5.dimshuffle(0, 'x')
        wq6 = q6 * w6  # w6.dimshuffle(0, 'x')
        wq7 = q7 * w7.dimshuffle(0, 'x')
        wq8 = q8 * w8.dimshuffle(0, 'x')
        wq9 = q9 * w9.dimshuffle(0, 'x')
        wq10 = q10 * w10.dimshuffle(0, 'x')
        wq11 = q11 * w11.dimshuffle(0, 'x')

        return wq1, wq2, wq3, wq4, wq5, wq6, wq7, wq8, wq9, wq10, wq11

    def error(self, *args, **kwargs):
        predictions = self.predictions(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
        return error

    def question_weights(self, q):
        """
        q is a list of matrices of length 11 (one for each question), like the output given by targets() and answer_probabilities()
        """
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = q  # unpack
        w1 = 1
        w2 = q1[:, 1]  # question 1 answer 2
        w3 = q2[:, 1] * w2  # question 2 answer 2 * w2
        w4 = w3
        w7 = q1[:, 0]  # question 1 answer 1
        w9 = q2[:, 0] * w2  # question 2 answer 1 * w2
        w10 = q4[:, 0] * w4  # question 4 answer 1 * w4
        w11 = w10
        w5 = w4  # THIS WAS WRONG BEFORE
        # THIS SHOULD TECHNICALLY BE w5 + w7 + w9, but as explained on the
        # forums, there was a mistake generating the dataset.
        w6 = 1
        # see
        # http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/forums/t/6706/is-question-6-also-answered-for-stars-artifacts-answer-1-3
        # for more info
        w8 = q6[:, 0] * w6  # question 6 answer 1 * w6

        return w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11

    def normreg(self, direct_weighting=True, *args, **kwargs):
        """
        direct_weighting: if True, the weighting is applied directly to the constraints (before the barrier function).
        if False, all constraints are 'sum-to-1' and the weighting is applied after the barrier function.
        """
        answer_probabilities = self.answer_probabilities(*args, **kwargs)
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = answer_probabilities
        weights = self.question_weights(answer_probabilities)

        constraints = [q.sum(1, keepdims=True) -
                       1 for q in answer_probabilities]

        def func(x): return x**2

        if direct_weighting:  # scale the constraints with the weights
            terms = [func(weight * constraint)
                     for weight, constraint in zip(weights, constraints)]
        else:
            terms = [weight * func(constraint)
                     for weight, constraint in zip(weights, constraints)]

        return T.mean(T.concatenate(terms, axis=1))

        # means = [T.mean(term) for term in terms] # mean over the minibatch
        # return sum(means)

    def error_with_normreg(self, scale=1.0, *args, **kwargs):
        error_term = self.error(*args, **kwargs)
        normreg_term = self.normreg(*args, **kwargs)
        return error_term + scale * normreg_term

    def predictions(self, *args, **kwargs):
        # concatenate all the columns together.
        return T.concatenate(self.weighted_answer_probabilities(*args, **kwargs), axis=1)
        # This might not be the best way to do this since we're summing everything afterwards.
        # Might be better to just write all of it as a sum straight away.


'''
		input_cat = [[0.]*(3),
			[0.]*(2),
			[0.]*(2),
			[0.]*(2),
			[0.]*(4),
			[0.]*(2),			
			[0.]*(3),
			[0.]*(7),
			[0.]*(3),
			[0.]*(3),
			[0.]*(6)
		]
'''

# categorised part not original


class OptimisedDivGalaxyOutputLayer(object):
    """
    divisive normalisation, optimised for performance.
    """

    def __init__(self, input_layer, categorised=False):
        self.input_layer = input_layer
        self.categorised = categorised
        self.input_shape = self.input_layer.get_output_shape()
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.target_var = T.matrix()  # variable for the labels

        self.question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9), slice(9, 13), slice(13, 15),
                                slice(15, 18), slice(18, 25), slice(25, 28), slice(28, 31), slice(31, 37)]

        # self.scaling_factor_indices = [None, [1], [1, 4], [1, 4], [1, 4], None, [0], [13], [1, 3], [1, 4, 7], [1, 4, 7]]
        # indices of all the probabilities that scale each question.

        self.normalisation_mask = theano.shared(
            self.generate_normalisation_mask())
        # self.scaling_mask = theano.shared(self.generate_scaling_mask())

        # sequence of scaling steps to be undertaken.
        # First element is a slice indicating the values to be scaled. Second element is an index indicating the scale factor.
        # these have to happen IN ORDER else it doesn't work correctly.
        self.scaling_sequence = [
            (slice(3, 5), 1),  # I: rescale Q2 by A1.2
            (slice(5, 13), 4),  # II: rescale Q3, Q4, Q5 by A2.2
            (slice(15, 18), 0),  # III: rescale Q7 by A1.1
            (slice(18, 25), 13),  # IV: rescale Q8 by A6.1
            (slice(25, 28), 3),  # V: rescale Q9 by A2.1
            (slice(28, 37), 7),  # VI: rescale Q10, Q11 by A4.1
        ]

    def generate_normalisation_mask(self):
        """
        when the clipped input is multiplied by the normalisation mask, the normalisation denominators are generated.
        So then we can just divide the input by the normalisation constants (elementwise).
        """
        mask = np.zeros((37, 37), dtype='float32')
        for s in self.question_slices:
            mask[s, s] = 1.0
        return mask

    # def generate_scaling_mask(self):
    #     """
    #     This mask needs to be applied to the LOGARITHM of the probabilities. The appropriate log probs are then summed,
    #     which corresponds to multiplying the raw probabilities, which is what we want to achieve.
    #     """
    #     mask = np.zeros((37, 37), dtype='float32')
    #     for s, factor_indices in zip(self.question_slices, self.scaling_factor_indices):
    #         if factor_indices is not None:
    #             mask[factor_indices, s] = 1.0
    #     return mask

    def answer_probabilities(self, *args, **kwargs):
        """
        normalise the answer groups for each question.
        """
        input = self.input_layer.output(*args, **kwargs)
        # T.clip(input, 0, 1) # T.maximum(input, 0)
        input_clipped = T.maximum(input, 0)

        # small constant to prevent division by 0
        normalisation_denoms = T.dot(
            input_clipped, self.normalisation_mask) + 1e-7
        input_normalised = input_clipped / normalisation_denoms

        if self.categorised:
            output = []
            for k in xrange(0, self.mb_size):
                input_q = [
                    input_normalised[k][0:3],  # 1.1 - 1.3,
                    input_normalised[k][3:5],  # 2.1 - 2.2
                    input_normalised[k][5:7],  # 3.1 - 3.2
                    input_normalised[k][7:9],  # 4.1 - 4.2
                    input_normalised[k][9:13],  # 5.1 - 5.4
                    input_normalised[k][13:15],  # 6.1 - 6.2
                    input_normalised[k][15:18],  # 7.1 - 7.3
                    input_normalised[k][18:25],  # 8.1 - 8.7
                    input_normalised[k][25:28],  # 9.1 - 9.3
                    input_normalised[k][28:31],  # 10.1 - 10.3
                    input_normalised[k][31:37],  # 11.1 - 11.6
                ]
                z_v = []
                for i in xrange(0, len(input_q)):
                    z = 1.
                    z = z * (1. - T.gt(input_q[0][2], 0.6))
                    if i == 1:
                        z = z * T.gt(input_q[0][1], 0.6)
                    if i == 2:
                        z = z * T.gt(input_q[1][1], 0.6)
                    if i == 3:
                        z = z * T.gt(input_q[1][1], 0.6)
                    if i == 4:
                        z = z * T.gt(input_q[1][1], 0.6)
                    if i == 6:
                        z = z * T.gt(input_q[0][0], 0.6)
                    if i == 9:
                        z = z * T.gt(input_q[4][0], 0.6)
                    if i == 10:
                        z = z * T.gt(input_q[4][0], 0.6)
                    for j in xrange(0, len(input_q)):
                        z_v.append(z)
                # FIXME must that be z or z_v here
                output.append(T.dot(input_normalised[k], z))
            # print output
            return output
            # input_normaised = sum(input_q,[]) #flattens lists of lists
            # input_normalised[0:3]=input_q[0]
            # input_normalised[3:5]=input_q[1]
            # input_normalised[5:7]=input_q[2]
            # input_normalised[7:9]=input_q[3]
            # input_normalised[9:13]=input_q[4]
            # input_normalised[13:15]=input_q[5]
            # input_normalised[15:18]=input_q[6]
            # input_normalised[18:25]=input_q[7]
            # input_normalised[25:28]=input_q[8]
            # input_normalised[28:31]=input_q[9]
            # input_normalised[31:37]=input_q[10]

        return input_normalised
        # return [input_normalised[:, s] for s in self.question_slices]

    # def weighted_answer_probabilities(self, *args, **kwargs):
    #     answer_probabilities = self.answer_probabilities(*args, **kwargs)

    #     log_scale_factors = T.dot(T.log(answer_probabilities), self.scaling_mask)
    # scale_factors = T.exp(T.switch(T.isnan(log_scale_factors), -np.inf,
    # log_scale_factors)) # need NaN shielding here because 0 * -inf = NaN.

    #     return answer_probabilities * scale_factors

    def weighted_answer_probabilities(self, *args, **kwargs):
        probs = self.answer_probabilities(*args, **kwargs)

        # go through the rescaling sequence in order (6 steps)
        if not self.categorised:
            for probs_slice, scale_idx in self.scaling_sequence:
                probs = T.set_subtensor(
                    probs[:, probs_slice], probs[:, probs_slice] * probs[:, scale_idx].dimshuffle(0, 'x'))

        return probs

    def predictions(self, normalisation=True, *args, **kwargs):
        return self.weighted_answer_probabilities(*args, **kwargs)

    def predictions_no_normalisation(self, *args, **kwargs):
        """
        Predict without normalisation. This can be used for the first few chunks to find good parameters.
        """
        input = self.input_layer.output(*args, **kwargs)
        # clip on both sides here, any predictions over 1.0 are going to get
        # normalised away anyway.
        input_clipped = T.clip(input, 0, 1)
        return input_clipped

    def error(self, normalisation=True, *args, **kwargs):
        if normalisation:
            predictions = self.predictions(*args, **kwargs)
        else:
            predictions = self.predictions_no_normalisation(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
        return error

# not original, not in use
    def sqrtError(self, normalisation=True, *args, **kwargs):
        return T.sqrt(self.error(normalisation=True, *args, **kwargs))

#not original

    def error_weighted(self, weight, normalisation=True, *args, **kwargs):
        if normalisation:
            predictions = self.predictions(*args, **kwargs)
        else:
            predictions = self.predictions_no_normalisation(*args, **kwargs)
        error = T.mean(((predictions - self.target_var) * weight) ** 2)
        return error
#not original
# not quadratic like the error!

    def ll_error(self, normalisation=True, *args, **kwargs):
        if normalisation:
            predictions = self.predictions(*args, **kwargs)
        else:
            predictions = self.predictions_no_normalisation(*args, **kwargs)
        error = logloss(self.target_var, predictions)
        return error


class ConstantWeightedDivGalaxyOutputLayer(object):
    """
    divisive normalisation, weights are considered constant when differentiating, optimised for performance.
    """

    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.target_var = T.matrix()  # variable for the labels

        self.question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9), slice(9, 13), slice(13, 15),
                                slice(15, 18), slice(18, 25), slice(25, 28), slice(28, 31), slice(31, 37)]

        # self.scaling_factor_indices = [None, [1], [1, 4], [1, 4], [1, 4], None, [0], [13], [1, 3], [1, 4, 7], [1, 4, 7]]
        # indices of all the probabilities that scale each question.

        self.normalisation_mask = theano.shared(
            self.generate_normalisation_mask())
        # self.scaling_mask = theano.shared(self.generate_scaling_mask())

        # sequence of scaling steps to be undertaken.
        # First element is a slice indicating the values to be scaled. Second element is an index indicating the scale factor.
        # these have to happen IN ORDER else it doesn't work correctly.
        self.scaling_sequence = [
            (slice(3, 5), 1),  # I: rescale Q2 by A1.2
            (slice(5, 13), 4),  # II: rescale Q3, Q4, Q5 by A2.2
            (slice(15, 18), 0),  # III: rescale Q7 by A1.1
            (slice(18, 25), 13),  # IV: rescale Q8 by A6.1
            (slice(25, 28), 3),  # V: rescale Q9 by A2.1
            (slice(28, 37), 7),  # VI: rescale Q10, Q11 by A4.1
        ]

    def generate_normalisation_mask(self):
        """
        when the clipped input is multiplied by the normalisation mask, the normalisation denominators are generated.
        So then we can just divide the input by the normalisation constants (elementwise).
        """
        mask = np.zeros((37, 37), dtype='float32')
        for s in self.question_slices:
            mask[s, s] = 1.0
        return mask

    # def generate_scaling_mask(self):
    #     """
    #     This mask needs to be applied to the LOGARITHM of the probabilities. The appropriate log probs are then summed,
    #     which corresponds to multiplying the raw probabilities, which is what we want to achieve.
    #     """
    #     mask = np.zeros((37, 37), dtype='float32')
    #     for s, factor_indices in zip(self.question_slices, self.scaling_factor_indices):
    #         if factor_indices is not None:
    #             mask[factor_indices, s] = 1.0
    #     return mask

    def answer_probabilities(self, *args, **kwargs):
        """
        normalise the answer groups for each question.
        """
        input = self.input_layer.output(*args, **kwargs)
        # T.clip(input, 0, 1) # T.maximum(input, 0)
        input_clipped = T.maximum(input, 0)

        # small constant to prevent division by 0
        normalisation_denoms = T.dot(
            input_clipped, self.normalisation_mask) + 1e-12
        input_normalised = input_clipped / normalisation_denoms

        return input_normalised
        # return [input_normalised[:, s] for s in self.question_slices]

    # def weighted_answer_probabilities(self, *args, **kwargs):
    #     answer_probabilities = self.answer_probabilities(*args, **kwargs)

    #     log_scale_factors = T.dot(T.log(answer_probabilities), self.scaling_mask)
    # scale_factors = T.exp(T.switch(T.isnan(log_scale_factors), -np.inf,
    # log_scale_factors)) # need NaN shielding here because 0 * -inf = NaN.

    #     return answer_probabilities * scale_factors

    def weighted_answer_probabilities(self, *args, **kwargs):
        probs = self.answer_probabilities(*args, **kwargs)

        # go through the rescaling sequence in order (6 steps)
        for probs_slice, scale_idx in self.scaling_sequence:
            probs = T.set_subtensor(probs[:, probs_slice], probs[:, probs_slice]
                                    * consider_constant(probs[:, scale_idx].dimshuffle(0, 'x')))

        return probs

    def predictions(self, normalisation=True, *args, **kwargs):
        return self.weighted_answer_probabilities(*args, **kwargs)

    def predictions_no_normalisation(self, *args, **kwargs):
        """
        Predict without normalisation. This can be used for the first few chunks to find good parameters.
        """
        input = self.input_layer.output(*args, **kwargs)
        # clip on both sides here, any predictions over 1.0 are going to get
        # normalised away anyway.
        input_clipped = T.clip(input, 0, 1)
        return input_clipped

    def error(self, normalisation=True, *args, **kwargs):
        if normalisation:
            predictions = self.predictions(*args, **kwargs)
        else:
            predictions = self.predictions_no_normalisation(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
        return error


class SoftplusDivGalaxyOutputLayer(object):
    """
    divisive normalisation with softplus function, optimised for performance.
    """

    def __init__(self, input_layer, scale=10.0):
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size
        self.scale = scale

        self.target_var = T.matrix()  # variable for the labels

        self.question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9), slice(9, 13), slice(13, 15),
                                slice(15, 18), slice(18, 25), slice(25, 28), slice(28, 31), slice(31, 37)]

        # self.scaling_factor_indices = [None, [1], [1, 4], [1, 4], [1, 4], None, [0], [13], [1, 3], [1, 4, 7], [1, 4, 7]]
        # indices of all the probabilities that scale each question.

        self.normalisation_mask = theano.shared(
            self.generate_normalisation_mask())
        # self.scaling_mask = theano.shared(self.generate_scaling_mask())

        # sequence of scaling steps to be undertaken.
        # First element is a slice indicating the values to be scaled. Second element is an index indicating the scale factor.
        # these have to happen IN ORDER else it doesn't work correctly.
        self.scaling_sequence = [
            (slice(3, 5), 1),  # I: rescale Q2 by A1.2
            (slice(5, 13), 4),  # II: rescale Q3, Q4, Q5 by A2.2
            (slice(15, 18), 0),  # III: rescale Q7 by A1.1
            (slice(18, 25), 13),  # IV: rescale Q8 by A6.1
            (slice(25, 28), 3),  # V: rescale Q9 by A2.1
            (slice(28, 37), 7),  # VI: rescale Q10, Q11 by A4.1
        ]

    def generate_normalisation_mask(self):
        """
        when the clipped input is multiplied by the normalisation mask, the normalisation denominators are generated.
        So then we can just divide the input by the normalisation constants (elementwise).
        """
        mask = np.zeros((37, 37), dtype='float32')
        for s in self.question_slices:
            mask[s, s] = 1.0
        return mask

    # def generate_scaling_mask(self):
    #     """
    #     This mask needs to be applied to the LOGARITHM of the probabilities. The appropriate log probs are then summed,
    #     which corresponds to multiplying the raw probabilities, which is what we want to achieve.
    #     """
    #     mask = np.zeros((37, 37), dtype='float32')
    #     for s, factor_indices in zip(self.question_slices, self.scaling_factor_indices):
    #         if factor_indices is not None:
    #             mask[factor_indices, s] = 1.0
    #     return mask

    def answer_probabilities(self, *args, **kwargs):
        """
        normalise the answer groups for each question.
        """
        input = self.input_layer.output(*args, **kwargs)
        # T.maximum(input, 0) # T.clip(input, 0, 1) # T.maximum(input, 0)
        input_clipped = T.nnet.softplus(input * self.scale)

        # small constant to prevent division by 0
        normalisation_denoms = T.dot(
            input_clipped, self.normalisation_mask) + 1e-12
        input_normalised = input_clipped / normalisation_denoms

        return input_normalised
        # return [input_normalised[:, s] for s in self.question_slices]

    # def weighted_answer_probabilities(self, *args, **kwargs):
    #     answer_probabilities = self.answer_probabilities(*args, **kwargs)

    #     log_scale_factors = T.dot(T.log(answer_probabilities), self.scaling_mask)
    # scale_factors = T.exp(T.switch(T.isnan(log_scale_factors), -np.inf,
    # log_scale_factors)) # need NaN shielding here because 0 * -inf = NaN.

    #     return answer_probabilities * scale_factors

    def weighted_answer_probabilities(self, *args, **kwargs):
        probs = self.answer_probabilities(*args, **kwargs)

        # go through the rescaling sequence in order (6 steps)
        for probs_slice, scale_idx in self.scaling_sequence:
            probs = T.set_subtensor(
                probs[:, probs_slice], probs[:, probs_slice] * probs[:, scale_idx].dimshuffle(0, 'x'))

        return probs

    def predictions(self, normalisation=True, *args, **kwargs):
        return self.weighted_answer_probabilities(*args, **kwargs)

    def predictions_no_normalisation(self, *args, **kwargs):
        """
        Predict without normalisation. This can be used for the first few chunks to find good parameters.
        """
        input = self.input_layer.output(*args, **kwargs)
        # clip on both sides here, any predictions over 1.0 are going to get
        # normalised away anyway.
        input_clipped = T.clip(input, 0, 1)
        return input_clipped

    def error(self, normalisation=True, *args, **kwargs):
        if normalisation:
            predictions = self.predictions(*args, **kwargs)
        else:
            predictions = self.predictions_no_normalisation(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
        return error


class SpiralArmsOnlyDivGalaxyOutputLayer(object):
    """
    divisive normalisation, optimised for performance.
    """

    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.target_var = T.matrix()  # variable for the labels

        self.question_slices = [slice(0, 7)]  # , slice(1, 7)]

        # self.scaling_factor_indices = [None, [1], [1, 4], [1, 4], [1, 4], None, [0], [13], [1, 3], [1, 4, 7], [1, 4, 7]]
        # indices of all the probabilities that scale each question.

        self.normalisation_mask = theano.shared(
            self.generate_normalisation_mask())
        # self.scaling_mask = theano.shared(self.generate_scaling_mask())

        # sequence of scaling steps to be undertaken.
        # First element is a slice indicating the values to be scaled. Second element is an index indicating the scale factor.
        # these have to happen IN ORDER else it doesn't work correctly.
# rescaling irrelevant
        # self.scaling_sequence = [
        #   (slice(3, 5), 1), # I: rescale Q2 by A1.2
        #   (slice(5, 13), 4), # II: rescale Q3, Q4, Q5 by A2.2
        #   (slice(15, 18), 0), # III: rescale Q7 by A1.1
        #   (slice(18, 25), 13), # IV: rescale Q8 by A6.1
        #   (slice(25, 28), 3), # V: rescale Q9 by A2.1
        #   (slice(28, 37), 7), # VI: rescale Q10, Q11 by A4.1
       # ]

    def generate_normalisation_mask(self):
        """
        when the clipped input is multiplied by the normalisation mask, the normalisation denominators are generated.
        So then we can just divide the input by the normalisation constants (elementwise).
        """
        mask = np.zeros((7, 7), dtype='float32')
        for s in self.question_slices:
            mask[s, s] = 1.0
        return mask

    # def generate_scaling_mask(self):
    #     """
    #     This mask needs to be applied to the LOGARITHM of the probabilities. The appropriate log probs are then summed,
    #     which corresponds to multiplying the raw probabilities, which is what we want to achieve.
    #     """
    #     mask = np.zeros((37, 37), dtype='float32')
    #     for s, factor_indices in zip(self.question_slices, self.scaling_factor_indices):
    #         if factor_indices is not None:
    #             mask[factor_indices, s] = 1.0
    #     return mask

    def answer_probabilities(self, *args, **kwargs):
        """
        normalise the answer groups for each question.
        """
        input = self.input_layer.output(*args, **kwargs)
        # T.clip(input, 0, 1) # T.maximum(input, 0)
        input_clipped = T.maximum(input, 0)

        # small constant to prevent division by 0
        normalisation_denoms = T.dot(
            input_clipped, self.normalisation_mask) + 1e-12
        input_normalised = input_clipped / normalisation_denoms

        return input_normalised
        # return [input_normalised[:, s] for s in self.question_slices]

    # def weighted_answer_probabilities(self, *args, **kwargs):
    #     answer_probabilities = self.answer_probabilities(*args, **kwargs)

    #     log_scale_factors = T.dot(T.log(answer_probabilities), self.scaling_mask)
    # scale_factors = T.exp(T.switch(T.isnan(log_scale_factors), -np.inf,
    # log_scale_factors)) # need NaN shielding here because 0 * -inf = NaN.

    #     return answer_probabilities * scale_factors

    def weighted_answer_probabilities(self, *args, **kwargs):
        probs = self.answer_probabilities(*args, **kwargs)

        # go through the rescaling sequence in order (6 steps)
        # for probs_slice, scale_idx in self.scaling_sequence:
        #   probs = T.set_subtensor(probs[:, probs_slice], probs[:, probs_slice] * probs[:, scale_idx].dimshuffle(0, 'x'))

        return probs

    def predictions(self, normalisation=True, *args, **kwargs):
        return self.weighted_answer_probabilities(*args, **kwargs)

    def predictions_no_normalisation(self, *args, **kwargs):
        """
        Predict without normalisation. This can be used for the first few chunks to find good parameters.
        """
        input = self.input_layer.output(*args, **kwargs)
        # clip on both sides here, any predictions over 1.0 are going to get
        # normalised away anyway.
        input_clipped = T.clip(input, 0, 1)
        return input_clipped

    def error(self, normalisation=True, *args, **kwargs):
        if normalisation:
            predictions = self.predictions(*args, **kwargs)
        else:
            predictions = self.predictions_no_normalisation(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
        return error
