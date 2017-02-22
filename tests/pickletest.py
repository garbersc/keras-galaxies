from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from loregTut import LogisticRegression, load_data

class ClassA(object):
    def __init__(self, i):
        self.i = i

class ClassB(object):
    def __init__(self, i,j,thatsK):
        self.j = j
	self.ClassA = thatsK
	self.i=i
	

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors

        self.input = input

def test():
	testClass1=ClassA(3)
	testClass2=ClassB(1,2,testClass1)
	print(testClass2.ClassA.i)
        with open('ClassB.pkl', 'w') as f:
            pickle.dump(testClass2, f)

	MLPtest=MLP(rng=numpy.random.RandomState(1234),input=numpy.ndarray([2,3]),n_in=2, n_hidden=3, n_out=4)
        with open('MLPtest.pkl', 'w') as f:
            pickle.dump(MLPtest, f)

if __name__ == '__main__':
    test()
