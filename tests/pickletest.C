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
