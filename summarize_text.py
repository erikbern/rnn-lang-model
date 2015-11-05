from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random, sys
import heapq
import string
import os
import theano
from scipy.spatial.distance import cosine, euclidean

chars = string.letters + string.digits + ' .,-^'
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def create_input(sentence):
    x = np.zeros((1, len(sentence), len(chars)), dtype=theano.config.floatX)
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.
    return x

sentence = '^' + sys.argv[1]
sentence = ''.join([c for c in sentence if c in char_indices])
x = create_input(sentence)

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
first_layer = LSTM(512, return_sequences=True, input_shape=(None, len(chars)))
model.add(first_layer)
model.add(Dropout(0.5))
second_layer = LSTM(512, return_sequences=True)
model.add(second_layer)
model.add(Dropout(0.5))
model.add(TimeDistributedDense(len(chars)))
model.add(Activation('softmax'))

print('creating function')
layer_output = theano.function([model.get_input(train=False)], second_layer.get_output(train=False))

W = layer_output(x)[0]
print(W.shape)

dists = []
for i in xrange(W.shape[0]):
    for j in xrange(i+1, W.shape[0]):
        # m = (W[i] + W[j]) / 2
        # d = sum([cosine(W[k], m) for k in xrange(i, j)])
        d = euclidean(W[i], W[j])
        dists.append((d, i, j))

dists.sort()
for d, i, j in dists[:100]:
    print(sentence, i, j, d)
    p = [' '] * len(sentence)
    p[i], p[j] = '^', '^'
    print(''.join(p))
    print()
