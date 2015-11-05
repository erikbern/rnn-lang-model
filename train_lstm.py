from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys
import string
import heapq
import os

chars = string.letters + string.digits + ' .,-^'
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 31

def read_data(n=20000):
    path = sys.argv[1]
    lines = []
    for line in open(path):
        if line == '\n': continue
        heapq.heappush(lines, (random.random(), line))
        if len(lines) > n:
            heapq.heappop(lines)

    prefix = '^'
    text = ''.join([prefix + line.strip() for _, line in lines])
    all = string.maketrans('', '')
    rem = all.translate(all, chars)
    text = text.translate(None, rem)
    print('corpus length:', len(text))

    width = (len(text) - maxlen) // step
    X = np.zeros((width, maxlen, len(chars)), dtype=np.bool)
    Y = np.zeros((width, maxlen, len(chars)), dtype=np.bool)

    for j in xrange(width):
        for t in xrange(maxlen):
            X[j, t, char_indices[text[step*j+t]]] = 1
            Y[j, t, char_indices[text[step*j+t+1]]] = 1

    return X, Y

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.5))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributedDense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

if os.path.exists('model.weights'):
    print('Loading existing weights')
    model.load_weights('model.weights')

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 500):
    print('Iteration', iteration)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        sentence = '^'
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(sentence)

        for iteration in range(200):
            x = np.zeros((1, len(sentence), len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0][-1]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence += next_char
            if len(sentence) == 100:
                # Cut off to make predictions faster
                sentence = sentence[-50:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

    X, Y = read_data()
    model.fit(X, Y, batch_size=128, nb_epoch=1)
    model.save_weights('model.weights', overwrite=True)

