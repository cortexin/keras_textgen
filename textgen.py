import argparse
import re
import os
import sys

import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers import (Dense, Dropout, LSTM, )
from keras.models import Sequential
from keras.utils import np_utils


# https://github.com/fchollet/keras/issues/3857
import tensorflow as tf
tf.python.control_flow_ops = tf


def parse():
    global input_file, output_file, weights_dir, mode

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='train', choices=['train', 'generate',
                        'continue'], help="Start training/Continue training/Generate output. Defaults to start training.")
    parser.add_argument("--input", help="Path to the input text file containing the training text.")
    parser.add_argument("--output", help="Path to the file into which you want the output to be saved. If no output path is provided, the output will be displayed in the shell.")
    parser.add_argument("--weights", default="./", help="Path to the directory where the weight files are to be stored. Default is the current directory")

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    weights_dir = args.weights
    mode = args.mode


def make_callbacks():
    checkpoint_path = os.path.join(weights_dir,
                                   "weights-{epoch:02d}-{loss:.4f}.hdf5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    return list(checkpoint)

def make_model():
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.15))
    model.add(Dense(Y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def get_best_weights():
    weights_pattern = re.compile(r'weights-(?P<epoch>\d+)-(?P<loss>[\d\.]+)\.hdf5')
    all_weights = list(filter(lambda fn: weights_pattern.search(fn),
                              os.listdir(weights_dir)))
    losses = {file: weights_pattern.search(file).group('loss') for file in all_weights}
    best_weights = min(losses, key=losses.get)

    return os.path.join(weights_dir, best_weights)


def load_weights_and_compile(model=None):
    if not model:
        model = make_model()
    weights = get_best_weights()
    model.load_weights(weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def train(model):
    cb_list = make_callbacks()
    model.fit(X, Y, nb_epoch=20, batch_size=128, callbacks=cb_list)


def continue_training():
    model = load_weights_and_compile()
    train(model)


def generate():
    model = load_weights_and_compile()

    if output_file:
        out = open(output_file, 'w+')
    else:
        out = sys.stdout

    # pick a random starting position
    seed = np.random.randint(0, len(dataX)-1)
    chunk = dataX[seed]

    print("Seed:  \"%s\"" % "".join([codes_to_chars[code] for code in chunk]))

    for i in range(1000):
        x = np.reshape(chunk, (1, len(chunk), 1))
        x = x / vocabulary

        prediction = model.predict(x)
        index = np.argmax(prediction)
        result = codes_to_chars[index]
        in_seq = [codes_to_chars[code] for code in chunk]

        out.write(result)

        chunk.append(index)
        chunk.pop(0)

    if output_file:
        out.close()

    print("Done\n\n\n")


if __name__ == '__main__':
    parse()
    text = open(input_file).read().lower()
    chars = sorted(list(set(text)))  # unique chars appearing in the text

    chars_to_codes = {c: i for i, c in enumerate(chars)}
    codes_to_chars = {i: c for i, c in enumerate(chars)}

    text_len = len(text)
    vocabulary = len(chars)

    print("Text length:  {} characters.".format(text_len))
    print("Vocabulary size:  {} characters.".format(vocabulary))

    chunk_size = 100
    dataX = []
    dataY = []

    for i in range(text_len-chunk_size):
        in_seq = text[i:i+chunk_size]
        out_seq = text[i+chunk_size]
        dataX.append([chars_to_codes[char] for char in in_seq])
        dataY.append(chars_to_codes[out_seq])

    chunks_number = len(dataX)
    print("Total chunks number:  {}".format(chunks_number))

    global X, Y

    X = np.reshape(dataX, (chunks_number, chunk_size, 1))
    X = X / vocabulary  # normalization
    Y = np_utils.to_categorical(dataY)

    if mode == 'generate':
        generate()
    elif mode == 'continue':
        continue_training()
    else:
        train()
