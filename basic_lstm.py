# https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537

import argparse
import collections
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from file_access import open_data_file, SAVED_MODEL_FOLDER

EPOCHS = 500
N_INPUT = 3
N_HIDDEN = 512
LEARNING_RATE = 0.001

GENERATE_INCREMENT = 10

model_file = 'saved_model'
corpus_name = 'twitter_test.txt'
test = True

END_TOKEN = '<end>'

def build_dataset(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds a data set of word-index mappings.

    Args:
        words: The raw list of words in the data set.

    Returns:
        dictionary: A mapping of words to their indices.
        reverse_dictionary: A mapping of indices to their words.
    """
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def create_rnn(x: tf.Tensor, weights: Dict[str, tf.Variable], biases: Dict[str, tf.Variable]) -> Tuple[rnn.BasicLSTMCell, tf.Tensor]:
    """
    Creates a basic LSTM.

    Args:
        x: A placeholder tensor for the input layer.
        weights: Weights going into the output layer.
        biases: Biases going into the output layer.

    Returns:
        rnn_cell: The basic LSTM cell that was created.
        (Tensor): A tensor for the output of the LSTM.
    """
    x = tf.reshape(x, [-1, N_INPUT])

    x = tf.split(x, N_INPUT, 1)

    rnn_cell = rnn.BasicLSTMCell(N_HIDDEN)

    outputs, _ = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    return rnn_cell, tf.matmul(outputs[-1], weights['out']) + biases['out']

def run() -> None:
    """
    Runs the LSTM.
    """
    punch_lines = ''
    sentence_length = 1
    with open_data_file(corpus_name) as corpus_file:
        line_counter = 0
        for line in corpus_file:
            if line_counter == 0:
                punch_lines += line[:-1] + ' ' + END_TOKEN + ' '
                sentence_length = max(sentence_length, len(line))
            line_counter += 1
            if line_counter > 2:
                line_counter = 0
    words = punch_lines.split(' ')

    dictionary, reverse_dictionary = build_dataset(words)

    vocab_size = len(dictionary)
    batch_size = len(words) - N_INPUT

    weights = {
        'out': tf.Variable(tf.random_normal([N_HIDDEN, vocab_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size]))
    }

    x = tf.placeholder(tf.float32, [None, N_INPUT])
    y = tf.placeholder(tf.float32, [None, vocab_size])

    rnn_cell, pred = create_rnn(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    max_accuracy = tf.Variable(0, dtype=tf.float32)

    init = tf.global_variables_initializer()

    tf.summary.scalar('accuracy', accuracy)
    summary_merge = tf.summary.merge_all()

    saver = tf.train.Saver([weights['out'], biases['out'], max_accuracy] + rnn_cell.variables, save_relative_paths=True)

    save_file_path = SAVED_MODEL_FOLDER + model_file
    summary_file = SAVED_MODEL_FOLDER + 'summary'

    with tf.Session() as sess:
        sess.run(init)
        if os.path.exists(save_file_path + '.index'):
            saver.restore(sess, save_file_path)

        _ = tf.summary.FileWriter(summary_file, sess.graph)

        def generate_sentence() -> None:
            end_key = dictionary[END_TOKEN]
            symbols_in_keys = []
            while len(symbols_in_keys) < N_INPUT:
                rand_key = random.randint(0, vocab_size - 1)
                if rand_key != end_key:
                    symbols_in_keys.append(rand_key)

            for i in range(sentence_length):
                keys = np.reshape(np.array(symbols_in_keys[i:]), [-1, N_INPUT])
                onehot_pred = sess.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                if onehot_pred_index == end_key:
                    break
                symbols_in_keys.append(onehot_pred_index)
            sentence = ''
            for symbol in symbols_in_keys:
                sentence += reverse_dictionary[symbol] + ' '
            sentence = sentence[:-1]
            print('Test sentence:', sentence + '\n')

        if test:
            generate_sentence()
        else:
            for epoch in range(EPOCHS):
                total_acc = 0
                for offset in range(batch_size):
                    symbols_in_keys = [[dictionary[str(words[i])] for i in range(offset, offset + N_INPUT)]]

                    symbols_out_onehot = np.zeros([vocab_size], dtype=float)
                    symbols_out_onehot[dictionary[str(words[offset + N_INPUT])]] = 1.0
                    symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

                    _, _, acc, _, _ = sess.run([summary_merge, optimizer, accuracy, cost, pred], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
                    total_acc += acc

                current_acc = total_acc / batch_size
                print('Current accuracy', current_acc)
                if max_accuracy.eval() < current_acc:
                    max_accuracy.assign(current_acc).op.run()
                    saver.save(sess, save_file_path)
                    print('Saved new model with accuracy', current_acc)
                    generate_sentence()
                elif epoch % GENERATE_INCREMENT == 0:
                    generate_sentence()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a basic LSTM.')
    parser.add_argument('-c', '--corpus-file', help='The name of the corpus to get data from.')
    parser.add_argument('-m', '--model-file', help='The name of the model to load and save.')
    parser.add_argument('-t', '--test', help='Test the current model.', action='store_true')
    args = parser.parse_args()

    test = args.test
    if args.model_file:
        model_file = args.model_file
        corpus_name = args.corpus_file

    run()