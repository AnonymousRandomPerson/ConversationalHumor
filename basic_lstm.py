# https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537

import argparse
import collections
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from file_access import open_data_file

epochs = 50
n_input = 3
n_hidden = 512
learning_rate = 0.001
test = True
story_length = 32

save_folder = './tfsave/'
save_file_name = 'saved_model'

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
    x = tf.reshape(x, [-1, n_input])

    x = tf.split(x, n_input, 1)

    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    outputs, _ = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    return rnn_cell, tf.matmul(outputs[-1], weights['out']) + biases['out']

def run() -> None:
    """
    Runs the LSTM.
    """
    with open_data_file('twitter_test.txt') as corpus_file:
        corpus = corpus_file.read()
    words = corpus.split(' ')

    dictionary, reverse_dictionary = build_dataset(words)

    vocab_size = len(dictionary)
    batch_size = len(words) - n_input

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size]))
    }

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, vocab_size])

    rnn_cell, pred = create_rnn(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    max_accuracy = tf.Variable(0, dtype=tf.float32)

    init = tf.global_variables_initializer()

    tf.summary.scalar('accuracy', accuracy)
    summary_merge = tf.summary.merge_all()

    saver = tf.train.Saver([weights['out'], biases['out'], max_accuracy] + rnn_cell.variables, save_relative_paths=True)

    save_file_path = save_folder + save_file_name
    summary_file = save_folder + 'summary'

    with tf.Session() as sess:
        sess.run(init)
        if os.path.exists(save_file_path + '.index'):
            saver.restore(sess, save_file_path)

        _ = tf.summary.FileWriter(summary_file, sess.graph)

        if test:
            symbols_in_keys = [random.randint(0, vocab_size - 1) for i in range(n_input)]
            for i in range(story_length):
                keys = np.reshape(np.array(symbols_in_keys[i:]), [-1, n_input])
                onehot_pred = sess.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                symbols_in_keys.append(onehot_pred_index)
            sentence = ''
            for symbol in symbols_in_keys:
                sentence += reverse_dictionary[symbol] + ' '
            sentence = sentence[:-1]
            print(sentence)
        else:
            for _ in range(epochs):
                total_acc = 0
                for offset in range(batch_size):
                    symbols_in_keys = [[dictionary[str(words[i])] for i in range(offset, offset + n_input)]]

                    symbols_out_onehot = np.zeros([vocab_size], dtype=float)
                    symbols_out_onehot[dictionary[str(words[offset + n_input])]] = 1.0
                    symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

                    _, _, acc, _, onehot_pred = sess.run([summary_merge, optimizer, accuracy, cost, pred], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
                    total_acc += acc

                current_acc = total_acc / batch_size
                print('Current accuracy', current_acc)
                if max_accuracy.eval() < current_acc:
                    max_accuracy.assign(current_acc).op.run()
                    saver.save(sess, save_file_path)
                    print('Saved new model with accuracy', current_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a basic LSTM.')
    parser.add_argument('-t', '--test', help='Test the current model.', action='store_true')
    parser.add_argument('-m', '--model-file', help='The name of the model to load and save.')
    args = parser.parse_args()

    test = args.test
    if args.model_file:
        save_file_name = args.model_file

    run()