# https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537

import argparse
import collections
import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from file_access import open_data_file, PICKLE_EXTENSION, SAVED_MODEL_FOLDER

EPOCHS = 500
N_INPUT = 3
N_HIDDEN = 512
LEARNING_RATE = 0.001

GENERATE_INCREMENT = 10

model_file = 'test_model'
corpus_name = 'twitter_test.txt'
embedding_file = None
test = True

END_TOKEN = '<e>'
START_TOKEN = '<s>'

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
    save_model_path = SAVED_MODEL_FOLDER + model_file
    save_embedding_path = SAVED_MODEL_FOLDER + embedding_file
    summary_file = SAVED_MODEL_FOLDER + 'summary'

    punch_lines = ''
    sentence_length = 1

    start_tokens = ''
    for _ in range(N_INPUT):
        start_tokens += START_TOKEN + ' '
    with open_data_file(corpus_name) as corpus_file:
        for line in corpus_file:
            if line[-1] == '\n':
                line = line[:-1]
            # Skip blank lines
            if not line:
                continue
            # Add start and end tokens around the sentence.
            punch_lines += start_tokens + line + ' ' + END_TOKEN + ' '
            sentence_length = max(sentence_length, len(line))
    words = punch_lines.split(' ')

    dictionary, reverse_dictionary = build_dataset(words)

    vocab_size = len(dictionary)
    batch_size = len(words) - N_INPUT
    num_outputs = vocab_size

    if embedding_file:
        with open(save_embedding_path + PICKLE_EXTENSION, 'rb') as f:
            embeddings = pickle.load(f)
            np_embeddings = np.array(embeddings)
        num_outputs = len(embeddings[0])

    # Define the network structure.
    weights = {
        'out': tf.Variable(tf.random_normal([N_HIDDEN, num_outputs]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_outputs]))
    }

    x = tf.placeholder(tf.float32, [None, N_INPUT])
    y = tf.placeholder(tf.float32, [None, num_outputs])

    rnn_cell, pred = create_rnn(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    max_accuracy = tf.Variable(0, dtype=tf.float32)

    init = tf.global_variables_initializer()

    tf.summary.scalar('accuracy', accuracy)
    summary_merge = tf.summary.merge_all()

    saver = tf.train.Saver([weights['out'], biases['out'], max_accuracy] + rnn_cell.variables, save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(init)
        if os.path.exists(save_model_path + '.index'):
            saver.restore(sess, save_model_path)

        _ = tf.summary.FileWriter(summary_file, sess.graph)

        def generate_sentence() -> None:
            """
            Generates a sentence using the current LSTM.
            """

            # Start with two start tokens and a random word.
            end_key = dictionary[END_TOKEN]
            start_key = dictionary[START_TOKEN]
            symbols_in_keys = [start_key for i in range(N_INPUT - 1)]
            rand_key = start_key
            while rand_key == start_key or rand_key == end_key:
                rand_key = random.randint(0, vocab_size - 1)
            symbols_in_keys.append(rand_key)

            # Create a sentence with word indices.
            for i in range(sentence_length):
                keys = np.reshape(np.array(symbols_in_keys[i:]), [-1, N_INPUT])
                pred_output = sess.run(pred, feed_dict={x: keys})
                if embedding_file:
                    pred_index = np.argmin(np.linalg.norm(np_embeddings - pred_output, axis=0)) + 1
                else:
                    pred_index = int(tf.argmax(pred_output, 1).eval())
                if pred_index == end_key:
                    break
                symbols_in_keys.append(pred_index)

            # Create the sentence by converting word indices to words.
            sentence = ''
            for symbol in symbols_in_keys:
                if symbol != start_key:
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

                    word_index = dictionary[str(words[offset + N_INPUT])]
                    if embedding_file:
                        # Minus one to account for no start token embedding.
                        symbols_out = np.asarray(embeddings[word_index - 1])
                    else:
                        symbols_out = np.zeros([num_outputs], dtype=float)
                        symbols_out[word_index] = 1.0
                    symbols_out = np.reshape(symbols_out, [1, -1])

                    _, _, acc, _, _ = sess.run([summary_merge, optimizer, accuracy, cost, pred], feed_dict={x: symbols_in_keys, y: symbols_out})
                    total_acc += acc

                current_acc = total_acc / batch_size
                print('Current accuracy', current_acc)
                if max_accuracy.eval() < current_acc:
                    max_accuracy.assign(current_acc).op.run()
                    saver.save(sess, save_model_path)
                    print('Saved new model with accuracy', current_acc)
                    generate_sentence()
                elif epoch % GENERATE_INCREMENT == 0:
                    generate_sentence()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a basic LSTM.')
    parser.add_argument('-c', '--corpus-file', help='The name of the corpus to get data from.')
    parser.add_argument('-e', '--embedding-file', help='The name of the embedding file to load.')
    parser.add_argument('-m', '--model-file', help='The name of the model to load and save.')
    parser.add_argument('-t', '--test', help='Test the current model.', action='store_true')
    args = parser.parse_args()

    test = args.test
    if args.model_file:
        model_file = args.model_file
    if args.corpus_file:
        corpus_name = args.corpus_file
    if args.embedding_file:
        embedding_file = args.embedding_file

    run()