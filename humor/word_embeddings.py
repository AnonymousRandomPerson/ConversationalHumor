# http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/

import argparse
import collections
import datetime as dt
import math
import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from .utils.file_access import open_data_file, PICKLE_EXTENSION, SAVED_MODEL_FOLDER

BATCH_SIZE = 128

# Random set of words to evaluate similarity on.
VALID_SIZE = 16

# Dimension of the embedding vector.
EMBEDDING_SIZE = 128

PRINT_LOSS_STEP = 100

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent

num_steps = 100000

NUM_TEST_NEIGHBORS = 8

model_file = 'test_embeddings'
final_model_path = SAVED_MODEL_FOLDER + 'final'
corpus_name = 'twitter_test.txt'
test = True
word_test = None

END_TOKEN = '<e>'
START_TOKEN = '<s>'
UNKNOWN_TOKEN = 'UNK'

class WordData(object):
    """
    Holds a data set of words to create word embeddings with.
    """

    VOCABULARY_SIZE = 10000
    # Only pick dev samples in the head of the distribution.
    VALID_WINDOW = 100

    # How many words to consider left and right.
    SKIP_WINDOW = 1
    # How many times to reuse an input to generate a context.
    NUM_SKIPS = 2
    # [ SKIP_WINDOW input_word SKIP_WINDOW ]
    SPAN = 2 * SKIP_WINDOW + 1

    # Number of negative examples to sample.
    NUM_SAMPLED = 64

    def __init__(self):
        """
        Initializes the data set.
        """
        self.collect_data()
        if END_TOKEN in self.dictionary:
            self.end_key = self.dictionary[END_TOKEN]
        else:
            self.end_key = -1
        self.VOCABULARY_SIZE = min(self.VOCABULARY_SIZE, len(self.dictionary))

        self.VALID_WINDOW = min(self.VALID_WINDOW, len(self.dictionary))
        self.valid_examples = np.random.choice(self.VALID_WINDOW, VALID_SIZE, replace=False)
        self.NUM_SAMPLED = min(self.NUM_SAMPLED, self.VALID_WINDOW)

        self.data_index = 0

    def build_dataset(self, words: List[str]):
        """
        Process raw inputs into a dataset.

        Args:
            words: A list of words from the raw input.
        """
        self.count = [(UNKNOWN_TOKEN, -1)]
        self.count.extend(collections.Counter(words).most_common(self.VOCABULARY_SIZE - 1))
        self.dictionary = dict()
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        self.data = list()
        unk_count = 0
        for word in words:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                # dictionary[UNKNOWN_TOKEN]
                index = 0
                unk_count += 1
            self.data.append(index)
        self.count[0] = (self.count[0][0], unk_count)
        self.reversed_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def collect_data(self):
        """
        Creates a data set to make a word embedding with.

        Args:
            vocabulary_size: The maximum number of words to include in the data set.
        """
        with open_data_file(corpus_name) as corpus_file:
            vocabulary = []
            for line in corpus_file:
                # Skip blank lines
                if line[-1] == '\n':
                    line = line[:-1]
                if not line:
                    continue
                vocabulary.extend(line.split(' ') + [END_TOKEN])


        self.build_dataset(vocabulary)

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate batch data.

        Returns:
            batch: The input words for the training set.
            context: The neighboring context words corresponding to the input words.
        """
        assert BATCH_SIZE % self.NUM_SKIPS == 0
        assert self.NUM_SKIPS <= 2 * self.SKIP_WINDOW
        batch = np.ndarray(shape=(BATCH_SIZE), dtype=np.int32)
        context = np.ndarray(shape=(BATCH_SIZE, 1), dtype=np.int32)

        buffer = self.fill_buffer()

        for i in range(BATCH_SIZE // self.NUM_SKIPS):
            # input word at the center of the buffer
            possible_targets = list(range(len(buffer)))
            possible_targets.remove(self.SKIP_WINDOW)
            targets = random.sample(possible_targets, self.NUM_SKIPS)
            for j, target in enumerate(targets):
                # this is the input word
                batch[i * self.NUM_SKIPS + j] = buffer[self.SKIP_WINDOW]
                # these are the context words
                context[i * self.NUM_SKIPS + j, 0] = buffer[target]

            if self.data[self.data_index] != self.end_key:
                buffer.append(self.data[self.data_index])
                self.increment_data_index()
            buffer.popleft()
            if len(buffer) <= self.NUM_SKIPS:
                buffer = self.fill_buffer()

        return batch, context

    def fill_buffer(self) -> collections.deque:
        """
        Fills a buffer with word indices.

        Returns:
            buffer: A deque of word indices in the buffer.
        """
        buffer = collections.deque()
        while len(buffer) <= self.NUM_SKIPS:
            cur_data_index = self.data_index
            buffer.clear()
            for _ in range(self.SPAN):
                if self.data[cur_data_index] == self.end_key:
                    break
                buffer.append(self.data[cur_data_index])
                cur_data_index += 1
            self.increment_data_index()
        return buffer

    def increment_data_index(self) -> None:
        """
        Increments the data index for creating training data with. Wraps around to 0 when the index exceeds the data size.
        """
        self.data_index += 1
        if self.data_index >= len(self.data):
            self.data_index = 0

class EmbeddingGraph(object):
    """
    Holds the tensor graph used for creating word embeddings.
    """

    def __init__(self, wordData: WordData):
        """
        Sets up the graph.

        Args:
            wordData: The word data that will be used when training the graph.
        """
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
            self.train_context = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
            valid_dataset = tf.constant(wordData.valid_examples, dtype=tf.int32)

            # Look up embeddings for inputs.
            self.embeddings = tf.Variable(
                tf.random_uniform([wordData.VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
            self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

            # Construct the variables for the softmax
            self.weights = tf.Variable(
                tf.truncated_normal([EMBEDDING_SIZE, wordData.VOCABULARY_SIZE],
                                    stddev=1.0 / math.sqrt(EMBEDDING_SIZE)))
            self.biases = tf.Variable(tf.zeros([wordData.VOCABULARY_SIZE]))
            hidden_out = tf.transpose(tf.matmul(tf.transpose(self.weights), tf.transpose(self.embed))) + self.biases

            # convert train_context to a one-hot format
            train_one_hot = tf.one_hot(self.train_context, wordData.VOCABULARY_SIZE)

            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hidden_out, labels=train_one_hot))

            # Construct the SGD optimizer using a learning rate of 1.0.
            self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.cross_entropy)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            self.normalized_embeddings = self.embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, valid_dataset)
            self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)

            self.max_accuracy = tf.Variable(float('inf'), dtype=tf.float32)

            # Add variable initializer.
            self.init = tf.global_variables_initializer()

def run() -> None:
    """
    Runs a training session of Word2Vec.
    """
    wordData = WordData()
    graph = EmbeddingGraph(wordData)

    #run_softmax(wordData, graph)
    run_nce(wordData, graph)

def train(wordData: WordData, graph: EmbeddingGraph) -> None:
    """
    Runs a training session of Word2Vec.

    Args:
        wordData: The word data to run the session with.
        graph: The tensor graph to run the session with.
    """
    save_file_path = SAVED_MODEL_FOLDER + model_file
    saver = tf.train.Saver([graph.weights, graph.biases, graph.embeddings, graph.max_accuracy], save_relative_paths=True)

    def test_model() -> None:
        """
        Tests the current model by printing out words that are similar to the most common words in the corpus.
        """
        sim = graph.similarity.eval()
        for i in range(VALID_SIZE):
            valid_word = wordData.reversed_dictionary[wordData.valid_examples[i]]
            if word_test:
                valid_word = word_test
            # number of nearest neighbors
            top_k = NUM_TEST_NEIGHBORS
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = wordData.reversed_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
            if word_test:
                break

    with tf.Session(graph=graph.graph) as session:
        # We must initialize all variables before we use them.
        graph.init.run()

        if os.path.exists(save_file_path + '.index'):
            print("Restored")
            saver.restore(session, save_file_path)

        if test:
            test_model()
        else:

            average_loss = 0

            for step in range(num_steps):
                batch_inputs, batch_context = wordData.generate_batch()
                feed_dict = {graph.train_inputs: batch_inputs, graph.train_context: batch_context}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([graph.optimizer, graph.cross_entropy], feed_dict=feed_dict)
                average_loss += loss_val

                last_step = step == num_steps - 1

                if step % 100 == 0:
                    print('Step', step)

                if step % PRINT_LOSS_STEP == 0 or last_step:
                    if step > 0:
                        average_loss /= PRINT_LOSS_STEP
                        if graph.max_accuracy.eval() > average_loss:
                            graph.max_accuracy.assign(average_loss).op.run()
                            saver.save(session, save_file_path)
                            print('Saved new model with loss', average_loss)
                            test_model()

                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step', str(step) + ':', average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0 or last_step:
                    graph.max_accuracy.assign(average_loss).op.run()
                    saver.save(session, final_model_path)
                    test_model()
        final_embeddings = graph.normalized_embeddings.eval()
        embedding_dict = {}
        for i, embedding in enumerate(final_embeddings):
            embedding_dict[wordData.reversed_dictionary[i]] = embedding
        with open(final_model_path + PICKLE_EXTENSION, 'wb') as f:
            pickle.dump(embedding_dict, f)

def run_softmax(wordData: WordData, graph: EmbeddingGraph) -> None:
    """
    Runs the softmax Word2Vec training method.

    Args:
        wordData: The corpus of words to run NCE on.
        graph: The tensor graph to run NCE with.
    """
    softmax_start_time = dt.datetime.now()
    train(wordData, graph)
    softmax_end_time = dt.datetime.now()
    print("Softmax method took {} seconds to run {} iterations".format((softmax_end_time-softmax_start_time).total_seconds(), num_steps))

def run_nce(wordData: WordData, graph: EmbeddingGraph) -> None:
    """
    Runs the noise contrastive estimation (NCE) training method.

    Args:
        wordData: The corpus of words to run NCE on.
        graph: The tensor graph to run NCE with.
    """
    with graph.graph.as_default():

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([wordData.VOCABULARY_SIZE, EMBEDDING_SIZE],
                                stddev=1.0 / math.sqrt(EMBEDDING_SIZE)))
        nce_biases = tf.Variable(tf.zeros([wordData.VOCABULARY_SIZE]))

        nce_loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=graph.train_context,
                           inputs=graph.embed,
                           num_sampled=wordData.NUM_SAMPLED,
                           num_classes=wordData.VOCABULARY_SIZE))

        _ = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

        # Add variable initializer.
        _ = tf.global_variables_initializer()

    nce_start_time = dt.datetime.now()
    train(wordData, graph)
    nce_end_time = dt.datetime.now()
    print("NCE method took {} seconds to run {} iterations".format((nce_end_time-nce_start_time).total_seconds(), num_steps))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a model for word embeddings.')
    parser.add_argument('-c', '--corpus-file', help='The name of the corpus to get data from.')
    parser.add_argument('-m', '--model-file', help='The name of the model to load and save.')
    parser.add_argument('-n', '--num-steps', type=int, help='The number of batches to train for')
    parser.add_argument('-t', '--test', help='Test the current model.', action='store_true')
    parser.add_argument('-w', '--word-test', help='A word to test the current model with')
    args = parser.parse_args()

    test = args.test
    if args.corpus_file:
        corpus_name = args.corpus_file
    if args.model_file:
        model_file = args.model_file
    if args.num_steps:
        num_steps = args.num_steps
    if args.word_test:
        word_test = args.word_test

    run()