# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import argparse
import numpy as np
import os
import random
import tensorflow as tf

from rl.test_dialogue import TestConversation
from utils.file_access import DATA_FOLDER,SAVED_MODEL_FOLDER
from utils.word_manipulation import build_word_indices

def run(args: argparse.Namespace) -> None:
    """
    Runs the q-learner.

    Args:
        args: Command-line arguments to use.
    """
    save_model_path = os.path.join(SAVED_MODEL_FOLDER, args.model_file)

    vocab_path = os.path.join(DATA_FOLDER, args.vocab_file)

    with open(vocab_path) as vocab_file:
        vocab = [word.rstrip('\n') for word in vocab_file.readlines()]

    env = TestConversation(vocab)

    dictionary, reverse_dictionary = build_word_indices(vocab)

    max_init_value = 0.01
    num_inputs = len(vocab)
    num_outputs = len(vocab)
    identity_mat = np.identity(num_inputs)

    #These lines establish the feed-forward part of the network used to choose actions
    inputs1 = tf.placeholder(shape=[1, num_inputs],dtype=tf.float32)
    weights = tf.Variable(tf.random_uniform([num_inputs, num_outputs], 0, max_init_value))
    biases = tf.Variable(tf.random_uniform([num_inputs], 0, max_init_value))
    q_out =  tf.nn.relu(tf.add(tf.matmul(inputs1, weights), biases))
    predict = tf.argmax(q_out,1)

    #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    next_q = tf.placeholder(shape=[1, num_outputs],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(next_q - q_out))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    update_model = trainer.minimize(loss)

    init = tf.initialize_all_variables()

    # Set learning parameters
    y = .99
    e = 0.1
    num_episodes = 50
    num_test = 100

    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver([weights, biases], save_relative_paths=True)
        if os.path.exists(save_model_path + '.index'):
            saver.restore(sess, save_model_path)

        def run_episodes(num_episodes: int, is_test: bool):
            """
            Runs a training or testing session.

            Args:
                num_episodes: The number of training/testing episodes to run in the session.
                is_test: Whether the session is a testing session.

            Returns:
                j_list: A list of the durations of each session (number of actions).
                r_list: A list of total rewards for each session.
            """
            current_epsilon = e
            j_list = []
            r_list = []
            for i in range(num_episodes):
                #Reset environment and get first new observation
                s = dictionary[env.start_conversation()]
                r_all = 0
                d = False
                j = 0
                #The Q-Network
                while j < 99:
                    j += 1
                    #Choose an action by greedily (with e chance of random action) from the Q-network
                    a, all_q = sess.run([predict, q_out],feed_dict={inputs1:identity_mat[s:s+1]})
                    if not is_test and np.random.rand(1) < current_epsilon:
                        a[0] = np.random.randint(0, len(vocab))
                    #Get new state and reward from environment
                    response = reverse_dictionary[a[0]]
                    s1, r, d = env.respond(response)
                    s1 = dictionary[s1]

                    if is_test:
                        if not i:
                            print(reverse_dictionary[s], response)
                    else:
                        #Obtain the Q' values by feeding the new state through our network
                        q1 = sess.run(q_out,feed_dict={inputs1:identity_mat[s1:s1+1]})
                        #Obtain maxQ' and set our target value for chosen action.
                        max_q1 = np.max(q1)
                        target_q = all_q
                        target_q[0, a[0]] = r + y * max_q1
                        #Train our network using target and predicted Q values
                        _, _ = sess.run([update_model, weights],feed_dict={inputs1:identity_mat[s:s+1], next_q:target_q})
                    r_all += r
                    s = s1
                    if d:
                        if not is_test:
                            #Reduce chance of random action as we train the model.
                            current_epsilon = 1./((i/50) + 10)
                        break
                j_list.append(j)
                r_list.append(r_all)
            return j_list, r_list

        if not args.test:
            _, r_list = run_episodes(num_episodes, False)
            print("Average episode reward (training): " + str(sum(r_list)/num_episodes))

            saver.save(sess, save_model_path)

        _, r_list = run_episodes(num_test, True)
        print("Average episode reward (testing): " + str(sum(r_list)/num_test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a reinforcement learner.')

    parser.add_argument('-m', '--model-file', required=True, help='The name of the model to load and save.')
    parser.add_argument('-t', '--test', default=False, help='Test the current model.', action='store_true')
    parser.add_argument('-v', '--vocab-file', required=True, help='The name of the model to load and save.')
    args = parser.parse_args()

    run(args)