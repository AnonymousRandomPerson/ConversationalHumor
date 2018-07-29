# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import argparse
import os
from typing import List

import numpy as np
import tensorflow as tf

import rl.chatbots as chatbots
from rl.evaluated_conversation import EvaluatedConversation
from utils.file_access import  add_module, SAVED_MODEL_FOLDER, CHATBOT_MODULE

add_module(CHATBOT_MODULE)

import DeepQA.chatbot.chatbot as chatbot

def run() -> None:
    """
    Runs the q-learner.
    """
    save_model_path = os.path.join(SAVED_MODEL_FOLDER, args.model_file)

    embedding_size = 64
    max_sentence_length = 10

    max_init_value = 0.01
    num_inputs = embedding_size * max_sentence_length
    hidden_size = 320

    with tf.Session() as sess:
        chatbot_object = chatbot.get_chatbot(sess)
        env = EvaluatedConversation(chatbot_object)
        normal_chatbot = chatbots.NormalChatbot(chatbot_object, 'Normal')
        prob_chatbot = chatbots.HumorProbChatbot(chatbot_object, 'HumorProb')

        responders = [normal_chatbot, prob_chatbot]

        num_outputs = len(responders)

        #These lines establish the feed-forward part of the network used to choose actions
        inputs1 = tf.placeholder(shape=[1, num_inputs], dtype=tf.float32, name='inputs1')
        input_weights = tf.Variable(tf.random_uniform([num_inputs, hidden_size], 0, max_init_value), name='input_weights')
        input_biases = tf.Variable(tf.random_uniform([hidden_size], 0, max_init_value), name='input_biases')

        hidden_weights = tf.Variable(tf.random_uniform([hidden_size, num_outputs], 0, max_init_value), name='hidden_weights')
        hidden_biases = tf.Variable(tf.random_uniform([num_outputs], 0, max_init_value), name='hidden_biases')
        hidden_out = tf.nn.sigmoid(tf.add(tf.matmul(inputs1, input_weights), input_biases), name='hidden_out')

        q_out = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out, hidden_weights), hidden_biases),name='q_out')
        predict = tf.argmax(q_out, 1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        next_q = tf.placeholder(shape=[1, num_outputs], dtype=tf.float32, name='next_q')
        loss = tf.reduce_sum(tf.square(next_q - q_out))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        update_model = trainer.minimize(loss)

        init = tf.variables_initializer([input_weights, input_biases, hidden_weights, hidden_biases])

        # Set learning parameters
        y = .99
        e = 0.5
        num_episodes = 50000
        num_test = 100
        max_steps = 20

        sess.run(init)

        saver = tf.train.Saver([input_weights, input_biases, hidden_weights, hidden_biases], save_relative_paths=True)
        if os.path.exists(save_model_path + '.index'):
            saver.restore(sess, save_model_path)
            print("Model restored.")

        def get_word_embeddings(sentence: List[str]) -> List[float]:
            """
            Gets a list of word embeddings for a sentence's words.

            Args:
                sentence: The sentence to get the embeddings for.
            Returns:
                The word embeddings for the sentence, with the unknown token embedding for unrecognized words.
            """
            return [normal_chatbot.chatbot.embeddings[env.get_word_index(word)] for word in sentence.split(' ')]

        def get_embedding_array(sentence: List[str]) -> np.ndarray:
            """
            Gets a 1D array representing word embeddings in a sentence.

            Args:
                sentence: The sentence to get the embeddings for.
            Returns:
                A 1D array representing word embeddings in a sentence.
            """
            s = get_word_embeddings(sentence)
            while len(s) < max_sentence_length:
                s.append(np.zeros(s[0].shape))
            if len(s) > max_sentence_length:
                s = s[:max_sentence_length]
            s = np.array([np.array(s).flatten()])
            return s

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

            for _ in range(num_episodes):
                print("\nStarting conversation.")
                last_sentence = env.start_conversation()
                s = get_embedding_array(last_sentence)
                r_all = 0
                j = 0
                end = False
                #The Q-Network
                while j < max_steps and not end:
                    j += 1
                    #Choose an action greedily (with e chance of random action) from the Q-network
                    a, all_q = sess.run([predict, q_out], feed_dict={inputs1:s})
                    print("All q-values:", all_q)
                    a = a[0]
                    if not is_test and np.random.rand(1) < current_epsilon:
                        a = np.random.randint(0, num_outputs)
                    #Get new state and reward from environment
                    response = responders[a].respond(last_sentence)

                    last_sentence, r, end = env.respond(response)

                    s1 = get_embedding_array(last_sentence)
                    s1 = np.array([s1[0]])

                    if not is_test:
                        #Obtain the Q' values by feeding the new state through our network
                        q1 = sess.run(q_out, feed_dict={inputs1:s1})
                        #Obtain maxQ' and set our target value for chosen action.
                        max_q1 = np.max(q1)
                        target_q = all_q
                        target_q[0, a] = r + y * max_q1
                        #Train our network using target and predicted Q values
                        _, _ = sess.run([update_model, input_weights], feed_dict={inputs1:s, next_q:target_q})
                    r_all += r
                    s = s1
                j_list.append(j)
                r_list.append(r_all)
            return j_list, r_list

        try:
            if not args.rl_test:
                _, r_list = run_episodes(num_episodes, False)
                print("Average episode reward (training): " + str(sum(r_list)/num_episodes))

            _, r_list = run_episodes(num_test, True)
            print("Average episode reward (testing): " + str(sum(r_list)/num_test))
        except KeyboardInterrupt:
            if not args.rl_test:
                saver.save(sess, save_model_path)
                print("Model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a reinforcement learner.')

    parser.add_argument('-f', '--model-file', required=True, help='The name of the model to load and save.')
    parser.add_argument('-t', '--rl-test', help='Test the current model.', action='store_true')
    args, _ = parser.parse_known_args()

    run()