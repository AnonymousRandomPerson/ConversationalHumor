# Wrapper classes around different chatbots.

import nltk
import tensorflow as tf
from utils.file_access import add_module, CHATBOT_MODULE

add_module(CHATBOT_MODULE)

import chatbot_rnn.chatbot as chatbot

class ChatbotWrapper(object):
    """
    Base class for a chatbot.
    """
    def __init__(self, name: str):
        """
        Initializes the chatbot.

        Args:
            name: The display name of the chatbot when printing responses.
        """
        self.name = name

    def respond(self, user_input: str, print_response: bool=True):
        """
        Responds to a message from the user.

        Args:
            user_input: The message sent by the user.
            print_response: Whether the response will be printed to the console.

        Returns:
            The chatbot's response to the user message.
        """
        if print_response:
            print(self.name + ': ', end='')
        return ''


class TestChatbot(ChatbotWrapper):
    """
    A testing chatbot that only outputs 'Test'.
    """

    def __init__(self):
        """
        Sets up the test chatbot.
        """
        ChatbotWrapper.__init__(self, 'Test')
        self.response = 'Test'

    def respond(self, user_input: str, print_response: bool=True):
        """
        Responds to a message from the user.

        Args:
            user_input: The message sent by the user.
            print_response: Whether the response will be printed to the console.

        Returns:
            The chatbot's response to the user message.
        """
        ChatbotWrapper.respond(self, user_input, print_response)
        if print_response:
            print(self.response)
        return self.response

class NormalChatbot(ChatbotWrapper):
    """
    A pretrained chatbot with no additional humor.
    """

    def __init__(self, sess: tf.Session, name: str):
        """
        Sets up the normal chatbot by loading the pretrained model.

        Args:
            sess: The TensorFlow session to use with the chatbot.
            name: The display name of the chatbot when printing responses.
        """
        ChatbotWrapper.__init__(self, name)
        self.chatbot = chatbot.get_chatbot(sess)

    def respond(self, user_input: str, print_response: bool=True):
        """
        Responds to a message from the user.

        Args:
            user_input: The message sent by the user.
            print_response: Whether the response will be printed to the console.

        Returns:
            The chatbot's response to the user message.
        """
        ChatbotWrapper.respond(self, user_input, print_response)
        return self.chatbot.respond(user_input, print_response)

class ReplaceChatbot(ChatbotWrapper):
    """
    A chatbot that replaces words in responses with more "funny" words before responding to them.
    """

    def __init__(self, sess: tf.Session, name: str):
        """
        Sets up the chatbot by loading the pretrained model.

        Args:
            sess: The TensorFlow session to use with the chatbot.
            name: The display name of the chatbot when printing responses.
        """
        ChatbotWrapper.__init__(self, name)
        self.chatbot = chatbot.get_chatbot(sess)

    def respond(self, user_input: str, print_response: bool=True):
        """
        Responds to a message from the user.

        Args:
            user_input: The message sent by the user.
            print_response: Whether the response will be printed to the console.

        Returns:
            The chatbot's response to the user message.
        """
        ChatbotWrapper.respond(self, user_input, print_response)

        response = self.chatbot.respond(user_input, print_response)

        words = nltk.word_tokenize(response)

        if len(words) > 1:
            response = response.replace(words[1], 'banana')
            print(response)

        return response