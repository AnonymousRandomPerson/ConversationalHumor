# Wrapper classes around different chatbots.

import nltk
import tensorflow as tf
from utils.file_access import add_module, CHATBOT_MODULE

add_module(CHATBOT_MODULE)

import DeepQA.chatbot.chatbot as chatbot

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

    def __init__(self, chatbotObject: chatbot.Chatbot, name: str):
        """
        Sets up the normal chatbot by loading the pretrained model.

        Args:
            chatbotObject: The chatbot object to create responses with.
            name: The display name of the chatbot when printing responses.
        """
        ChatbotWrapper.__init__(self, name)
        self.chatbot = chatbotObject

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

class HumorProbChatbot(ChatbotWrapper):
    """
    A chatbot that increases the probability of humorous words being selected during beam search.
    """

    def __init__(self, chatbotObject: chatbot.Chatbot, name: str):
        """
        Sets up the chatbot by loading the pretrained model.

        Args:
            chatbotObject: The chatbot object to create responses with.
            name: The display name of the chatbot when printing responses.
        """
        ChatbotWrapper.__init__(self, name)
        self.chatbot = chatbotObject

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

        self.chatbot.humorProb = True
        response = self.chatbot.respond(user_input, print_response)
        self.chatbot.humorProb = False

        return response