import tensorflow as tf

from rl.conversation import Conversation
from utils.file_access import add_module, CHATBOT_MODULE

add_module(CHATBOT_MODULE)

import chatbot_rnn.chatbot as chatbot

class EvaluatedConversation(Conversation):
    """
    A conversation that is evaluated with a non-trivial function.
    """

    def __init__(self, sess: tf.Session):
        """
        Sets up the chatbot to be used in the conversation.

        Args:
            sess: The Tensorflow session to use with the chatbot.
        """
        self.chatbot = chatbot.get_chatbot(sess, "Other")

    def start_conversation(self) -> str:
        """
        Resets the conversation.

        Returns:
            The first message in the conversation.
        """
        return self.choose_message('')

    def choose_message(self, response: str) -> str:
        """
        Chooses a message to send in the conversation.

        Args:
            response: The response to the conversation.

        Returns:
            The next message in the conversation.
        """
        return self.chatbot.respond(response)

    def evaluate_response(self, response: str) -> float:
        """
        Returns a reward for a certain response.

        Args:
            response: The response to the conversation.

        Returns: The reward for the response.
        """
        return 0

    def is_ended(self) -> bool:
        """
        Checks if the conversation has ended.

        Returns: Whether the conversation has ended.
        """
        return False

    def on_response(self):
        """
        Does pre-processing before a response is evaluated.
        """
        pass