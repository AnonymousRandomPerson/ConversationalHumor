import tensorflow as tf

import rl.chatbots as chatbots
from rl.conversation import Conversation

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
        self.chatbot = chatbots.NormalChatbot(sess, 'Other')
        self.conversation = []

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
        chatbot_response = self.chatbot.respond(response)
        self.conversation.append(chatbot_response)
        return chatbot_response

    def evaluate_response(self, response: str) -> float:
        """
        Returns a reward for a certain response.

        Args:
            response: The response to the conversation.

        Returns: The reward for the response.
        """
        self.conversation.append(response)
        if response == 'Test':
            return 0
        return 10

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