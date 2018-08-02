import random
from typing import List, Tuple

class Conversation(object):
    """
    Base class for a conversation.
    """

    def __init__(self):
        pass

    def start_conversation(self, starter: str) -> str:
        """
        Resets the conversation.

        Args:
            starter: An optional pre-defined conversation starter.

        Returns:
            The first message in the conversation.
        """
        return self.choose_message('')

    def choose_message(self, response: str, next_message: str = '') -> str:
        """
        Chooses a message to send in the conversation.

        Args:
            response: The response to the conversation.
            next_message: A predefined next message in the conversation, if not blank.

        Returns:
            The next message in the conversation.
        """
        return next_message

    def evaluate_response(self, response: str, next_message: str) -> float:
        """
        Returns a reward for a certain response.

        Args:
            response: The response to the conversation.
            next_message: The next message that will be said in response to the first response.

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

    def respond(self, response: str, next_message: str = '') -> Tuple[str, float, bool]:
        """
        Takes a response to the previous dialogue and gives a reward and the next conversation message.

        Args:
            response: The response to the conversation.
            next_message: A predefined next message in the conversation, if not blank.

        Returns:
            next_message: The next message in the conversation.
            reward: The reward received for the response.
            end: Whether the conversation is over.
        """
        self.on_response()
        is_ended = self.is_ended()
        if not is_ended:
            next_message = self.choose_message(response, next_message)
        reward = self.evaluate_response(response, next_message)
        return next_message, reward, is_ended


class TestConversation(Conversation):
    """
    A testing conversation that rewards the agent for repeating after it.
    """

    def __init__(self, vocab: List[str]):
        """
        Initializes the conversation with a certain vocabulary.

        Args:
            vocab: The vocabulary to get words to output during the conversation.
        """
        self.max_sentences = 10
        self.vocab = vocab[:-1]

    def start_conversation(self) -> str:
        """
        Resets the conversation.

        Returns:
            The first message in the conversation.
        """
        self.num_sentences = 0
        return self.choose_message()

    def choose_message(self, response: str) -> str:
        """
        Chooses a message to send in the conversation.

        Args:
            response: The response to the conversation.

        Returns:
            The next message in the conversation.
        """
        self.current_word = random.choice(self.vocab)
        return self.current_word

    def evaluate_response(self, response: str) -> float:
        """
        Returns a reward for a certain response.

        Args:
            response: The response to the conversation.

        Returns: The reward for the response.
        """
        if response == self.current_word:
            reward = 1
        else:
            reward = 0
        return reward

    def is_ended(self) -> bool:
        """
        Checks if the conversation has ended.

        Returns: Whether the conversation has ended.
        """
        return self.num_sentences >= self.max_sentences

    def on_response(self):
        """
        Increases the sentence count when a response is received.
        """
        self.num_sentences += 1