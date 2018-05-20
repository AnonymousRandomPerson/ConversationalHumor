import random
from typing import List, Tuple

class TestConversation(object):
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
        self.vocab = vocab

    def start_conversation(self):
        """
        Resets the conversation.

        Returns:
            The first message in the conversation.
        """
        self.num_sentences = 0
        return self.choose_message()

    def choose_message(self) -> str:
        """
        Chooses a message to send in the conversation.

        Returns:
            The next message in the conversation.
        """
        self.current_word = random.choice(self.vocab)
        return self.current_word

    def respond(self, response: str) -> Tuple[str, float, bool]:
        """
        Takes a response to the previous dialogue and gives a reward and the next conversation message.

        Args:
            response: The response to the conversation.

        Returns:
            next_message: The next message in the conversation.
            reward: The reward received for the response.
            end: Whether the conversation is over.
        """
        if response == self.current_word:
            reward = 1
        else:
            reward = 0
        self.num_sentences += 1

        return self.choose_message(), reward, self.num_sentences >= self.max_sentences
