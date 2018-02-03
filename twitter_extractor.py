import os
from extractor import Extractor, Message
from file_access import open_data_file

encoding = 'utf-8'

class TwitterExtractor(Extractor):
    """
    Extracts from a Twitter message corpus.
    """

    def __init__(self):
        """
        Initializes the message counter to only write connected conversations.
        """
        Extractor.__init__(self)
        self.message_counter = 0

    @property
    def corpus_name(self) -> str:
        """
        The name of the corpus to extract from.
        """
        return "twitter"

    @property
    def conversation_length(self) -> int:
        """
        The maximum number of messages in a conversation.
        """
        return 2

    def get_corpus(self) -> list:
        """
        Gets the corpus data to extract from.

        Returns:
            The corpus data to extract from.
        """
        twitter_text = []
        for file_name in ('twitter_en.txt', 'twitter_en_big.txt'):
            with open_data_file(file_name) as corpus_file:
                twitter_text += corpus_file.readlines()
        message_list = []
        for message in twitter_text:
            message_list.append(Message(message[:-1], None))
        return message_list

    def allow_write(self) -> bool:
        """
        Checks if the current state of the extractor allows a conversation to be written.
        """
        self.message_counter += 1
        return self.message_counter & 1 == 0