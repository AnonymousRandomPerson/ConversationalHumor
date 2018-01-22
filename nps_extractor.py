import re
from nltk.corpus import nps_chat as nps
from extractor import Extractor, Message

username = 'User'

class NPSExtractor(Extractor):
    """
    Extracts from the NPS Chat corpus.
    """

    @property
    def corpus_name(self):
        """
        The name of the corpus to extract from.
        """
        return "nps"

    @property
    def skip_words(self):
        """
        Server-side messages that should be skipped.
        """
        return {'PART', 'JOIN', 'NICK :' + username}

    def get_corpus(self) -> list:
        """
        Gets the corpus data to extract from.

        Returns:
            The corpus data to extract from.
        """
        message_list = []
        for post in nps.xml_posts():
            message_list.append(Message(post.text, post.attrib['user']))
        return message_list

    def post_process_text(self, text: str) -> str:
        """
        Processes a raw message to convert it to a usable format.

        Args:
            text: The text to process.

        Returns:
            A cleaned version of the given text.
        """
        text = Extractor.post_process_text(self, text)

        # Replace all unique usernames and user actions with 'User'.
        text = re.sub('([0-3]\d-\d{2}-.*User\d+)|(.ACTION)', username, text)
        return text