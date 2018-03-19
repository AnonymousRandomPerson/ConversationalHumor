import json
from typing import List

from .extractor import Extractor, Message

class NUSExtractor(Extractor):
    """
    Extracts from the NUS SMS corpus.
    """

    @property
    def corpus_name(self) -> str:
        """
        The name of the corpus to extract from.
        """
        return "nus"

    def get_corpus(self) -> List[Message]:
        """
        Gets the corpus data to extract from.

        Returns:
            The corpus data to extract from.
        """
        with open('data/smsCorpus_en_2015.03.09_all.json') as corpus_file:
            corpus_text = corpus_file.read()
        corpus_json = json.loads(corpus_text)
        message_list = []
        for message in corpus_json['smsCorpus']['message']:
            message_list.append(Message(str(message['text']['$']), None))
        return message_list