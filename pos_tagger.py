from pickle import dump, load
import nltk
from file_access import open_binary_file

bigram_file = 'pos_tagger.pkl'

class POSTagger:

    def __init__(self):
        """
        Initializes the tagger.
        """
        with open_binary_file(bigram_file) as tag_file:
            self.bigram_tagger = load(tag_file)

    def tag_words(self, words: list) -> list:
        """
        Tags words in a sentence with their parts of speech.

        Args:
            words: The words to tag.

        Returns:
            A list of tagged words.
        """
        tag_list = self.bigram_tagger.tag(words)
        alt_tag_list = nltk.pos_tag(words)

        for i, tag in enumerate(tag_list):
            if not tag[1]:
                tag_list[i] = alt_tag_list[i]
        return tag_list


def save_tagger():
    """
    Saves a tagger as a Pickle file.
    """
    nps_tags = nltk.corpus.nps_chat.tagged_posts()
    unigram_nps = nltk.UnigramTagger(nps_tags)
    bigram_nps = nltk.BigramTagger(nps_tags, backoff=unigram_nps)

    brown_tags = nltk.corpus.brown.tagged_sents()
    unigram_brown = nltk.UnigramTagger(brown_tags, backoff=bigram_nps)
    bigram_brown = nltk.BigramTagger(brown_tags, backoff=unigram_brown)

    with open_binary_file(bigram_file, 'wb') as output:
        dump(bigram_brown, output, -1)