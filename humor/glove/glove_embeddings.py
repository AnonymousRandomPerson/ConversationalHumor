# https://nlp.stanford.edu/projects/glove/

from collections import defaultdict
from typing import Tuple

class GloveEmbedding(object):
    """
    Contains a pretrained word embedding using GloVe.
    """

    def __init__(self, pretrained_file_path: str):
        """
        Loads the pretrained word embeddings.

        Args:
            pretrained_file_path: The relative file path of the file containing the word embeddings.
        """
        size_limit = 100000

        first = True
        self.word_embeddings = defaultdict(self.get_default_embedding)
        size = 0
        with open(pretrained_file_path) as pretrained_file:
            for line in pretrained_file:
                line_split = line.split(' ')
                word = line_split[0]
                embedding = line_split[1:]
                self.word_embeddings[word] = tuple([float(weight) for weight in embedding])
                if first:
                    self.dimensions = len(embedding)
                    self.default_embedding = tuple([0] * self.dimensions)
                size += 1
                if size > size_limit:
                    break

    def get_default_embedding(self) -> Tuple[int]:
        """
        Gets a default word embedding to use if a word is not known.
        """
        return self.default_embedding