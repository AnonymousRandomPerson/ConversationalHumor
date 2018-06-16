import collections
import csv
from typing import Dict, List, Tuple
from utils.file_access import HUMOR_VALUES_FILE

def build_word_indices(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds a data set of word-index mappings.

    Args:
        words: The raw list of words in the data set.

    Returns:
        dictionary: A mapping of words to their indices.
        reverse_dictionary: A mapping of indices to their words.
    """
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def build_word_humor_values() -> Dict[str, int]:
    """
    Builds a dictionary of words to their humor values.
    """
    humor_dict = collections.defaultdict(float)
    weight_sum = 0
    with open(HUMOR_VALUES_FILE) as humor_file:
        humor_csv = csv.reader(humor_file)
        first = True
        for row in humor_csv:
            if first:
                first = False
                continue
            weight = float(row[1])
            humor_dict[row[0]] = weight
            weight_sum += weight

    mean_weight = weight_sum / len(humor_dict)
    for word in humor_dict:
        humor_dict[word] -= mean_weight
    return humor_dict