import collections
import csv
from typing import Dict, List, Tuple
from utils.file_access import CORNELL_BASE_FOLDER, file_exists, HUMOR_VALUES_FILE, MOVIE_CONVERSATIONS_FILE, MOVIE_LINES_FILE, open_data_file, STARTER_LINES_FILE

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

    Returns:
        A dictionary of words to their humor values.
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

def get_starter_lines() -> List[str]:
    """
    Gets all starting conversation lines from the Cornell movie dialogues corpus.

    Returns: A list of all starting conversation lines from the Cornell movie dialogues corpus.
    """
    if file_exists(STARTER_LINES_FILE):
        with open_data_file(STARTER_LINES_FILE) as starter_file:
            starter_lines = [line[:-1] for line in starter_file]
    else:
        with open_data_file(MOVIE_LINES_FILE, prefix=CORNELL_BASE_FOLDER) as lines:
            line_dict = {line[:line.index(' ')]: line[line.rfind('+++$+++') + 8:-1] for line in lines}

        starter_lines = []
        with open_data_file(STARTER_LINES_FILE, 'w+', CORNELL_BASE_FOLDER) as starter_file:
            with open_data_file(MOVIE_CONVERSATIONS_FILE, prefix=CORNELL_BASE_FOLDER) as conversations:
                for conversation in conversations:
                    starter_line_index = conversation[conversation.index('[') + 2 : conversation.index(',') - 1]
                    starter_line = line_dict[starter_line_index]
                    starter_file.write(starter_line + '\n')
                    starter_lines.append(starter_line)

    return starter_lines