import os
import sys
from typing import TextIO

# The root folder for code in this repository.
ROOT_FOLDER = 'humor'
# The folder where data files are stored.
DATA_FOLDER = 'data'
# The folder where saved TensorFlow models are stored.
SAVED_MODEL_FOLDER = 'tfsave'
# The folder where GloVe files are stored.
GLOVE_FOLDER = 'glove'

# The extension for pickled files.
PICKLE_EXTENSION = '.pickle'
# The extension for plain text files.
TEXT_EXTENSION = '.txt'

# The DeepQA submodule name.
CHATBOT_MODULE = 'DeepQA'

# The file containing humor values for words.
HUMOR_VALUES_FILE = os.path.join(DATA_FOLDER, 'humor_dataset.csv')

# The file containing a pretrained GloVe model.
GLOVE_FILE = os.path.join(GLOVE_FOLDER, 'glove.twitter.27B.25d.txt')

def open_data_file(file_name: str, mode: str = 'r') -> TextIO:
    """
    Opens a file for conversation data.

    Args:
        file_path: The name of the file to open.
        mode: The mode to open the file with.

    Returns:
        An opened file for conversation data.
    """
    return open(os.path.join(DATA_FOLDER, file_name), mode=mode, encoding='utf-8')

def open_binary_file(file_name: str, mode: str = 'rb'):
    """
    Opens a file for binary data.

    Args:
        file_path: The name of the file to open.
        mode: The mode to open the file with.

    Returns:
        An opened file for binary data.
    """
    return open(os.path.join(DATA_FOLDER, file_name), mode=mode)

def add_module(module_name: str):
    """
    Adds the submodule to the path so that it can be imported.

    Args:
        module_name: The name of the submodule to be added to the path.
    """
    sys.path.append(module_name)