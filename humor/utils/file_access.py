import os
import sys
from typing import TextIO

# The root folder for code in this repository.
ROOT_FOLDER = 'humor'
# The folder where data files are stored.
DATA_FOLDER = 'data'
# The folder where saved TensorFlow models are stored.
SAVED_MODEL_FOLDER = 'tfsave'

# The extension for pickled files.
PICKLE_EXTENSION = '.pickle'
# The extension for plain text files.
TEXT_EXTENSION = '.txt'

# The DeepQA submodule name.
CHATBOT_MODULE = 'DeepQA'

# The file containing humor values for words.
HUMOR_VALUES_FILE = os.path.join(DATA_FOLDER, 'humor_dataset.csv')

# The folder containing Cornell movie dialogue files.
CORNELL_BASE_FOLDER = os.path.join(CHATBOT_MODULE, 'data', 'cornell')
# The file containing Cornell movie conversation indices.
MOVIE_CONVERSATIONS_FILE = 'movie_conversations.txt'
# The file containing raw Cornell movie lines.
MOVIE_LINES_FILE = 'movie_lines.txt'
# The file containing conversation starter lines from Cornell movies.
STARTER_LINES_FILE = 'starter_lines.txt'

def file_exists(file_name: str, prefix: str = DATA_FOLDER) -> bool:
    """
    Checks if a file exists on the file system.

    Args:
        file_path: The name of the file to check.
        prefix: A prefix to use for the file path.

    Returns:
        Whether the file exists on the file system.
    """
    return os.path.exists(os.path.join(prefix, file_name))

def open_data_file(file_name: str, mode: str = 'r', prefix: str = DATA_FOLDER) -> TextIO:
    """
    Opens a file for conversation data.

    Args:
        file_path: The name of the file to open.
        mode: The mode to open the file with.
        prefix: A prefix to use for the file path.

    Returns:
        An opened file for conversation data.
    """
    return open(os.path.join(prefix, file_name), mode=mode, encoding='utf-8', errors='ignore')

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