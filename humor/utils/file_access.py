import os
import sys
from typing import TextIO

# The root folder for code in this repository.
ROOT_FOLDER = 'humor'
# The folder where data files are stored.
DATA_FOLDER = 'data'
# The folder where configuration files are stored.
CONFIG_FOLDER = 'config'
# The folder where log files are stored.
LOG_FOLDER = 'logs'
# The folder where saved TensorFlow models are stored.
SAVED_MODEL_FOLDER = 'tfsave'

# The extension for pickled files.
PICKLE_EXTENSION = '.pickle'

# The seq2seq submodule name.
SEQ2SEQ_MODULE = 'seq2seq_sub'
# The subword-nmt submodule name.
SUBWORD_MODULE = 'subword_nmt'

# The output file for seq2seq test output.
TEST_OUTPUT_FILE = os.path.join(LOG_FOLDER, 'test_output.txt')

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