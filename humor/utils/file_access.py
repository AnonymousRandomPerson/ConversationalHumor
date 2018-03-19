import os
from typing import TextIO

ROOT_FOLDER = 'humor'
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
SAVED_MODEL_FOLDER = os.path.join(ROOT_FOLDER, 'tfsave')
PICKLE_EXTENSION = '.pickle'

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