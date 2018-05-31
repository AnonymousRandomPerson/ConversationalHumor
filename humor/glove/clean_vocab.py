import os

from utils.file_access import GLOVE_FOLDER, GLOVE_FILE

def in_range(char: str, lower: str, upper: str) -> bool:
    """
    Checks if a character is within a range of characters.

    Args:
        char: The character to check.
        lower: The lower bound of the range (inclusive).
        upper: The upper bound of the range (inclusive).

    Returns:
        Whether the character is within range of the bound.
    """
    return char >= lower and char <= upper

def is_non_english(char: str) -> bool:
    """
    Checks if a character is from a non-English language.

    Args:
        char: The character to check:

    Returns:
        Whether the character is from a non-English language.
    """
    if in_range(char, u'\u0600', u'\u06ff') or in_range(char, u'\ufb50', u'\ufdff') or in_range(char, u'\ufe70', u'\ufeff'):
        # Arabic
        return True
    if in_range(char, u'\u0530', u'\u058f'):
        # Armenian
        return True
    if in_range(char, u'\u4e00', u'\u9fff'):
        # Chinese
        return True
    if in_range(char, u'\u0400', u'\u04ff'):
        # Cyrillic
        return True
    if in_range(char, u'\u0370', u'\u03ff'):
        # Greek
        return True
    if in_range(char, u'\u0590', u'\u05ff'):
        # Hebrew
        return True
    if in_range(char, u'\u0900', u'\u097f'):
        # Hindi
        return True
    if in_range(char, u'\uf900', u'\uffef') or in_range(char, u'\u30a0', u'\u30ff') or in_range(char, u'\u3040', u'\u309f'):
        # Japanese
        return True
    if in_range(char, u'\uac00', u'\ud7a3') or in_range(char, u'\u3130', u'\u318f'):
        # Korean
        return True
    if in_range(char, u'\u0b80', u'\u0bff'):
        # Tamil
        return True
    if in_range(char, u'\u0e00', u'\u0e7f'):
        # Thai
        return True
    return False

with open(GLOVE_FILE) as pretrained_file:
    with open(os.path.join(GLOVE_FOLDER, 'cleaned.txt'), 'w+') as target_file:
        for line in pretrained_file:
            first_char = line[0]
            last_char = line[line.index(' ') - 1]
            skip = False
            for char in (first_char, last_char):
                if is_non_english(char):
                    skip = True
                    break
            if skip:
                continue

            target_file.write(line)