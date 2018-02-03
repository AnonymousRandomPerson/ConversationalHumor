import nltk.tokenize as tokenize
from file_access import open_data_file

def preprocess_text(file_name: str) -> list:
    """
    Does preprocessing on a file to return a tokenized list of conversations.

    Args:
        file_name: The name of the file to get text from.

    Returns:
        A tokenized list of conversations from the file.
    """
    with open_data_file(file_name) as data_file:
        conversations = data_file.read().split('\n\n')
        conversation_tokens = []
        for conversation in conversations:
            lines = conversation.split('\n')
            token_line = [preprocess_line(line) for line in lines]
            conversation_tokens.append(token_line)


    return []

def preprocess_line(line_text: str) -> list:
    """
    Preprocesses a line by converting it to tokens.

    Args:
        line_text: The text to convert to tokens.

    Returns:
        A list of tokens representing the line.
    """
    token_sents = tokenize.sent_tokenize(line_text)
    token_words = [tokenize.word_tokenize(sentence) for sentence in token_sents]
    return token_words