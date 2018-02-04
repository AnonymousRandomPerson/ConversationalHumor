from collections import namedtuple
from nltk.corpus import stopwords
import nltk.tokenize as tokenize
from file_access import open_data_file
from pos_tagger import POSTagger

pos_tagger = POSTagger()
stopwords = set(stopwords.words('english'))

TaggedLine = namedtuple('TaggedLine', ['line', 'tagged_line', 'filtered_line'])

def preprocess_text(file_name: str, limit: int = None) -> list:
    """
    Does preprocessing on a file to return a tokenized list of conversations.

    Args:
        file_name: The name of the file to get text from.
        limit: A limit to the number of entries to read.

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
            if limit and len(conversation_tokens) >= limit:
                break

    return conversation_tokens

def preprocess_line(line_text: str) -> list:
    """
    Preprocesses a line by converting it to tokens.

    Args:
        line_text: The text to convert to tokens.

    Returns:
        A list of tokens representing the line.
    """
    token_sents = tokenize.sent_tokenize(line_text)
    tagged_sents = []
    for sentence in token_sents:
        tokens = tokenize.word_tokenize(sentence)
        merged_tokens = merge_tokens(tokens)
        tags = pos_tagger.tag_words(merged_tokens)
        filtered_tags = filter_stopwords(tags)

        tagged_sents.append(TaggedLine(sentence, tags, filtered_tags))

    return tagged_sents

def merge_tokens(tokens: list) -> list:
    """
    Merges tokens that were incorrectly split and should be combined, like emoticons.

    Args:
        tokens: The original split token list.

    Returns:
        A list of merged tokens.
    """
    i = 0
    new_tokens = []
    while i < len(tokens) - 1:
        combined_token = tokens[i] + tokens[i + 1]
        if len(combined_token) == 2 and tokens[i] in {':', '='}:
            new_tokens.append(combined_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    if i < len(tokens):
        new_tokens.append(tokens[i])
    return new_tokens

def filter_stopwords(tags: list) -> list:
    """
    Filters out stopwords from a word list.

    Args:
        tags: The tagged word list to filter.

    Returns:
        A list of filtered tags.
    """
    return [tag for tag in tags if tag.word not in stopwords]