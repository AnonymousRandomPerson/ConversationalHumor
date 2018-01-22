import collections
import os
import ahocorasick

Message = collections.namedtuple('Message', ['text', 'user'])
punctuation = {'.', '?', '!', '"', "'"}
humor_words = ['lol', 'haha', 'ha ha', 'lmao', 'rofl', 'rotfl', ':p', '=p', 'xd', '😂', '😆', '😜']
data_folder = os.path.join(os.path.dirname(__file__), 'data')
automaton = ahocorasick.Automaton()
for word in humor_words:
    automaton.add_word(word, word)
automaton.make_automaton()

class Extractor(object):
    """
    Extracts humorous excerpts from a corpus.
    """

    def __init__(self, debug: bool = False):
        """
        Initializes the files to write to.

        Args:
            debug: Whether to enable debug printing.
        """
        self.debug = debug
        self.num_excerpts = 0

    @property
    def corpus_name(self) -> str:
        """
        The name of the corpus to extract from.
        """
        return "corpus"

    @property
    def skip_words(self) -> set:
        """
        Server-side messages that should be skipped.
        """
        return {}

    @property
    def conversation_length(self) -> int:
        """
        The maximum number of messages in a conversation.
        """
        return 3

    def get_corpus(self) -> list:
        """
        Gets the corpus data to extract from.

        Returns:
            The corpus data to extract from.
        """
        return []

    def post_process_text(self, text: str) -> str:
        """
        Processes a raw message to convert it to a usable format.

        Args:
            text: The text to process.

        Returns:
            A cleaned version of the given text.
        """
        return text.strip()

    def allow_write(self) -> bool:
        """
        Checks if the current state of the extractor allows a conversation to be written.
        """
        return True

    def extract(self):
        """
        Extracts humorous excerpts from a corpus.
        """
        filtered_file = open(os.path.join(data_folder, self.corpus_name + '_filtered.txt'), 'w')
        negative_file = open(os.path.join(data_folder, self.corpus_name + '_negative.txt'), 'w')
        raw_file = open(os.path.join(data_folder, self.corpus_name + '_raw.txt'), 'w')
        message_queue = collections.deque(maxlen=self.conversation_length)

        found_humor = False
        num_negatives = 0

        def check_write_queue(self):
            """
            Writes the current message queue if humor has been found.
            """
            if self.allow_write():
                if found_humor:
                    new_write = ''
                    for queued_message in message_queue:
                        new_write += queued_message.text + '\n'
                    if self.debug:
                        print(new_write)
                    filtered_file.write(new_write + '\n')
                    self.num_excerpts += 1
                elif num_negatives >= self.conversation_length:
                    new_write = ''
                    for queued_message in message_queue:
                        new_write += queued_message.text + '\n'
                    negative_file.write(new_write + '\n')

        for post in self.get_corpus():
            post_text = post.text
            raw_file.write(post_text + '\n')

            # Skip server messages.
            if post_text in self.skip_words:
                continue

            post_text = self.post_process_text(post_text)

            # Check if this message was posted by the same user as the last message.
            if message_queue:
                last_message = message_queue.pop()
                if post.user and last_message.user == post.user:
                    last_text = last_message.text
                    if last_text[-1] not in punctuation:
                        last_text += '.'
                    post_text = last_text + ' ' + post_text
                else:
                    message_queue.append(last_message)
                    check_write_queue(self)

            message = Message(post_text, post.user)
            message_queue.append(message)

            # Use the Aho-Corasick algorithm for fast simultaneous string searching.
            found_humor = False
            lower_text = post_text.lower()
            for _, _ in automaton.iter(lower_text):
                found_humor = True
                break

            if found_humor:
                num_negatives = 0
            else:
                num_negatives += 1

        check_write_queue(self)

        print('Number of excerpts:', self.num_excerpts)