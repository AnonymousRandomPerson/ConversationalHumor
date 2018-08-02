import string

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import spatial

from rl.chatbots import ChatbotWrapper, NormalChatbot
from rl.conversation import Conversation
from utils.file_access import add_module, CHATBOT_MODULE

add_module(CHATBOT_MODULE)

import DeepQA.chatbot.chatbot as chatbot

class EvaluatedConversation(Conversation):
    """
    A conversation that is evaluated with a non-trivial function.
    """

    def __init__(self, chatbot_object: chatbot.Chatbot):
        """
        Sets up the chatbot to be used in the conversation.

        Args:
            sess: The Tensorflow session to use with the chatbot.
        """
        Conversation.__init__(self)
        self.chatbot = NormalChatbot(chatbot_object, 'Other')
        self.chatbot.show_other = False
        self.conversation = []
        self.conversation_set = set()
        self.stopwords = stopwords.words('english')
        self.ended = False
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def start_conversation(self, starter: str = '') -> str:
        """
        Resets the conversation.

        Args:
            starter: An optional pre-defined conversation starter.

        Returns:
            The first message in the conversation.
        """
        self.conversation = []
        self.conversation_set = set()
        self.ended = False

        if not starter:
            starter = self.chatbot.chatbot.getRandomStarter()
            ChatbotWrapper.respond(self.chatbot, '')
            print(starter)
        self.add_response(starter)
        return starter

    def choose_message(self, response: str, next_message: str = '') -> str:
        """
        Chooses a message to send in the conversation.

        Args:
            response: The response to the conversation.
            next_message: A predefined next message in the conversation, if not blank.

        Returns:
            The next message in the conversation.
        """
        if response in self.conversation_set:
            self.ended = True

        self.add_response(response)
        if not next_message:
            next_message = self.chatbot.respond(response)
        self.add_response(next_message)
        return next_message

    def evaluate_response(self, response: str, next_message: str) -> float:
        """
        Returns a reward for a certain response.

        Args:
            response: The response to the conversation.
            next_message: The next message that will be said in response to the first response.

        Returns: The reward for the response.
        """
        last_response = ''
        if len(self.conversation) > 3:
            last_response = self.conversation[-3]

        last_response_keywords = [word for word in last_response.split(' ') if word not in self.stopwords]
        response_keywords = [word for word in response.split(' ') if word not in self.stopwords]
        num_current_keywords = len(response_keywords)

        embeddings = self.chatbot.chatbot.embeddings

        if last_response:
            # Reward for using a different response than the previous response.
            average_similarity = 0

            for current_word in response_keywords:
                max_similarity = 0
                current_index = self.get_word_index(current_word)
                for last_word in last_response_keywords:
                    last_index = self.get_word_index(last_word)
                    cos_similarity = 1 - spatial.distance.cosine(embeddings[current_index], embeddings[last_index])
                    max_similarity = max(max_similarity, cos_similarity)
                average_similarity += max_similarity

            average_similarity /= num_current_keywords

            dissimilarity_score = min(0.0, 0.5 -average_similarity)
        else:
            dissimilarity_score = 0.0

        # Reward for using humor when the other side is in a good mood.
        current_sentiment = self.get_sentiment(response)
        if response.endswith('!'):
            current_sentiment_score = current_sentiment
        else:
            current_sentiment_score = 0.0

        # Reward for positively changing the other side's sentiment
        next_sentiment = self.get_sentiment(next_message)

        sentiment_change_score = 0.0
        sentiment_difference = next_sentiment - current_sentiment

        if sentiment_difference < 0.0 and next_sentiment >= 0.0:
            # Avoid negative reward for fluctuating between positive and neutral sentiment.
            sentiment_difference = 0.0

        sentiment_change_score = sentiment_difference

        final_score = (dissimilarity_score + current_sentiment_score + sentiment_change_score) / 3
        if self.chatbot.chatbot.outer_args.debug_print:
            print('Score:', str(final_score) + ', Dissimilarity:', str(dissimilarity_score) + ', Current sentiment:', str(current_sentiment_score) + ', Sentiment change:', sentiment_change_score)
        return final_score

    def is_ended(self) -> bool:
        """
        Checks if the conversation has ended.

        Returns: Whether the conversation has ended.
        """
        return self.ended

    def on_response(self):
        """
        Does pre-processing before a response is evaluated.
        """
        pass

    def add_response(self, response: str):
        """
        Adds a response to the conversation history.
        """
        self.conversation.append(response)
        self.conversation_set.add(response)

    def get_word_index(self, word: str) -> int:
        """
        Gets the index of a word in the chatbot vocabulary.

        Args:
            word: The word to get an index for.

        Returns: The index of the word in the chatbot vocabulary.
        """
        word2id = self.chatbot.chatbot.textData.word2id

        if word and word[-1] in string.punctuation:
            word = word[:-1]

        word = word.lower()
        if word in word2id:
            return word2id[word]

        return self.chatbot.chatbot.textData.unknownToken

    def get_sentiment(self, sentence: str) -> float:
        """
        Gets the compound sentiment value of a sentence.

        Args:
            sentence: The sentence to get a sentiment value for.

        Returns: The compound sentiment value of a sentence.
        """
        return self.sentiment_analyzer.polarity_scores(sentence)['compound']