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
        self.conversation = []
        self.conversation_set = set()
        self.stopwords = stopwords.words('english')
        self.ended = False
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def start_conversation(self) -> str:
        """
        Resets the conversation.

        Returns:
            The first message in the conversation.
        """
        starter = self.chatbot.chatbot.getRandomStarter()
        self.add_response(starter)
        ChatbotWrapper.respond(self.chatbot, '')
        print(starter)
        return starter

    def choose_message(self, response: str) -> str:
        """
        Chooses a message to send in the conversation.

        Args:
            response: The response to the conversation.

        Returns:
            The next message in the conversation.
        """
        chatbot_response = self.chatbot.respond(response)
        self.add_response(chatbot_response)
        return chatbot_response

    def evaluate_response(self, response: str) -> float:
        """
        Returns a reward for a certain response.

        Args:
            response: The response to the conversation.

        Returns: The reward for the response.
        """
        if response in self.conversation_set:
            self.ended = True
        self.add_response(response)

        last_response = ''
        if len(self.conversation) > 2:
            last_response = self.conversation[-3]
        current_message = self.conversation[-2]

        last_response_keywords = [word for word in last_response.split(' ') if word not in self.stopwords]
        response_keywords = [word for word in response.split(' ') if word not in self.stopwords]
        num_last_keywords = len(last_response_keywords)
        num_current_keywords = len(response_keywords)

        embeddings = self.chatbot.chatbot.embeddings

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

        dissimilarity_score = 0.5 - average_similarity

        sentiment_score = self.sentiment_analyzer.polarity_scores(response)['compound']

        final_score = (dissimilarity_score + sentiment_score) / 2
        print('Score:', final_score, ', Dissimilarity:', dissimilarity_score, ', Sentiment:', sentiment_score)
        return final_score

    def is_ended(self) -> bool:
        """
        Checks if the conversation has ended.

        Returns: Whether the conversation has ended.
        """
        return False

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

    def get_word_index(self, word):
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
