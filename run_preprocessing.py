from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessor import preprocess_text

example_limit = 5

positive_examples = preprocess_text('twitter_filtered.txt', example_limit)
negative_examples = preprocess_text('twitter_negative.txt', example_limit)
all_examples = positive_examples + negative_examples

all_sentences = []
for conversation in all_examples:
    for tagged_lines in conversation:
        for tagged_line in tagged_lines:
            all_sentences.append(tagged_line.line)

count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(all_sentences)
freq_matrix = count_vectorizer.transform(all_sentences)

tfidf = TfidfTransformer()
tfidf.fit(freq_matrix)
tfidf_matrix = tfidf.transform(freq_matrix)
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
pass