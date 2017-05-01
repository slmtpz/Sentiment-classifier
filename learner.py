import file_reader
from collections import Counter
import re


# Trains the model reading files in train set excluding stop words.
class Learner:
    def __init__(self, alpha):
        self.negative_counts = Counter()
        self.positive_counts = Counter()
        self.alpha = alpha

    def learn(self):
        print('learning P(word | class) for each word')
        regex = r'(?!the|a|and|to|of|is|in|s|as|i)([a-z]+)\b'
        texts = file_reader.read_files('train', 'neg')
        self.negative_training_text_count = len(texts)
        for text in texts:
            self.negative_counts.update(Counter(re.findall(regex, text)))

        self.negative_normalizer = sum(self.negative_counts.values()) + len(self.negative_counts) * self.alpha
        print('learning P(word | class="negative") completed')

        texts = file_reader.read_files('train', 'pos')
        self.positive_training_text_count = len(texts)
        for text in texts:
            self.positive_counts.update(Counter(re.findall(regex, text)))

        self.positive_normalizer = sum(self.positive_counts.values()) + len(self.positive_counts) * self.alpha

        self.total_training_text_count = self.negative_training_text_count + self.positive_training_text_count
        print('learning P(word | class="positive") completed')

        # print(self.negative_counts)
        # print(self.negative_normalizer)
        # print(self.negative_training_text_count)
        # print(self.positive_counts)
        # print(self.positive_normalizer)
        # print(self.positive_training_text_count)

    def get_likelihood_probability(self, word, _class):
        if _class == 'negative':
            try:
                enumerator = self.negative_counts[word]
            except KeyError:
                enumerator = 0
            return (enumerator + self.alpha) / self.negative_normalizer
        elif _class == 'positive':
            try:
                enumerator = self.positive_counts[word]
            except KeyError:
                enumerator = 0
            return (enumerator + self.alpha) / self.positive_normalizer

    def get_prior_probability(self, _class):
        if _class == 'negative':
            return self.negative_training_text_count / self.total_training_text_count
        elif _class == 'positive':
            return self.positive_training_text_count / self.total_training_text_count
