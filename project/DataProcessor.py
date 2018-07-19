import numpy as np
import re
from collections import OrderedDict
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


def contains_digit_and_char(word, ch):
    return bool(re.search('\d', word)) and ch in word


GLOBAL_CATEGORIES = OrderedDict({
    'twoDigitNum': lambda w: len(w) == 2 and w.isdigit() and w[0] != '0',
    'fourDigitNum': lambda w: len(w) == 4 and w.isdigit() and w[0] != '0',
    'containsDigitAndAlpha': lambda w: bool(re.search('\d', w)) and bool(re.search('[a-zA-Z_]', w)),
    'containsDigitAndDash': lambda w: contains_digit_and_char(w, '-'),
    'containsDigitAndSlash': lambda w: contains_digit_and_char(w, '/'),
    'containsDigitAndComma': lambda w: contains_digit_and_char(w, ','),
    'containsDigitAndPeriod': lambda w: contains_digit_and_char(w, '.'),
    'otherNum': lambda w: w.isdigit(),
    'allCaps': lambda w: w.isupper(),
    'capPeriod': lambda w: len(w) == 2 and w[1] == '.' and w[0].isupper(),
    'initCap': lambda w: len(w) > 1 and w[0].isupper(),
    'lowerCase': lambda w: w.islower(),
    'punkMark': lambda w: w in (",", ".", ";", "?", "!", ":", ";", "-", '&'),
    'containsNonAlphaNumeric': lambda w: bool(re.search('\W', w)),
    'percent': lambda w: len(w) > 1 and w[0] == '%' and w[1:].isdigit()
})

PUNCTUATION_MARKS = [
    ".", ",", "$", "\"", "'", "-LRB-", "-RRB-", "(", ")", "''", "#",
    "[", "]", "{", "}", "<", ">", ":", ";", "?", "!", "-", "--", "``"
]


class DataProcessor(object):

    def __init__(self, max_seq_len=40, rare_word_threshold=1):
        self.word2idx = None
        self.tag2idx = None
        self.vocab_size = None
        self.n_classes = None
        self.max_seq_len = max_seq_len
        self.rare_word_threshold = rare_word_threshold

    @staticmethod
    def read_conll_pos_file(path):
        """
            Takes a path to a file and returns a list of word/tag pairs
        """
        sents = []
        with open(path, "r") as f:
            curr = []
            for line in f:
                line = line.strip()
                if line == "":
                    sents.append(curr)
                    curr = []
                else:
                    tokens = line.strip().split("\t")
                    word, pos_tag = tokens[1], tokens[3]

                    if word in PUNCTUATION_MARKS:
                        pos_tag = "PUNCT"

                    curr.append((word, pos_tag))
        return sents

    @staticmethod
    def replace_rare_word(word):
        """
        Replaces rare words with categories (numbers, dates, etc...)
        """

        # OrderedDict ensures us that the order of the insertion is preserved
        # the order of the condition checking does matter here
        for cat, cond in GLOBAL_CATEGORIES.items():
            if cond(word):
                return cat

        # if none of the conditions supplied - return generic "UNK" symbol
        return "UNK"

    def initiate_word2idx_dict(self, total_words_dict):
        words_set = set()
        for word, count in total_words_dict.items():
            if count > self.rare_word_threshold:
                words_set.add(word)

        # add rare word categories
        for category in GLOBAL_CATEGORIES.keys():
            words_set.add(category)

        # and also the UNK and PADD symbols
        words_set.add("UNK")
        words_set.add("PADD")

        # transform words to indices
        word2idx = {w: idx for idx, w in enumerate(words_set)}

        return word2idx

    def initiate_tag2idx_dict(self, total_tags_dict):
        tag2idx = {t: idx for idx, t in enumerate(total_tags_dict.keys())}

        # add padding symbol
        tag2idx["PADD"] = len(tag2idx)
        return tag2idx

    def transform_to_one_hot(self, tag_sample):
        """
        Transform one hot encoding for every sequence tags
        :param sample: np.ndarray object of shape (num_samples, max_seq_length) containing
        indexes of words
        :return: np.ndarray of shape (num_samples, max_seq_length, n_tags)
        every index is replaced with one hot vector of size n_tags
        """
        n_tags = len(self.tag2idx)
        tag_sample_one_hot = np.array([to_categorical(t, n_tags) for t in tag_sample])

        return tag_sample_one_hot

    def create_boolean_mask(self, x_sample):
        """
        Create a boolean mask from a padded sequence
        :param x_sample: padded sequence matrix of shape (total_samples, max_input_length)
        :return: mask_sample: padded sequence of boolean values with shape (total_sample, max_input_length)
        """
        padding_idx = self.word2idx["PADD"]
        padding_matrix = np.full(shape=(x_sample.shape[0], x_sample.shape[1]), fill_value=padding_idx)
        boolean_mask = x_sample != padding_matrix

        return boolean_mask

    def preprocess_train_set(self, train_path):
        """
        Processes a training set from a given file path and initiate the word2idx and
        tax2idx members
        Returns a tuple of np.ndarrays of training set features (word idx) and training labels
        """
        train_sents = DataProcessor.read_conll_pos_file(train_path)

        # Iterate all tokens in training set
        words, tags = OrderedDict([]), OrderedDict([])
        for sent in train_sents:
            for w, t in sent:
                words[w] = words.get(w, 0) + 1
                tags[t] = tags.get(t, 0) + 1

        # Initiate word-to-index and tag-to-index dictionaries
        self.word2idx = self.initiate_word2idx_dict(words)
        self.tag2idx = self.initiate_tag2idx_dict(tags)
        self.vocab_size = len(self.word2idx)
        self.n_classes = len(self.tag2idx)

        # build sample and labels by replacing words and tags with matching idx
        # Words too rate will be replaced by the index of their category
        x_train = [[self.word2idx.get(w, self.word2idx[self.replace_rare_word(w)]) for w, _ in sent] for sent in train_sents]
        y_train = [[self.tag2idx[t] for _, t in sent] for sent in train_sents]

        # perform padding to structure every sentence example to a pre-defined size
        # use "PADD" symbol as padding value
        x_train = pad_sequences(maxlen=self.max_seq_len, sequences=x_train, padding="post", truncating="post", value=self.word2idx["PADD"])
        y_train = pad_sequences(maxlen=self.max_seq_len, sequences=y_train, padding="post", truncating="post", value=self.tag2idx["PADD"])

        return x_train, y_train

    def preprocess_test_set(self, test_path):
        """
        Processes a test set from a given file path
        Returns a tuple of np.ndarrays for test set features (word idx) and test labels
        """
        if self.word2idx is None or self.tag2idx is None:
            raise AssertionError(
                "You must perform preprocessing for a training set first in order to initiate words indexes")

        test_sents = DataProcessor.read_conll_pos_file(test_path)

        # build sample and labels by replacing words and tags with matching idx
        # Words never seen before will be replaced by the index of their category
        x_test = [[self.word2idx.get(w, self.word2idx[self.replace_rare_word(w)]) for w, _ in sent] for sent in test_sents]
        y_test = [[self.tag2idx[t] for _, t in sent] for sent in test_sents]

        # perform padding to structure every sentence example to a defined size
        # use "PADD" symbol as padding value

        x_test = pad_sequences(maxlen=self.max_seq_len, sequences=x_test, padding="post", truncating="post", value=self.word2idx["PADD"])
        y_test = pad_sequences(maxlen=self.max_seq_len, sequences=y_test, padding="post", truncating="post", value=self.tag2idx["PADD"])

        return x_test, y_test
