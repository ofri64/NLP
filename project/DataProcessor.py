import numpy as np
import re
import pickle
import os
import json
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

    def __init__(self, max_seq_len=40, rare_word_threshold=1, from_file=False, save_load_path=None):
        self.word2idx = None
        self.tag2idx = None
        self.vocab_size = None
        self.n_classes = None
        self.max_seq_len = max_seq_len
        self.rare_word_threshold = rare_word_threshold
        self.from_file = from_file
        self.save_load_path = save_load_path

        if from_file:
            try:
                with open(save_load_path, "rb") as handle:
                    self.word2idx, self.tag2idx, self.vocab_size, self.n_classes = pickle.load(handle)
            except (FileNotFoundError, TypeError) as e:
                print("Error while trying to load model parameters from file\n" + e)

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

                    if word in PUNCTUATION_MARKS or pos_tag in PUNCTUATION_MARKS:
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

    def initiate_word_tags_dicts(self, train_path):
        train_sents = self.read_conll_pos_file(train_path)

        # Iterate all tokens in training set
        words, tags = OrderedDict([]), OrderedDict([])
        for sent in train_sents:
            for w, t in sent:
                words[w] = words.get(w, 0) + 1
                tags[t] = tags.get(t, 0) + 1

        # Initiate word-to-index and tag-to-index dictionaries
        self.word2idx = self._initiate_word2idx_dict(words)
        self.tag2idx = self._initiate_tag2idx_dict(tags)
        self.vocab_size = len(self.word2idx)
        self.n_classes = len(self.tag2idx)

        if self.save_load_path is None:
            self.save_load_path = os.getcwd() + "/words_tags_dict.pickle"
        try:
            attributes = [self.word2idx, self.tag2idx, self.vocab_size, self.n_classes]
            with open(self.save_load_path, "wb") as handle:
                pickle.dump(attributes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except (FileNotFoundError, TypeError) as e:
            print("Could not save the word and tags dict to a file."
                  "please check the path you provided exists\n" + e)

    def _initiate_word2idx_dict(self, total_words_dict):
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

    def _initiate_tag2idx_dict(self, total_tags_dict):
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

    def preprocess_sample_set(self, sample_path):
        """
        Processes a sample set from a given file path
        Returns a tuple of np.ndarrays for test set features (word idx) and test labels
        """
        if self.word2idx is None or self.tag2idx is None:
            raise AssertionError(
                "You must perform preprocessing for a training set first in order to initiate words indexes")

        sents = self.read_conll_pos_file(sample_path)

        # build sample and labels by replacing words and tags with matching idx
        # Words never seen before will be replaced by the index of their category
        x = [[self.word2idx.get(w, self.word2idx[self.replace_rare_word(w)]) for w, _ in sent] for sent in sents]
        y = [[self.tag2idx[t] for _, t in sent] for sent in sents]

        # perform padding to structure every sentence example to a defined size
        # use "PADD" symbol as padding value

        x = pad_sequences(maxlen=self.max_seq_len, sequences=x, padding="post", truncating="post", value=self.word2idx["PADD"])
        y = pad_sequences(maxlen=self.max_seq_len, sequences=y, padding="post", truncating="post", value=self.tag2idx["PADD"])

        return x, y

    def save_words_dict(self):
        if self.word2idx is not None:
            with open('word2idx.json', 'w') as outfile:
                json.dump(self.word2idx, outfile)

        else:
            print("Words were not initiated.")

class HebrewDataProcessor(DataProcessor):

    def __init__(self, max_seq_len=40, rare_word_threshold=1, from_file=False, save_load_path=None):
        super(HebrewDataProcessor, self).__init__(max_seq_len, rare_word_threshold, from_file, save_load_path)

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
                    index = tokens[0]

                    # Non-separated words contain a hyphen in index number,
                    # Comments start with hash
                    if '-' in index or index[0] == '#':
                        continue

                    word, pos_tag = tokens[1], tokens[4]
                    curr.append((word, pos_tag))
        return sents

    def save_words_dict(self):
        if self.word2idx is not None:
            with open('word2idx-heb.json', 'w') as outfile:
                json.dump(self.word2idx, outfile)

        else:
            print("Words were not initiated.")