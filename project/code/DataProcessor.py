from collections import OrderedDict
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

import os
import re
import numpy as np
import pickle

NO_VALUE = 'NO_VALUE'
PADD = 'PADD'
UNK = 'UNK'

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


def contains_digit_and_char(word, ch):
    return bool(re.search('\d', word)) and ch in word


def pickle_path(name):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'pickles', name + '.pickle')
    return os.path.abspath(path)


class DataProcessor(object):
    def __init__(self, max_seq_len=40, rare_word_threshold=1, replace_global=False, name='DataProcessor'):
        self.word2idx = None
        self.tag2idx = None
        self.features2idx = None
        self.idx2tag = None
        self.max_seq_len = max_seq_len
        self.rare_word_threshold = rare_word_threshold
        self.unk_indices = []
        self.GLOBAL_CATEGORIES = OrderedDict({}) if not replace_global else GLOBAL_CATEGORIES
        self.name = name

    @staticmethod
    def transform_to_one_hot(sample, mapping_dict_length):
        """
        Transform one hot encoding for every sequence tags
        :param sample: np.ndarray object of shape (num_samples, max_seq_length) containing indices of words
        :param mapping_dict_length int representing the vector desired size
        :return: np.ndarray of shape (num_samples, max_seq_length, n_tags)
        every index is replaced with one hot vector of size n_tags
        """
        sample_one_hot = to_categorical(sample, mapping_dict_length)

        return sample_one_hot

    @staticmethod
    def transform_to_index(one_hot_sample):
        """
        Transform to the index of the one hot encoding a certain word
        :param one_hot_sample:
        :return: the one hot index
        """
        index_sample = []
        for sent in one_hot_sample:
            index_sample.append([np.where(x == 1)[0][0] for x in sent])

        return np.array(index_sample)

    @staticmethod
    def create_boolean_mask(x_sample, padding_index):
        """
        Create a boolean mask from a padded sequence
        :param x_sample: padded sequence matrix of shape (total_samples, max_input_length)
        :param padding_index: Int, index to use for padding
        :return: mask_sample: padded sequence of boolean values with shape (total_sample, max_input_length)
        """
        padding_matrix = np.full(shape=(x_sample.shape[0], x_sample.shape[1]), fill_value=padding_index)
        boolean_mask = x_sample != padding_matrix

        return boolean_mask

    def _init_word2idx_dict(self, total_words_dict):
        words = []
        unk_indices = []

        # only add words that exceed the rare threshold
        for word, count in total_words_dict.items():
            if count > self.rare_word_threshold:
                words.append(word)

        # add rare word categories
        for category in self.GLOBAL_CATEGORIES.keys():
            words.append(category)

        # and also the UNK and PADD symbols
        words.append(UNK)
        words.append(PADD)

        # transform words to indices
        word2idx = {w: idx for idx, w in enumerate(words)}

        # add unknown indices to the instance list
        unk_indices.append(word2idx['UNK'])
        for cat in self.GLOBAL_CATEGORIES.keys():
            unk_indices.append(word2idx[cat])

        self.unk_indices = unk_indices
        self.word2idx = word2idx

    def _replace_rare_word(self, word):
        """
        Replaces rare words with categories (numbers, dates, etc...)
        """

        # OrderedDict ensures us that the order of the insertion is preserved
        # the order of the condition checking does matter here
        for cat, cond in self.GLOBAL_CATEGORIES.items():
            if cond(word):
                return cat

        # if none of the conditions supplied - return generic "UNK" symbol
        return UNK

    def compute_percentile_sequence_length(self, file_path, percentile):
        sentences = self.read_file(file_path)
        sentences_lengths = []
        for sent in sentences:
            sentences_lengths.append(len(sent))

        return np.percentile(sentences_lengths, percentile)

    def read_file(self, path):
        """
            Takes a path to a file and returns a list of word/tag pairs
        """
        sents = []
        with open(path, encoding='utf-8', mode='r') as f:
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

                    word, pos_tag, raw_features = tokens[1], tokens[3], tokens[5]
                    features = {}

                    def fn(k):
                        # AKA featurename
                        return k.lower().replace('[', '').replace(']', '')

                    if raw_features != '_':
                        features = {fn(k): v for k, v in [rf.split('=') for rf in raw_features.split('|')]}

                    curr.append((word, pos_tag, features))

        return sents

    def process(self, dataset_path):
        sents = self.read_file(dataset_path)
        words, tags, features_dict = OrderedDict([]), set(), {}
        for sent in sents:
            for w, t, features in sent:
                words[w] = words.get(w, 0) + 1
                tags.add(t)
                for k, v in features.items():
                    if k not in features_dict:
                        features_dict[k] = set()
                    features_dict[k].add(v)

        tags = list(tags) + [PADD, UNK]
        features_dict = {k.lower(): list(v) + [NO_VALUE, PADD, UNK] for k, v in features_dict.items()}

        # Initiate word-to-index and tag-to-index and features-to-index dictionaries
        self._init_word2idx_dict(words)
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.features2idx = {k: {v: idx for idx, v in enumerate(feat)} for k, feat in features_dict.items()}

    def load(self, file_path=None):
        if not file_path:
            file_path = pickle_path(self.name)

        with open(file_path, 'rb') as handle:
            tmp_dict = pickle.load(handle)
            self.__dict__.update(tmp_dict)

    def save(self, file_path=None):
        if not file_path:
            file_path = pickle_path(self.name)

        with open(file_path, "wb") as handle:
            # attributes = [self.word2idx, self.tag2idx]
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_word2idx_dict(self):
        return self.word2idx

    def get_tag2idx_dict(self):
        return self.tag2idx

    def get_features2idx_dicts(self):
        return self.features2idx

    def get_max_sequence_length(self):
        return self.max_seq_len

    def set_max_sequence_length(self, max_len):
        self.max_seq_len = max_len

    def get_idx2tag_dict(self):
        if self.idx2tag is None:
            self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        return self.idx2tag

    def get_features(self):
        return list(self.features2idx.keys())

    def get_name(self):
        return self.name

    def preprocess_sample(self, sample_path):
        if self.word2idx is None or self.tag2idx is None or self.features2idx is None:
            raise AssertionError(
                "You must perform preprocessing for a training set first in order to initiate words indexes")

        sents = self.read_file(sample_path)

        # build sample and labels by replacing words and tags with matching idx
        # Words never seen before will be replaced by the index of their category
        x = [[self.word2idx.get(w, self.word2idx[self._replace_rare_word(w)]) for w, _, _ in sent] for sent in sents]
        y = [[self.tag2idx.get(t, self.tag2idx[UNK]) for _, t, _ in sent] for sent in sents]
        # y_features = {k: [[feature2idx[f] for _, _, f in sent] for sent in sents] for k, feature2idx in
        #               self.features2idx.items()}

        # TODO: notice! UNK fallback added!
        y_features = {k.lower(): [[feature2idx.get(f.get(k.lower(), NO_VALUE), feature2idx[UNK]) for _, _, f in sent] for sent in sents] for
                      k, feature2idx in self.features2idx.items()}

        # perform padding to structure every sentence example to a defined size
        # use "PADD" symbol as padding value

        x = pad_sequences(maxlen=self.max_seq_len, sequences=x, padding="post", truncating="post",
                          value=self.word2idx[PADD])
        y = pad_sequences(maxlen=self.max_seq_len, sequences=y, padding="post", truncating="post",
                          value=self.tag2idx[PADD])
        y_features = {k: pad_sequences(maxlen=self.max_seq_len, sequences=y_features[k],
                                       padding="post", truncating="post",
                                       value=feature2idx[PADD]) for k, feature2idx in self.features2idx.items()}

        return x, y, y_features

    # used in order to predict sample.
    def preprocess_sentence(self, sent):
        if type(sent) is str:
            sent = sent.split(' ')
        x = [[self.word2idx.get(w, self.word2idx[self._replace_rare_word(w)]) for w in sent]]
        x = pad_sequences(maxlen=self.max_seq_len, sequences=x, padding="post", truncating="post",
                          value=self.word2idx[PADD])

        return x[0]
