import numpy as np
import pickle
from collections import OrderedDict
from keras.utils import to_categorical


class DataProcessorInterface:

    def __init__(self, max_seq_len=None, rare_word_threshold=1):
        self.word2idx = None
        self.tag2idx = None
        self.max_seq_len = max_seq_len
        self.rare_word_threshold = rare_word_threshold

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
        raise NotImplementedError("Init word2idx dict method is not yet implemented")

    def _init_tag2idx_dict(self, total_tags_dict):
        raise NotImplementedError("Init tag2idx dict method is not yet implemented")

    def compute_percentile_sequence_length(self, file_path, percentile):
        sentences = self.read_file(file_path)
        sentences_lengths = []
        for sent in sentences:
            sentences_lengths.append(len(sent))

        return np.percentile(sentences_lengths, percentile)

    def create_word_tags_dicts(self, train_file_path):
        train_sents = self.read_file(train_file_path)

        # Iterate all tokens in training set
        words, tags = OrderedDict([]), OrderedDict([])
        for sent in train_sents:
            for w, t in sent:
                words[w] = words.get(w, 0) + 1
                tags[t] = tags.get(t, 0) + 1

        # Initiate word-to-index and tag-to-index dictionaries
        self.word2idx = self._init_word2idx_dict(words)
        self.tag2idx = self._init_tag2idx_dict(tags)

        return self.word2idx, self.tag2idx

    def load_word_tags_dicts(self, file_path):
        with open(file_path, "rb") as handle:
            self.word2idx, self.tag2idx = pickle.load(handle)

    def save_word_tags_dict(self, file_path):
        with open(file_path, "wb") as handle:
            attributes = [self.word2idx, self.tag2idx]
            pickle.dump(attributes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def read_file(self, file_path):
        raise NotImplementedError("Read file method is not implemented yet")

    def preprocess_sample(self, file_path):
        """
        Processes a sample set from a given file path
        Returns a tuple of np.ndarrays for data set features (word idx) and data set labels
        """
        raise NotImplementedError("Preprocess sample method is not implemented yet")

    def get_word2idx_vocab(self):
        return self.word2idx

    def get_tag2idx_vocab(self):
        return self.tag2idx

    def get_max_sequence_length(self):
        return self.max_seq_len

    def set_max_sequence_length(self, max_len):
        self.max_seq_len = max_len

