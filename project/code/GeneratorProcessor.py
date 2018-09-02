import os
import pickle
import sys

from collections import OrderedDict
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

NO_VALUE = 'NO_VALUE'
PADD = 'PADD'
UNK = 'UNK'


def pickle_path(name):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'pickles', name + '.pickle')
    return os.path.abspath(path)


def dataset_path(name, file_type):
    path = os.path.join(os.path.dirname(__file__), os.pardir, "datasets", name, "{0}.conllu".format(file_type))
    return os.path.abspath(path)


class GeneratorDataProcessor(object):

    def __init__(self, max_seq_len=40, rare_word_threshold=1, batch_size=64, name='GeneratorDataProcessor'):
        self.name = name
        self.word2idx = None
        self.tag2idx = None
        self.max_seq_len = max_seq_len
        self.rare_word_threshold = rare_word_threshold
        self.batch_size = batch_size
        self.ambig_words_indices = set()

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

    def _create_word_tags_counters(self, conllu_file):
        word_counter = OrderedDict()
        tags_counter = OrderedDict()
        word_tag_comb = OrderedDict()
        ambiguous_words = set()

        with open(conllu_file, encoding='utf-8', mode='r') as f:
            for line in f:
                line = line.strip()
                if line == "":  # it is empty line (end of sentence)
                    continue

                else:
                    tokens = line.strip().split("\t")
                    index = tokens[0]

                # Non-separated words contain a hyphen in index number,
                # Comments start with hash
                if '-' in index or index[0] == '#':
                    continue

                # else - it is a valid line
                word, pos_tag = tokens[1], tokens[3]

                # add to word and tag counters
                word_counter[word] = word_counter.get(word, 0) + 1
                tags_counter[pos_tag] = tags_counter.get(pos_tag, 0) + 1

                # add word and tag combination add check if ambiguous
                if word not in word_tag_comb:
                    word_tag_comb[word] = set()

                word_tags = word_tag_comb[word]
                word_tags.add(pos_tag)
                if len(word_tags) > 1:
                    ambiguous_words.add(word)

        return word_counter, tags_counter, ambiguous_words

    def create_vocabs(self, conllu_file):

        word_counter, tags_counter, ambiguous_words = self._create_word_tags_counters(conllu_file)

        # Now we create word2index and tag2index dicts
        vocab_words = set()
        vocab_tags = set()

        # only add words that exceed the rare threshold
        for word, count in word_counter.items():
            if count > self.rare_word_threshold:
                vocab_words.add(word)

        for tag, count in tags_counter.items():
            if count > self.rare_word_threshold:
                vocab_tags.add(tag)

        # add the UNK and PADD symbols
        vocab_words.add(UNK)
        vocab_words.add(PADD)

        vocab_tags.add(UNK)
        vocab_tags.add(PADD)

        # transform words and tags to indices
        self.word2idx = {w: idx for idx, w in enumerate(vocab_words)}
        self.tag2idx = {t: idx for idx, t in enumerate(vocab_tags)}

        # transform ambiguous words to indices
        for word in ambiguous_words:
            self.ambig_words_indices.add(self.word2idx[word])

    def get_sample_generator(self, conllu_file):
        while True:
            current_sent = []
            current_tags = []
            current_sents_batch = []
            current_tags_batch = []
            current_batch_size = 0
            n_classes = len(self.tag2idx)
            with open(conllu_file, encoding='utf-8', mode='r') as f:
                for line in f:
                    line = line.strip()

                    if line == "":  # it is empty line (end of sentence)

                        current_sents_batch.append(current_sent)
                        current_tags_batch.append(current_tags )
                        current_sent = []
                        current_tags = []
                        current_batch_size += 1

                        if current_batch_size == self.batch_size:  # keep collecting until reaching mini batch size
                            x = pad_sequences(maxlen=self.max_seq_len, sequences=current_sents_batch, padding="post",
                                              truncating="post", value=self.word2idx[PADD])
                            y = pad_sequences(maxlen=self.max_seq_len, sequences=current_tags_batch, padding="post",
                                              truncating="post", value=self.tag2idx[PADD])
                            y = to_categorical(y, n_classes)

                            current_batch_size = 0
                            current_sents_batch = []
                            current_tags_batch = []

                            yield (x, y)

                    else:
                        tokens = line.strip().split("\t")
                        index = tokens[0]

                        # Non-separated words contain a hyphen in index number,
                        # Comments start with hash
                        if '-' in index or index[0] == '#':
                            continue

                        else:
                            word, pos_tag = tokens[1], tokens[3]
                            word_unk_index = self.word2idx[UNK]
                            tag_unk_index = self.tag2idx[UNK]
                            word = self.word2idx.get(word, word_unk_index)
                            pos_tag = self.tag2idx.get(pos_tag, tag_unk_index)
                            current_sent.append(word)
                            current_tags.append(pos_tag)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("Usage {0}: Please insert file name to process".format(sys.argv[0]))

    dataset_name = sys.argv[1]
    dataset_type = sys.argv[2]
    train_set = dataset_path(dataset_name, dataset_type)

    generator_dp = GeneratorDataProcessor()
    # generator_dp.create_vocabs(train_set)
    generator_dp.load()
    generator = generator_dp.get_sample_generator(train_set)
    for i in range(5):
        a = next(generator)
        print(a[0].shape)

