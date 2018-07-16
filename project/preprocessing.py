import os
import numpy as np

from collections import OrderedDict

from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

MIN_FREQ = 3


TRAIN_PATH = 'Penn_Treebank/train.gold.conll'
TEST_PATH = 'Penn_Treebank/dev.gold.conll'
# MAX_LENGTH = 120  # longest sequence to parse
# N_TAGS = 36


def invert_dict(d):
    res = {}
    for k, v in d.iteritems():
        res[v] = k
    return res


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
                curr.append((tokens[1], tokens[3]))
    return sents


def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1


def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab


def replace_word(word):
    """
        Replaces rare words with categories (numbers, dates, etc...)
    """
    ### YOUR CODE HERE
    import re

    def contains_digit_and_char(w, ch):
        return bool(re.search('\d', w)) and ch in word

    categories = {
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
        'punkMark': lambda w: word in (",", ".", ";", "?", "!", ":", ";", "-", '&'),
        'containsNonAlphaNumeric': lambda w: bool(re.search('\W', w)),
        'percent': lambda w: len(w) > 1 and w[0] == '%' and w[1:].isdigit()
    }

    for cat, cond in categories.iteritems():
        if cond(word):
            return cat

    ### END YOUR CODE
    return "UNK"


def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                new_sent.append((replace_word(token[0]), token[1]))
                replaced += 1
            total += 1
        res.append(new_sent)
    print("replaced: " + str(float(replaced) / total))
    return res


def preprocess_dataset(dataset_path):
    sents = read_conll_pos_file(dataset_path)
    words, tags = OrderedDict([]), OrderedDict([])
    max_length = 0
    for sent in sents:
        max_length = max(max_length, len(sent))
        for w, t in sent:
            words[w] = words.get(w, 0) + 1
            tags[t] = tags.get(t, 0) + 1

    vocab_size = len(words.items())
    n_tags = len(tags.items())

    word2idx = {w: idx for idx, (w, _) in enumerate(words.items())}
    tag2idx = {t: idx for idx, (t, _) in enumerate(tags.items())}

    print(tag2idx)

    x = [[word2idx[w] for w, _ in sent] for sent in sents]
    y = [[tag2idx[t] for _, t in sent] for sent in sents]

    x = pad_sequences(maxlen=max_length, sequences=x, padding="post", value=vocab_size-1)
    y = pad_sequences(maxlen=max_length, sequences=y, padding="post", value=tag2idx['#'])

    y = [to_categorical(t, n_tags) for t in y]

    return x, np.array(y), vocab_size, n_tags, max_length

    # vocab = compute_vocab_count(sents)
    # vocab_size = len(vocab)
    #
    # # x = [one_hot(' '.join(w for w, _ in sent), vocab_size) for sent in sents]
    # # x = pad_sequences(x, maxlen=MAX_LENGTH, padding='post')
    # # y = [one_hot(' '.join(t for _, t in sent), N_TAGS) for sent in sents]
    # # y = pad_sequences(y, maxlen=MAX_LENGTH, padding='post')
    #
    #
    #
    # return x, y, vocab_size


def preprocess_datasets(train_path, test_path):
    train_sents = read_conll_pos_file(train_path)
    test_sents = read_conll_pos_file(test_path)
    sents = train_sents + test_sents

    words, tags = OrderedDict([]), OrderedDict([])
    max_length = 0
    for sent in sents:
        max_length = max(max_length, len(sent))
        for w, t in sent:
            words[w] = words.get(w, 0) + 1
            tags[t] = tags.get(t, 0) + 1

    vocab_size = len(words.items())
    n_tags = len(tags.items())

    word2idx = {w: idx for idx, (w, _) in enumerate(words.items())}
    tag2idx = {t: idx for idx, (t, _) in enumerate(tags.items())}

    x_train = [[word2idx[w] for w, _ in sent] for sent in train_sents]
    y_train = [[tag2idx[t] for _, t in sent] for sent in train_sents]

    x_train = pad_sequences(maxlen=max_length, sequences=x_train, padding="post", value=vocab_size-1)
    y_train = pad_sequences(maxlen=max_length, sequences=y_train, padding="post", value=tag2idx['#'])

    y_train = [to_categorical(t, n_tags) for t in y_train]

    x_test = [[word2idx[w] for w, _ in sent] for sent in test_sents]
    y_test = [[tag2idx[t] for _, t in sent] for sent in test_sents]

    x_test = pad_sequences(maxlen=max_length, sequences=x_test, padding="post", value=vocab_size-1)
    y_test = pad_sequences(maxlen=max_length, sequences=y_test, padding="post", value=tag2idx['#'])

    y_test = [to_categorical(t, n_tags) for t in y_test]

    return x_train, np.array(y_train), x_test, np.array(y_test), vocab_size, n_tags, max_length


def load_data():
    x_train, y_train, x_test, y_test, vocab_size, n_tags, max_length = preprocess_datasets(TRAIN_PATH, TEST_PATH)

    return x_train, y_train, x_test, y_test, vocab_size, n_tags, max_length

