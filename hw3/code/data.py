import os

MIN_FREQ = 3


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
    print "replaced: " + str(float(replaced) / total)
    return res
