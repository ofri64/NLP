import re
from collections import OrderedDict
from keras.preprocessing.sequence import pad_sequences
from DataProcessorInterface import DataProcessorInterface


class EnglishDataProcessor(DataProcessorInterface):

    def __init__(self, max_seq_len=40, rare_word_threshold=1):

        # initiate with a default sequence length of 40
        super(EnglishDataProcessor, self).__init__(max_seq_len, rare_word_threshold)
        self.GLOBAL_CATEGORIES = OrderedDict({
            'twoDigitNum': lambda w: len(w) == 2 and w.isdigit() and w[0] != '0',
            'fourDigitNum': lambda w: len(w) == 4 and w.isdigit() and w[0] != '0',
            'containsDigitAndAlpha': lambda w: bool(re.search('\d', w)) and bool(re.search('[a-zA-Z_]', w)),
            'containsDigitAndDash': lambda w: self._contains_digit_and_char(w, '-'),
            'containsDigitAndSlash': lambda w: self._contains_digit_and_char(w, '/'),
            'containsDigitAndComma': lambda w: self._contains_digit_and_char(w, ','),
            'containsDigitAndPeriod': lambda w: self._contains_digit_and_char(w, '.'),
            'otherNum': lambda w: w.isdigit(),
            'allCaps': lambda w: w.isupper(),
            'capPeriod': lambda w: len(w) == 2 and w[1] == '.' and w[0].isupper(),
            'initCap': lambda w: len(w) > 1 and w[0].isupper(),
            'lowerCase': lambda w: w.islower(),
            'puncMark': lambda w: w in (",", ".", ";", "?", "!", ":", ";", "-", '&'),
            'containsNonAlphaNumeric': lambda w: bool(re.search('\W', w)),
            'percent': lambda w: len(w) > 1 and w[0] == '%' and w[1:].isdigit()
        })
        self.PUNCTUATION_MARKS = [
            ".", ",", "$", "\"", "'", "-LRB-", "-RRB-", "(", ")", "''", "#",
            "[", "]", "{", "}", "<", ">", ":", ";", "?", "!", "-", "--", "``"
        ]

    @staticmethod
    def _contains_digit_and_char(word, ch):
        return bool(re.search('\d', word)) and ch in word

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
        return "UNK"

    def _init_word2idx_dict(self, total_words_dict):
        words_set = set()
        for word, count in total_words_dict.items():
            if count > self.rare_word_threshold:
                words_set.add(word)

        # add rare word categories
        for category in self.GLOBAL_CATEGORIES.keys():
            words_set.add(category)

        # and also the UNK and PADD symbols
        words_set.add("UNK")
        words_set.add("PADD")

        # transform words to indices
        word2idx = {w: idx for idx, w in enumerate(words_set)}

        return word2idx

    def _init_tag2idx_dict(self, total_tags_dict):
        tag2idx = {t: idx for idx, t in enumerate(total_tags_dict.keys())}

        # add padding symbol
        tag2idx["PADD"] = len(tag2idx)
        return tag2idx

    def read_file(self, file_path):
        sents = []
        with open(file_path, "r") as f:
            curr = []
            for line in f:
                line = line.strip()
                if line == "":
                    sents.append(curr)
                    curr = []
                else:
                    tokens = line.strip().split("\t")
                    word, pos_tag = tokens[1], tokens[3]

                    if word in self.PUNCTUATION_MARKS or pos_tag in self.PUNCTUATION_MARKS:
                        pos_tag = "PUNCT"

                    curr.append((word, pos_tag))
        return sents

    def preprocess_sample(self, file_path):
        if self.word2idx is None or self.tag2idx is None:
            raise AssertionError("You must perform preprocessing for a training set first in order to initiate words indexes")

        sents = self.read_file(file_path)

        # build sample and labels by replacing words and tags with matching idx
        # Words never seen before will be replaced by the index of their category
        x = [[self.word2idx.get(w, self.word2idx[self._replace_rare_word(w)]) for w, _ in sent] for sent in sents]
        y = [[self.tag2idx[t] for _, t in sent] for sent in sents]

        # perform padding to structure every sentence example to a defined size
        # use "PADD" symbol as padding value

        x = pad_sequences(maxlen=self.max_seq_len, sequences=x, padding="post", truncating="post",
                          value=self.word2idx["PADD"])
        y = pad_sequences(maxlen=self.max_seq_len, sequences=y, padding="post", truncating="post",
                          value=self.tag2idx["PADD"])

        return x, y
