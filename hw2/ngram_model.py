#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv
import itertools

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                      index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)


def bigrams(sentence):
    a, b = itertools.tee(sentence)
    next(b, None)
    return zip(a, b)


def trigrams(sentence):
    a, b, c = itertools.tee(sentence, 3)
    next(c, None)
    next(c, None)
    next(b, None)
    return zip(a, b, c)


def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0

    ### YOUR CODE HERE
    for sentence in dataset:
        for unigram in sentence:
            unigram_counts[unigram] = unigram_counts.get(unigram, 0) + 1
            token_count += 1

        for bigram in bigrams(sentence):
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        for trigram in trigrams(sentence):
            trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1


    ### END YOUR CODE
    return trigram_counts, bigram_counts, unigram_counts, token_count


def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    ### YOUR CODE HERE

    def trigram_probability(trigram):
        bigram = trigram[:-1]
        if bigram not in bigram_counts:
            return 0
        return float(trigram_counts.get(trigram, 0)) / bigram_counts[bigram]

    def bigram_probability(bigram):
        unigram = bigram[0]
        if unigram not in unigram_counts:
            return 0
        return float(bigram_counts.get(bigram, 0)) / unigram_counts[unigram]

    def unigram_probability(unigram):
        return float(unigram_counts.get(unigram, 0)) / train_token_count

    def linear_interpolation(trigram):
        lambda3 = 1 - lambda1 - lambda2
        bigram = trigram[:-1]
        unigram = trigram[0]
        return np.sum([
            lambda1 * trigram_probability(trigram),
            lambda2 * bigram_probability(bigram),
            lambda3 * unigram_probability(unigram)
        ])

    text_log_like = 0
    for sentence in eval_dataset:
        sentence_prob = 1

        # compute p(si)
        for trigram in trigrams(sentence):
            q_li = linear_interpolation(trigram)
            sentence_prob *= q_li

        # add log2(p(si)) to sum
        text_log_like += np.log2(sentence_prob)

    M = len(eval_dataset)
    perplexity = np.power(2, -text_log_like / M)
    ### END YOUR CODE
    return perplexity


def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    # Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)

    print sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)[:100]


    # perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    # print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    ### END YOUR CODE


if __name__ == "__main__":
    test_ngram()
