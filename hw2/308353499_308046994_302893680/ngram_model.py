#!/usr/local/bin/python
# -*- coding: utf-8 -*-

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


def ngrams(sentence, n):
    if n == 1:
        return sentence[1:]

    iters = itertools.tee(sentence, n)
    forward = 0
    for it in iters:
        for _ in xrange(forward):
            next(it, None)
        forward += 1

    return zip(*iters)


# n_asterisks is the number of asterisks in the beginning of each sentence in the dataset
def count_ngrams(dataset, n):
    ngram_counts = {}

    for sentence in dataset:
        for ngram in ngrams(sentence, n):
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    return ngram_counts


def bigrams(sentence):
    return ngrams(sentence, 2)


def trigrams(sentence):
    return ngrams(sentence, 3)


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

    trigram_counts = count_ngrams(dataset, 3)
    bigram_counts = count_ngrams(dataset, 2)
    unigram_counts = count_ngrams(dataset, 1)
    token_count = np.sum([len(s) - 2 for s in dataset])

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
        return np.float128(trigram_counts.get(trigram, 0)) / bigram_counts[bigram]

    def bigram_probability(bigram):
        unigram = bigram[0]
        if unigram not in unigram_counts:
            return 0
        return np.float128(bigram_counts.get(bigram, 0)) / unigram_counts[unigram]

    def unigram_probability(unigram):
        return np.float128(unigram_counts.get(unigram, 0)) / train_token_count

    def linear_interpolation(trigram):
        lambda3 = 1 - lambda1 - lambda2
        bigram = trigram[1:]
        unigram = trigram[-1]
        return np.sum([lambda1 * trigram_probability(trigram),
                       lambda2 * bigram_probability(bigram),
                       lambda3 * unigram_probability(unigram)])

    # q((x, y ,z)) is the (empirical) probability to see x given that the words before it was (y, z)
    # in this case, q is a linear interpolation of unigram, bigram and trigram probablity models
    q = linear_interpolation

    def sentence_probability(s):
        return np.prod([q(trigram) for trigram in trigrams(s)])

    # p(s) is the (empirical) probability to see a sentence s
    p = sentence_probability

    # len(s) - 2 because each sentence starts with two *
    M = np.float128(np.sum([len(s) - 2 for s in eval_dataset]))
    l = np.sum([np.log2(p(s)) for s in eval_dataset]) / M
    perplexity = np.power(2, -l)

    ### END YOUR CODE
    return perplexity


def grid_search_params(validation_set, trigram_counts, bigram_counts, unigram_counts, token_count,
                       resolution, epsilon, step, left=0, right=1, verbose=False):
    l1_left = l2_left = left
    l1_right = l2_right = right

    global_min_pair = (0, 0)
    global_min_perp = np.inf

    l1d, l2d = {}, {}

    while True:
        min_perp = global_min_perp
        min_pair = global_min_pair
        l1_grid = np.arange(l1_left, l1_right, resolution)
        l2_grid = np.arange(l2_left, l2_right, resolution)
        for l1 in l1_grid:
            l2d = {}
            for l2 in l2_grid:
                if l1 + l2 <= 1:
                    l2d[l2] = perplexity = evaluate_ngrams(validation_set, trigram_counts, bigram_counts,
                                                           unigram_counts,
                                                           token_count, l1, l2)
                    if perplexity < min_perp:
                        min_perp = perplexity
                        min_pair = (l1, l2)

                    if verbose:
                        print 'perplexity = {0}; 𝝀1 = {1}, 𝝀2 = {2}'.format(perplexity, l1, l2)

            l1d[l1] = l2d

        if global_min_perp - min_perp < epsilon:
            global_min_perp = min_perp
            global_min_pair = min_pair
            return global_min_perp, global_min_pair, l1d

        global_min_perp = min_perp
        global_min_pair = min_pair

        # if verbose:
        #     print "global min perplexity is: " + str(global_min_perp)
        #     print "global opt lambda values: " + str(global_min_pair)

        l1, l2 = min_pair
        l1_left, l1_right = l1 - resolution, l1 + resolution
        l2_left, l2_right = l2 - resolution, l2 + resolution
        resolution /= step


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
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE

    print

    perp, (l1, l2), df = grid_search_params(S_dev, trigram_counts, bigram_counts, unigram_counts,
                                            token_count,
                                            resolution=0.1, epsilon=1.0, step=2.0, verbose=True)

    print pd.DataFrame(df)
    print "best params are: perplexity = {0}; 𝝀1 = {1}, 𝝀2 = {2}".format(perp, l1, l2)

    ### END YOUR CODE


if __name__ == "__main__":
    test_ngram()
