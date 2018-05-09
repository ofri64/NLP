from data import *
import time
from submitters_details import get_details
from tester import verify_hmm_model

import numpy as np
import pandas as pd
import pickle

# For the sake of optimization
_S_ = {}


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print "Start training"
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = {}, {}, {}, {}, {}
    ### YOUR CODE HERE

    for _sent in sents:
        total_tokens += len(_sent)
        sent = [('*', '*'), ('*', '*')] + _sent + [('STOP', 'STOP')]
        for i in xrange(2, len(sent)):
            tri = (sent[i - 2][1], sent[i - 1][1], sent[i][1])
            bi = (sent[i - 1][1], sent[i][1])
            uni = sent[i][1]
            wordtag = sent[i]

            q_tri_counts[tri] = q_tri_counts.get(tri, 0) + 1
            q_bi_counts[bi] = q_bi_counts.get(bi, 0) + 1
            q_uni_counts[uni] = q_uni_counts.get(uni, 0) + 1
            e_tag_counts[uni] = e_tag_counts.get(uni, 0) + 1
            e_word_tag_counts[wordtag] = e_word_tag_counts.get(wordtag, 0) + 1

    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts, lambda1,
                lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))

    ### YOUR CODE HERE

    def q(prevprev, prev, cur):
        """
        Computes linear interpolation probability for a trigram == (prevprev, prev, cur)
        """
        assert lambda1 + lambda2 <= 1

        lambda3 = 1 - lambda1 - lambda2
        tri, bi, uni = (prevprev, prev, cur), (prev, cur), cur

        return sum([lambda1 * q_tri_counts.get(tri, 0) / q_bi_counts.get(tri[:-1], np.inf),
                    lambda2 * q_bi_counts.get(bi, 0) / q_uni_counts.get(bi[0], np.inf),
                    lambda3 * q_uni_counts.get(uni, 0) / total_tokens])

    def e(word, tag):
        return float(e_word_tag_counts.get((word, tag), 0)) / e_tag_counts.get(tag, np.inf)

    def S(i):
        if i < 0:
            return ['*']
        word = sent[i][0]
        if word not in _S_:
            _S_[word] = [t for (w, t) in e_word_tag_counts if w == word]
        return _S_[word]

    n = len(sent)
    bp = {k: {} for k in xrange(n)}
    pi = {k: {} for k in xrange(n)}
    pi[-1] = {('*', '*'): 1}

    for k in xrange(n):
        xk = sent[k][0]
        for v in S(k):  # v == cur
            for u in S(k - 1):  # u == prev
                pi[k][u, v] = -1
                for i, w in enumerate(S(k - 2)):  # w == prevprev
                    p = pi[k - 1][w, u] * q(w, u, v) * e(xk, v)
                    if p > pi[k][u, v]:
                        pi[k][u, v] = p
                        bp[k][u, v] = w

    y = predicted_tags
    u, v = max(pi[n - 1], key=lambda (_u, _v): pi[n - 1][_u, _v] * q(_u, _v, 'STOP'))

    if n == 1:
        y[-1] = v
    else:
        y[-2], y[-1] = u, v
        for k in xrange(n - 3, -1, -1):
            y[k] = bp[k + 2][y[k + 1], y[k + 2]]

    ### END YOUR CODE
    return predicted_tags


def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    ### YOUR CODE HERE

    n_mistakes, n_test_tokens = 0, 0
    for sent in test_data:
        predicted_tags = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                     e_word_tag_counts, e_tag_counts, 0.9, 0.0)

        for i, (_, tag) in enumerate(sent):
            n_mistakes += (tag != predicted_tags[i])

        n_test_tokens += len(sent)

    error = float(n_mistakes) / n_test_tokens
    acc_viterbi = 1 - error
    ### END YOUR CODE

    return str(acc_viterbi)


def evaluate_lambda_values(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                           e_tag_counts, lambda_1, lambda_2):
    acc_viterbi = 0.0
    n_mistakes, n_test_tokens = 0, 0
    for sent in dev_sents:
        predicted_tags = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                     e_word_tag_counts, e_tag_counts, lambda_1, lambda_2)

        for i, (_, tag) in enumerate(sent):
            n_mistakes += (tag != predicted_tags[i])

        n_test_tokens += len(sent)

    error = float(n_mistakes) / n_test_tokens
    acc_viterbi = 1 - error
    return str(acc_viterbi)


def grid_search_lambda(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                           e_tag_counts):
    accuracy_results = {
        'lambda_1': [],
        'lambda_2': [],
        'lambda_3': [],
        'accuracy': []
    }
    for lambda_1 in np.arange(0, 1, 0.1):
        for lambda_2 in np.arange(0, 1 - lambda_1, 0.1):
            lambda_3 = 1 - (lambda_1 + lambda_2)
            accuracy = evaluate_lambda_values(dev_sents, total_tokens, q_tri_counts, q_bi_counts,
                                              q_uni_counts, e_word_tag_counts, e_tag_counts, lambda_1, lambda_2)
            accuracy_results['lambda_1'].append(lambda_1)
            accuracy_results['lambda_2'].append(lambda_2)
            accuracy_results['lambda_3'].append(lambda_3)
            accuracy_results['accuracy'].append(accuracy)
            print "lambda 1: {0}, lambda 2: {1}, lambda 3: {2}, accuracy: {3}".format(
                lambda_1, lambda_2, lambda_3, accuracy)

    results = pd.DataFrame(accuracy_results)
    results.to_csv('../grid_search_res.csv')


def load_pickle(path):
    with open(path, "rb") as input_object:
        return pickle.load(input_object)

if __name__ == "__main__":
    print (get_details())
    start_time = time.time()
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)

    verify_hmm_model(total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                           e_tag_counts)
    print "Dev: Accuracy of Viterbi hmm: " + acc_viterbi

    train_dev_time = time.time()
    print "Train and dev evaluation elapsed: " + str(train_dev_time - start_time) + " seconds"

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                               e_word_tag_counts, e_tag_counts)
        print "Test: Accuracy of Viterbi hmm: " + acc_viterbi
        full_flow_end = time.time()
        print "Full flow elapsed: " + str(full_flow_end - start_time) + " seconds"


    # grid search part

    # pickle_path = "../pickles/"
    # pickle_suffix = ".pkl"
    # dev_sents = load_pickle(pickle_path + 'dev_sents' + pickle_suffix)
    # total_tokens = load_pickle(pickle_path + 'total_tokens' + pickle_suffix)
    # tri_count = load_pickle(pickle_path + 'tri_count' + pickle_suffix)
    # bi_count = load_pickle(pickle_path + 'bi_counts' + pickle_suffix)
    # uni_counts = load_pickle(pickle_path + 'uni_counts' + pickle_suffix)
    # word_tags_counts = load_pickle(pickle_path + 'word_tag_counts' + pickle_suffix)
    # tag_counts = load_pickle(pickle_path + 'tag_counts' + pickle_suffix)
    #
    # grid_search_lambda(dev_sents, total_tokens, tri_count, bi_count, uni_counts, word_tags_counts, tag_counts)
