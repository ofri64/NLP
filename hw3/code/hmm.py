from data import *
import time
from submitters_details import get_details
from tester import verify_hmm_model

import numpy as np

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print "Start training"
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = {}, {}, {}, {}, {}
    ### YOUR CODE HERE

    c_tri_counts, c_bi_counts, c_uni_counts, c_word_tag_counts, c_tag_counts = {}, {}, {}, {}, {}

    for sent in sents:
        for i in range(len(sent)):

            if i >= 2:
                trigram = (sent[i-2][1], sent[i-1][1], sent[i][1])
                c_tri_counts[trigram] = c_tri_counts.get(trigram, 0) + 1

            if i >= 1:
                bigram = (sent[i-1][1], sent[i][1])
                c_bi_counts[bigram] = c_bi_counts.get(bigram, 0) + 1

            unigram = sent[i][1]
            c_uni_counts[unigram] = c_uni_counts.get(unigram, 0) + 1

            word_tag = (sent[i][0], sent[i][1])
            c_word_tag_counts[word_tag] = c_word_tag_counts.get(word_tag, 0) + 1

            total_tokens += 1

    c_tag_counts = c_uni_counts # the two dictionaries are identical

    for trigram, trigram_count in c_tri_counts.items():
        bigram = trigram[1:]
        q_tri_counts[trigram] = float(trigram_count) / c_bi_counts[bigram]

    for bigram, bigram_count in c_bi_counts.items():
        unigram = bigram[1]
        q_bi_counts[bigram] = float(bigram_count) / c_uni_counts[unigram]

    for unigram, unigram_count in c_uni_counts.items():
        M = total_tokens
        q_uni_counts[unigram] = float(unigram_count) / M

    for word_tag, word_tag_count in c_word_tag_counts.items():
        tag = word_tag[1]
        e_word_tag_counts[word_tag] = float(word_tag_count) / c_tag_counts[tag]

    e_tag_counts = q_uni_counts

    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE

    def q_li(trigram):
        '''
            Compute linear interpolation probability for a trigram
        '''
        assert lambda1 + lambda2 <= 1

        lambda3 = 1 - (lambda1 + lambda2)
        bigram = trigram[1:]
        unigram = bigram[1]

        weighted_li = lambda1 * q_tri_counts.get(trigram, 0) + \
            lambda2 * q_bi_counts.get(bigram, 0) + \
            lambda3 * q_uni_counts.get(unigram, 0)

        return weighted_li

    def get_pi_and_bi_k(k, u, v):
        # Base Case
        if k == 0:
            return 1

        # Recursion with memoization
        key = (k, u, v)
        if key not in pi_dict:
            possible_tags = [get_pi_and_bi_k(k-1, w, u) * q_li((w, u, v)) * e_word_tag_counts.get((sent[k][0], v), 0)
                             for w in tags]
            pi_k = np.max(possible_tags)
            bi_k = np.argmax(possible_tags)
            pi_dict[key] = pi_k
            bp_dict[key] = bi_k
        return pi_dict[key]

    # Initiate variables for viterbi results
    tags = e_tag_counts.keys()
    pi_dict = {}
    bp_dict = {}

    # Run viterbi algorithm
    n = len(sent) - 1
    for _u in tags:
        for _v in tags:
            get_pi_and_bi_k(n, _u, _v)

    # Now we have all the values we need in our dictionary
    # Specifically all the (n, u, v) values
    # pi_n_u_v = filter(lambda (key, score): key[0] == n, *pi_dict.items())
    pi_n_u_v = [(key, value) for key, value in pi_dict.items() if key[0] == n]
    # y_n_minus_1, y_n = np.argmax()


    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    for sent in test_data:
        hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                    e_word_tag_counts, e_tag_counts, 0.35, 0.45)
    ### END YOUR CODE

    return str(acc_viterbi)

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
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts)
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