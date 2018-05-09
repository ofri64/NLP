from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
from submitters_details import get_details

import numpy as np
from itertools import product
import os.path
import pickle

# For the sake of optimization
_S_ = {}
_tag_to_idx_dict_ = {}


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    features.update({
        'next_word': next_word,
        'prev_word': prev_word,
        'prevprev_word': prevprev_word,
        'prev_tag': prev_tag,
        'prevprev_tag': prevprev_tag,
        'prev_2_tags': prevprev_tag + prev_tag
    })

    for i in xrange(min(5, len(curr_word) + 1)):
        features['prefix{0}'.format(i)] = curr_word[:i]
        features['suffix{0}'.format(i)] = curr_word[-i:]
    ### END YOUR CODE
    return features


def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1],
                                 prevprev_token[1])


def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)


def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greeedy(sent, logreg, vec, index_to_tag_dict):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))

    ### YOUR CODE HERE
    predicted_sent = [(word, '<untagged>') for word, _ in sent]

    for k, word in enumerate(sent):
        features = extract_features(predicted_sent, k)
        vectorized_features = vectorize_features(vec, features)
        index_to_tag = logreg.predict(vectorized_features)[0]
        predicted_tags[k] = index_to_tag_dict[index_to_tag]
        predicted_sent[k] = (word, predicted_tags[k])
    ### END YOUR CODE
    return predicted_tags


def memm_viterbi(sent, logreg, vec, index_to_tag_dict):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE

    def S(i):
        if i < 0:
            return ['*']
        _word = sent[i][0]
        return _S_[_word]

    def logregprobs(k):
        features_list = []
        features_idxs = {}

        # extracting defualt features with true tags
        default_features = extract_features(sent, k)

        # for each pair of tags (prevprev = t, prev = u),
        # create features dict which uses these, as they were predicted
        for i, (t, u) in enumerate(product(S(k - 2), S(k - 1))):
            features_idxs[t, u] = i
            t_u_features = dict(default_features)
            t_u_features.update({
                'prev_tag': u,
                'prevprev_tag': t,
                'prev_2_tags': t + u
            })
            features_list.append(t_u_features)

        vectorized_features_list = vec.transform(features_list)
        probs_matrix = logreg.predict_log_proba(vectorized_features_list)

        return {(t, u): probs_matrix[i] for (t, u), i in features_idxs.iteritems()}

    n = len(sent)
    bp = {k: {} for k in xrange(n)}
    pi = {k: {} for k in xrange(n)}
    pi[-1] = {('*', '*'): 0}

    # iterate through sentence and use matrix probs to define relevant q()
    for k in xrange(n):
        q = logregprobs(k)
        for v in S(k):  # v == cur
            v_idx = _tag_to_idx_dict_[v]
            for u in S(k - 1):  # u == prev
                pi_opt, bp_opt = -np.inf, None
                for i, t in enumerate(S(k - 2)):  # t == prevprev
                    p = pi[k - 1][t, u] + q[t, u][v_idx]  # addition because we use log probs
                    if p > pi_opt:
                        pi_opt = p
                        bp_opt = t

                pi[k][u, v] = pi_opt
                bp[k][u, v] = bp_opt

    # Dynamically store all y values
    y = predicted_tags
    u, v = max(pi[n - 1], key=pi[n - 1].get)

    if n == 1:
        y[-1] = v
    else:
        y[-2], y[-1] = u, v
        for k in xrange(n - 3, -1, -1):
            y[k] = bp[k + 2][y[k + 1], y[k + 2]]

    ### END YOUR CODE
    return predicted_tags


def should_add_eval_log(sentene_index):
    if sentene_index > 0 and sentene_index % 10 == 0:
        if sentene_index < 150 or sentene_index % 200 == 0:
            return True

    return False


def memm_eval(test_data, logreg, vec, index_to_tag_dict):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    greedy_correct = 0.0
    viterbi_correct = 0.0
    total_words = 0.0

    mistakes = []
    should_log_mistakes = True
    max_log_mistakes = 50

    for i, sen in enumerate(test_data):

        ### YOUR CODE HERE
        ### Make sure to update Viterbi and greedy accuracy
        n = len(sen)
        total_words += n

        greedy_predictions = memm_greeedy(sen, logreg, vec, index_to_tag_dict)
        viterbi_predictions = memm_viterbi(sen, logreg, vec, index_to_tag_dict)
        real_predictions = [t for (w, t) in sen]

        greedy_correct += sum(real_predictions[i] == greedy_predictions[i] for i in range(n))
        viterbi_correct += sum(real_predictions[i] == viterbi_predictions[i] for i in range(n))

        # Store first 50 mistakes, only relate to sentences with >=2 mistakes
        if should_log_mistakes:
            viterbi_incorrect = [(sen[idx], viterbi_predictions[idx]) for idx in range(n)
                                 if real_predictions[idx] != viterbi_predictions[idx]]
            if len(viterbi_incorrect) >= 2:
                mistakes.append((i, viterbi_incorrect))
            if len(mistakes) >= max_log_mistakes:
                should_log_mistakes = False

        acc_greedy = greedy_correct / total_words
        acc_viterbi = viterbi_correct / total_words
        ### END YOUR CODE

        if should_add_eval_log(i):
            if acc_greedy == 0 and acc_viterbi == 0:
                raise NotImplementedError
            eval_end_timer = time.time()
            print str.format("Sentence index: {} greedy_acc: {}    Viterbi_acc:{} , elapsed: {} ", str(i),
                             str(acc_greedy), str(acc_viterbi), str(eval_end_timer - eval_start_timer))
            eval_start_timer = time.time()

    # Mistakes log:
    print "Mistakes log:"
    print "-------------"
    for mistake in mistakes:
        index = mistake[0]
        comparisons = mistake[1]
        sentence = [w for (w, t) in test_data[index]]
        print "Sentence: " + str(sentence)
        for comp in comparisons:
            print "real value: " + str(comp[0])
            print "viterbi: " + str(comp[1])

    return str(acc_viterbi), str(acc_greedy)


def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    print (get_details())
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    # Optimization - Save training set tags dict
    print "Optimizing tags"
    _tag_to_idx_dict_ = tag_to_idx_dict
    for sent in train_sents:
        for word, tag in sent:
            if word not in _S_:
                _S_[word] = set()
            _S_[word].add(tag)

    _S_ = {w: list(s) for w, s in _S_.iteritems()}
    print "Done"
    # End of optimization

    # The log-linear model training.
    # NOTE: this part of the code is just a suggestion! You can change it as you wish!

    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    # Use pickle to store existing logreg
    filename = 'logreg.sav'
    logreg = None
    if os.path.isfile(filename):
        print "Fitting..."
        logreg = pickle.load(open(filename, 'rb'))
    else:
        logreg = linear_model.LogisticRegression(
            multi_class='multinomial', max_iter=128, solver='lbfgs', C=1, verbose=1, n_jobs=4)
        print "Fitting..."
        start = time.time()
        model = logreg.fit(train_examples_vectorized, train_labels)
        end = time.time()
        print "End training, elapsed " + str(end - start) + " seconds"
        pickle.dump(model, open(filename, 'wb'))
        # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print "Start evaluation on dev set"
    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec, index_to_tag_dict)
    end = time.time()
    print "Dev: Accuracy greedy memm : " + acc_greedy
    print "Dev: Accuracy Viterbi memm : " + acc_viterbi

    print "Evaluation on dev set elapsed: " + str(end - start) + " seconds"
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        start = time.time()
        print "Start evaluation on test set"
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec, index_to_tag_dict)
        end = time.time()

        print "Test: Accuracy greedy memm: " + acc_greedy
        print "Test:  Accuracy Viterbi memm: " + acc_viterbi

        print "Evaluation on test set elapsed: " + str(end - start) + " seconds"
        full_flow_end = time.time()
        print "The execution of the full flow elapsed: " + str(full_flow_end - full_flow_start) + " seconds"
