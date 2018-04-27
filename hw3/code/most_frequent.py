from data import *
from submitters_details import get_details
import tester


def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    ### YOUR CODE HERE

    word_tags_count = {}
    for sent in train_data:
        for token in sent:
            word, tag = token
            if word not in word_tags_count:
                word_tags_count[word] = {}
            word_tags_count[word][tag] = word_tags_count[word].get(tag, 0) + 1

    return {word: max(tags, key=lambda t: tags[t]) for word, tags in word_tags_count.iteritems()}

    ### END YOUR CODE


def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    ### YOUR CODE HERE

    num_samples = 0
    num_mistakes = 0

    for sent in test_set:
        for token in sent:
            word, label = token
            prediction = pred_tags.get(word, 'UNK')
            if label != prediction:
                num_mistakes += 1
            num_samples += 1

    error = float(num_mistakes) / num_samples
    accuracy = 1 - error
    return str(accuracy)

    ### END YOUR CODE


if __name__ == "__main__":
    print (get_details())
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: ", most_frequent_eval(dev_sents, model)

    tester.verify_most_frequent_model(model)

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + most_frequent_eval(test_sents, model)
