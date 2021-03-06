#!/usr/bin/env python

import numpy as np
import random

from q1b_softmax import softmax
from q1e_gradcheck import gradcheck_naive
from q1d_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    rows_norm = np.linalg.norm(x, axis=1).reshape(x.shape[0], 1)
    x = x / rows_norm
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print x
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word vector
    grad -- the gradient with respect to all the other word vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE

    output_products_vector = np.dot(outputVectors, predicted) # vector of uoT * vc for o = 1,2, ... ,W
    output_probabilities_vector = softmax(output_products_vector)  # vector of p(o|c) for o = 1,2, ... ,W
    u_o = outputVectors[target]

    cost = -np.log(output_probabilities_vector[target]) # -log(p(oi|c))

    gradPred = -u_o + np.sum(output_probabilities_vector[:, np.newaxis] * outputVectors, axis=0) # returning as ndarray shape=(|v|,1)

    target_indicator = np.zeros(outputVectors.shape[0]) # shape of vector is number of words in corpus
    target_indicator[target] = -1 # create a vector with 0 in every index expect for target index (o index)
    grad = predicted * (target_indicator + output_probabilities_vector)[:, np.newaxis]

    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    output_products_vector = np.dot(outputVectors, predicted) # vector of uoT * Vc for o = 1,2, ... ,W
    output_sigmoid_vector = sigmoid(output_products_vector)  # vector of sig(uoT * Vc) for o = 1,2, ... ,W
    output_minus_sigmoid_vector = 1 - output_sigmoid_vector

    cost = -np.log(output_sigmoid_vector[target]) -np.sum(np.log(output_minus_sigmoid_vector[indices[1:]]))
    # cost = -log(sig(uoT * Vc) - sum_k[log(sig(-ukT * Vc))]

    grad_pred_max_part = -output_minus_sigmoid_vector[target] * outputVectors[target] # -sig(uoT * Vc) * uo
    grad_pred_neg_samp_part = outputVectors[indices[1:]] * output_sigmoid_vector[indices[1:]][:, np.newaxis] # matrix with k rows of sig(ukT * Vc) * uk
    grad_pred_sum_neg_samp = np.sum(grad_pred_neg_samp_part, axis=0) # sum the k vectors
    gradPred = grad_pred_max_part + grad_pred_sum_neg_samp

    grad = np.zeros(outputVectors.shape) # besides u0 and uk's gradient of rest is zero
    grad[target] = (output_sigmoid_vector[target] - 1) * predicted # (grad(u0) = sig(u0T * Vc0) -1) * Vc
    for k in indices[1:]: # k is small ~10 maximum so this is still efficient
        grad[k] += output_sigmoid_vector[k] * predicted # grad(uk) = sig(ukT * Vc) * Vc

    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE

    center_word_index = tokens[currentWord]
    vc = inputVectors[center_word_index]
    for output_word in contextWords:
        output_word_index = tokens[output_word]
        output_word_cost, vc_grad, output_grads = word2vecCostAndGradient(vc, output_word_index, outputVectors, dataset)
        cost += output_word_cost
        gradIn[center_word_index] += vc_grad
        gradOut += output_grads

    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
