#!/usr/bin/env python

import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE

    return s


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ### YOUR CODE HERE
    ds = s * (1-s)
    ### END YOUR CODE

    return ds


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print (f)
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print (g)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print ("You should verify these results by hand!\n")


def test_sigmoid():
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print ("Running your tests...")
    ### YOUR CODE HERE

    test1_sig = sigmoid(np.array([0, 0, 0, 0]))
    print(test1_sig)
    test1_sig_ans = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(test1_sig, test1_sig_ans, rtol=1e-05, atol=1e-06)
    test1_grad = sigmoid_grad(test1_sig)
    print(test1_grad)
    test1_grad_ans = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.allclose(test1_grad, test1_grad_ans, rtol=1e-05, atol=1e-06)

    test2_sig = sigmoid(np.array([[1, 2], [3, 4], [5, 6]]))
    print(test2_sig)
    test2_sig_ans = np.array([[0.731058579, 0.880797078], [0.952574127, 0.98201379], [0.993307149, 0.997527377]])
    assert np.allclose(test2_sig, test2_sig_ans, rtol=1e-05, atol=1e-06)
    test2_grad = sigmoid_grad(test2_sig)
    print(test2_grad)
    test2_grad_ans = np.array([[0.196611933, 0.104993585], [0.04517666, 0.017662706], [0.006648057, 0.002466509]])
    assert np.allclose(test2_grad, test2_grad_ans, rtol=1e-05, atol=1e-06)

    ### END YOUR CODE


if __name__ == "__main__":
    test_sigmoid_basic()
    test_sigmoid()
