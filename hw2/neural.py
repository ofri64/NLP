#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive

def forward(data, label, params, dimensions):
    """
    runs a forward pass and returns the probability of the correct word for eval.
    label here is an integer for the index of the label.
    This function is used for model evaluation.
    """
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Compute the probability
    ### YOUR CODE HERE: forward propagation

    z1 = np.dot(data, W1) + b1
    h = sigmoid(z1)
    z2 = np.dot(h, W2) + b2
    y_hat = softmax(z2)

    return y_hat[label]

    ### END YOUR CODE

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation

    z1 = np.dot(data, W1) + b1 # z is an M x H matrix, row for each batch sample
    h = sigmoid(z1) # h is also an M x H matrix, apply sigmoid on each matrix element
    z2 = np.dot(h, W2) + b2 # z2 is an M x Dy matrix
    y_hat = softmax(z2) # y_pred is also an M x Dy matrix

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation

    cost = -np.sum(labels * np.log(y_hat))

    delta_2 = y_hat - labels # an M x Dy matrix
    delta_1 = np.dot(delta_2, W2.transpose()) * sigmoid_grad(h) # an M x H matrix

    gradb1 = np.sum(delta_1, axis=0) # 1 x H vector
    gradb2 = np.sum(delta_2, axis=0) # 1 X Dy vector

    gradW1 = np.dot(data.T, delta_1) # Dx x H matrix
    gradW2 = np.dot(h.T, delta_2) # H x M matrix

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    N = 5
    dimensions = [5, 2, 5]
    data = np.array([
        [1, 2, 1, 3, 1],
        [1, 2, 1, 1, 0],
        [0, 1, 1, 0, 1],
        [-1, 0, -1, 1, 1],
        [0, 0, -1, 1, 0]
    ])

    labels = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],

    ])

    W1 = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 0]])
    b1 = np.array([[0, 1]])
    W2 = np.array([[1, 2, 1, 1, 1], [0, 1, 2, 1, 0]])
    b2 = np.array([[0, 1, 1, 0, 0]])

    params = np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
