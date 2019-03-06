import os
import numpy as np
import random

from util import sigmoid, sigmoid_prime


class NeuralNetwork(object):

    def __init__(self, sizes=list(), learning_rate = 0.01, mini_batch_size= 128,
                 epochs=10):
        """Initialize a Neural Network model.
        Parameters
        ----------
        sizes : list, optional
            A list of integers specifying number of neurns in each layer. Not
            required if a pretrained model is used.
        learning_rate : float, optional
            Learning rate for gradient descent optimization. Defaults to 1.0
        mini_batch_size : int, optional
            Size of each mini batch of training examples as used by Stochastic
            Gradient Descent. Denotes after how many examples the weights
            and biases would be updated. Default size is 16.
        """
        # Input layer is layer 0, followed by hidden layers layer 1, 2, 3...
        self.sizes = sizes
        self.num_layers = len(sizes)

        # First term corresponds to layer 0 (input layer). No weights enter the
        # input layer and hence self.weights[0] is redundant.
        self.weights = [np.array([0])] + [np.random.randn(y, x) for y, x in
                                          zip(sizes[1:], sizes[:-1])]

        # Input layer does not have any biases. self.biases[0] is redundant.
        self.biases = [np.random.randn(y, 1) for y in sizes]

        # Input layer has no weights, biases associated. Hence z = wx + b is not
        # defined for input layer. self.zs[0] is redundant.
        self._zs = [np.zeros(bias.shape) for bias in self.biases]

        # Training examples can be treated as activations coming out of input
        # layer. Hence self.activations[0] = (training_example).
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, training_data, validation_data=None):

        for epoch in range(self.epochs):

            random.shuffle(training_data)
            mini_batches = [training_data[k:k + self.mini_batch_size]] for k in range(0, len(training_data), self.mini_batch_size)