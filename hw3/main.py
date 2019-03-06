import math
import numpy as np
# from download_mnist import load
import operator
import time
from numpy import linalg as LA
from urllib import request
import gzip
import pickle
from network import *

file_name = 'data/mnist.pkl'
def load():
    with open(file_name,'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


# Load the training and testing dataset

train_images, train_labels, test_images, test_labels = load()

train_images = train_images.reshape(60000,28*28)
test_images  = test_images.reshape(10000,28*28)
# convert the data type from unint8 to float
train_images = train_images.astype(float)
test_images = test_images.astype(float)

training_data = zip(train_images, train_labels)
validation_data = zip(test_images, test_labels)


# FINAL GOAL：
# FC(testing_input, training_dataset, training)


# Set up the network config
# Its structure is 784‐200‐50‐10.
# 784 means the input layer has 784 input neurons. This is because each image in MNIST dataset is 28x28.
# 200 and 50 are the number of neurons in hidden layers.
# 10 is the number of neurons in output layer since there are 10 types of digits.
# The two hidden layers are followed by ReLU layers.
# The output layer is a softmax layer.

layers = [784,20,50,10]

#initialize learning rate
learning_rate = 0.01

#initialize mini batch size
mini_batch_size = 128

#initialize epoch
epochs = 10

#initialize neuralnet
nn = NeuralNetwork(layers, learning_rate, mini_batch_size, epochs)

#training neural network
# nn.fit(train_images, train_labels)

#testing neural network
# accuracy = nn.validate(test_data) / 100.0
# print("Test Accuracy: " + str(accuracy) + "%")

#save the model
# nn.save()
