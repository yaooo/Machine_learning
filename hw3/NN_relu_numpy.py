#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
import pickle


# In[2]:


# Util functions:
def load(file_name):
    with open(file_name,'rb') as f:
        mnist = pickle.load(f)
    training_images, training_labels, testing_images, testing_labels = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
    
    # Normalize the images
    training_images.astype('float32')
    testing_images.astype('float32')
    training_images = training_images/255.
    testing_images = testing_images/255.
    return training_images, training_labels, testing_images, testing_labels

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def softmax_prime(z):
    return softmax(z) * (1 - softmax(z))

def relu(z):
    return np.maximum(z, 0)

# return 1 or 0
def relu_prime(z):
    return (z > 0)

#     Forward equations:
#     z1 = x.w1+b1
#     a1 = relu(z1)

#     z2 = a1.w2+b2
#     a2 = relu(z2)

#     z3 = a2.w3+b3
#     a3 = softmax(z3)

#     Back propagation equations:
#     a3_delta = a3-y    

#     z2_delta = a3_delta.w3.T
#     a2_delta = z2_delta.relu_prime(a2)

#     z1_delta = a2_delta.w2.T
#     a1_delta = z1_delta.relu_prime(a1)


# In[9]:


class Dense:
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        # better initialize the weights      
        self.weights = np.random.randn(input_units, output_units)*np.sqrt(2/(output_units+input_units))
        self.biases = np.zeros(output_units) + 0.01
        
    def forward(self,input):
        return np.dot(input, self.weights) + self.biases
      
    def backward(self,input,grad_output):
        grad_input = np.dot(grad_output,np.transpose(self.weights))

        grad_weights = np.transpose(np.dot(np.transpose(grad_output),input))
        grad_biases = np.sum(grad_output, axis = 0)
        
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases
        return grad_input


# In[10]:


class ReLU:
    def __init__(self):
        pass
    
    def forward(self, input):
        return relu(input)

    def backward(self, input, output):
        return output*relu_prime(input)


# In[11]:


def softmax_crossentropy(X, y):
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss

def grad_softmax_crossentropy(X, y):
    ones_for_answers = np.zeros_like(X)
    ones_for_answers[np.arange(len(X)),y] = 1
    
    p = np.exp(X) / np.exp(X).sum(axis=-1,keepdims=True)
    return (- ones_for_answers + p) / X.shape[0]


# In[12]:


def forward(network, X):
    forward_propagation = []
    for i in range(len(network)):
        X = network[i].forward(X)
        forward_propagation.append(X)
    return forward_propagation

def predict(network,X):
    indice = forward(network,X)[-1]
    return indice.argmax(axis=-1)

def train(network,X,y):
    # Get the layer activations
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
    out = layer_activations[-1]
    
    # Compute the loss and the initial gradient
    loss = softmax_crossentropy(out,y)
    loss_grad = grad_softmax_crossentropy(out,y)
    
    for i in range(1, len(network)):
        loss_grad = network[len(network) - i].backward(layer_activations[len(network) - i - 1], loss_grad)
    


# In[20]:


from random import shuffle
from time import time
def main():
    # Load the dataset
    file_name = 'data/mnist.pkl'

    X_train, y_train, X_test, y_test = load(file_name)
    
    print('training_image shape:', X_train.shape)
    print('Number of images in training set:', X_train.shape[0])
    print('Number of images in testing set:', X_test.shape[0])

    network = []
    network.append(Dense(784,200))
    network.append(ReLU())
    network.append(Dense(200,50))
    network.append(ReLU())
    network.append(Dense(50,10))
    batchsize = 32
    train_log = []
    val_log = []
    time_log = []
    
    for epoch in range(10):
        
        init_time = time()
        for i in range(0, len(X_train) - batchsize + 1, batchsize):
            interval = slice(i, i + batchsize)
            x_batch,y_batch =  X_train[interval], y_train[interval]
            train(network,x_batch,y_batch)
            
        time_log.append(time() - init_time)
        
        acc = predict(network,X_train)==y_train
        train_log.append(np.count_nonzero(acc == 1)/len(y_train))
        
        acc1 = predict(network,X_test)==y_test
        val_log.append(np.count_nonzero(acc1 == 1)/len(y_test))
        
        print("\nEpoch",epoch)
        print("Training accuracy:",train_log[-1])
        print("Validation accuracy:",val_log[-1])
        print("Training Time:", time_log[-1], "sec")
    
    
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




