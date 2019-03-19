#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# make the number scale between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())    ##input layer
model.add(tf.keras.layers.Dense(200, activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(50, activation= tf.nn.relu))

# output layer that contains the number of classfications
model.add(tf.keras.layers.Dense(10, activation= tf.nn.softmax))


# In[2]:


# always try to minimize loss
# 'adam' is one of the most common ones
model.compile(optimizer='adam',  loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],  lr=0.01)
model.fit(x_train, y_train, epochs=10)


validation_loss, validation_accuracy = model.evaluate(x_test, y_test)
print("Validation loss", validation_loss)
print("Validation accuracy:",validation_accuracy)

# Show the figure in binary
# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()
model.summary()


# In[20]:


# To save a mdoel and make predictions
# Save not working for now. Need to modify "Flatten()"
# model.save('Lecture_one.model')
# new_model = tf.keras.models.load_model('Lecture_one.model')
# predictions = new_model.predict([x_test])
# print(np.argmax(predictions[0]))

prediction = model.predict([x_test])

correct = 0
for i in range(y_test.size):
    predicted = np.argmax(prediction[i])
    
    if predicted == y_test[i]:
        correct += 1

print("Validation rate:", correct/y_test.size)


# In[ ]:




