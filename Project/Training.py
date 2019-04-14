#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data
import numpy as np
import pandas as pd
import os
import csv
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Framework
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

# Imaging
import cv2
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# import seaborn as sns


from tensorflow.python.client import device_lib
device_lib.list_local_devices()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))


# In[2]:


from zipfile import ZipFile
file_name = 'data.zip'

if not os.path.isdir("blood-cells"):
    with ZipFile(file_name, 'r') as zip:
        zip.extractall()
        print("Done")


# In[3]:



def get_data(src_folder):
    imgs = []
    labels = []
    names = ['NEUTROPHIL', 'EOSINOPHIL', 'MONOCYTE', 'LYMPHOCYTE']
    for name in names:
        label = names.index(name)+1
        for img_name in tqdm(os.listdir(src_folder + name)):
            path = os.path.join(src_folder, name, img_name)
            img_file = cv2.imread(path)

            if img_file is not None:
                img_file = cv2.resize(img_file, (80, 80))
                img_arr = np.asarray(img_file)
                imgs.append(img_arr)
                labels.append(label)
    return np.asarray(imgs), np.asarray(labels)

X_train, y_train = get_data('blood-cells/dataset2-master/images/TRAIN/')
X_test, y_test = get_data('blood-cells/dataset2-master/images/TEST/')

y_trainHot = to_categorical(y_train, num_classes = 5)
y_testHot = to_categorical(y_test, num_classes = 5)


# In[4]:



# Normalize the dataset

X_train=np.array(X_train)
X_train=X_train/255.

X_test=np.array(X_test)
X_test=X_test/255.

# plotHistogram(X_train[1])
# print(X_train[1].size/3/80)


# In[5]:


# Tensorboard Usage
# ##########################################################################################
# # Command for calling tensorboard: 
# #tensorboard --logdir=logs/ --host localhost --port 8088
# ##########################################################################################


# In[9]:


import time
num_category = len(y_trainHot[0])
image_shape = X_train[0].shape

NAME = "logs/{}-{}".format("testing", int(time.time()))

tensorboard = TensorBoard(log_dir=NAME)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=image_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3)))
model.add(LeakyReLU(alpha = 0.05))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dropout(0.5))


model.add(Dense(num_category))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True) 

history = model.fit_generator(datagen.flow(X_train, y_trainHot, batch_size=32),  
                              validation_data = (X_test, y_testHot),
                              epochs= 30, 
                              callbacks = [tensorboard])
model.save("model/"+ NAME +' .h5')


# In[10]:


model.save("my_model.h5")
print("Saved the model as my_model.h5\n")

# In[11]:


# model.evaluate(X_test, y_testHot)


# In[ ]:




