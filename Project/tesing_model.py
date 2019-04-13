#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Data
import numpy as np
import pandas as pd
import os
import csv
import xml.etree.ElementTree as ET

# Framework
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

# Imaging
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.333)
sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))

dict_characters = {1:'NEUTROPHIL',2:'EOSINOPHIL',3:'MONOCYTE',4:'LYMPHOCYTE'}


# Can modify the input model name here
model_name = 'acc_95_83.h5'


# In[11]:


from tqdm import tqdm
from tensorflow.keras.utils import to_categorical


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


# In[12]:


# Normalize the dataset
X_train=np.array(X_train)
X_train=X_train/255.0

X_test=np.array(X_test)
X_test=X_test/255.0


# In[13]:


# Load the trained model before plotting

from tensorflow.keras.models import load_model
model = load_model(model_name)
print("\nPrinting model summary...")
model.summary()


# In[14]:


validation_loss, validation_accuracy = model.evaluate(X_test, y_testHot)
print("\n\nTesting set validation loss", validation_loss)
print("Testing set Validation accuracy:",validation_accuracy)


# In[15]:


validation_loss, validation_accuracy = model.evaluate(X_train, y_trainHot)
print("\nTraining set validation loss", validation_loss)
print("Training set validation accuracy:",validation_accuracy)


# In[16]:


model.get_config()


# In[17]:


from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
test_result = model.predict(X_test)
x1 = np.argmax(test_result, axis=1)
x2 = np.argmax(y_testHot, axis = 1)
cm = confusion_matrix(x2, x1)
cm = cm/np.sum(cm)
print("\nPrinting the confusion matrix...\n", cm)


# In[18]:


names = ['NEUTROPHIL', 'EOSINOPHIL', 'MONOCYTE', 'LYMPHOCYTE']
df_cm = pd.DataFrame(cm, index = names,
                  columns = names)
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
plt.title("Normalized Confusion Matrix")
plt.savefig("Normalized_CM.png")
plt.show()


# In[ ]:





# In[ ]:




