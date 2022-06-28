#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Description: This program classifies clothes form the Fashion MNIST data set using artificial neural networks


# In[5]:


#importing libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


#load the data set
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images,test_labels) = fashion_mnist.load_data()


# In[13]:


#view a training image
img_index = 0
img = train_images[img_index]
print("Image label: ",train_labels[img_index])
plt.imshow(img)


# In[15]:


#print the shape
print(train_images.shape)
print(test_images.shape)
# 60k train and 10k test images with 28x28px 


# In[20]:


#create the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
#faltten for input layer, how many neurons we have with hidden layer. 10 for uniqe labels.
#activation with 2 different way


# In[23]:


#compile the model
model.compile(
    optimizer = tf.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)


# In[25]:


#train the model
model.fit(train_images, train_labels, epochs=5,batch_size=32)


# In[26]:


#Evaluate the model
model.evaluate(test_images,test_labels)


# In[53]:


#make a prediction
predictions = model.predict(test_images[0:5])

#print the predicted labels
print(np.argmax(predictions,axis=1))
#the label for the first image is 9, second is 2 and so on
#print the actual label values
print(test_labels[0:15])
#its matching with test yey! Not for 15 btw


# In[54]:


#print the first 5 images
for i in range(0,5):
    plt.imshow(test_images[i])
    plt.show()
    print(test_labels[i])


# In[ ]:




