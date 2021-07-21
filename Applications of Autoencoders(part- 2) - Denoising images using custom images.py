# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 00:48:12 2021

@author: abc
"""

"""

Applications of Autoencoders


"""

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
from keras.preprocessing.image import img_to_array
from tqdm import tqdm   #for visualization

np.random.seed(42)

SIZE = 320

noisy_data=[]
path1 = ""
files = os.listdir(path1)
for i in tqdm(files):
    img = cv2.imread(path1 + '/' +i, 0) #Change 0 to 1 for color image
    img = cv2.resize(img, (SIZE,SIZE))
    noisy_data.append(img_to_array(img))
    
    
clean_data=[]
path2 =''
files = os.listdir(path2)
for i in tqdm(files):
    img = cv2.imread(path2+'/'+i, 0) #change 0 t0 1 for color image
    img = cv2.resize(img, (SIZE,SIZE))
    clean_data.append(img_to_array(img))
    
noisy_train = np.reshape(noisy_data, (len(noisy_data), SIZE, SIZE, 1))
noisy_train = noisy_train.astype('float32') / 255.

clean_train = np.reshape(clean_data, (len(clean_data), SIZE, SIZE, 1))
clean_train = clean_train.astype('float32') / 255.

#Displaying images with noise
plt.figure(figsize=(10,2))
for i in range(1,4):
    ax = plt.subplot(1,4, i)
    plt.imshow(noisy_train[i].reshape(SIZE, SIZE), cmap="binary")
plt.show()

#Displaying clean images
plt.figure(figsize=(10,2))
for i in range(1,4):
    ax = plt.subplot(1, 4, i)
    plt.imshow(clean_train[i].reshape(SIZE, SIZE), cmap="binary")
plt.show()

model = Sequential()
model.add(Conv2D(32, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2), padding="same"))
model.add(Conv2D(8, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2), padding="same"))
model.add(Conv2D(8, (3,3), activation="relu", padding="same"))

model.add(MaxPooling2D((2,2), padding="same"))

model.add(Conv2D(8, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(8, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(1, (3,3), activation="relu", padding="same"))

model.compile(optimizer= " adam", loss='mean_squared_error', metrics=['accuracy'])

model.summary()

#split our data into training and testing dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(noisy_train, clean_train,
                                                    test_size=0.20, random_state=0)

#Fit the model
model.fit(x_train, y_train, epochs=10, batch_size=8, shuffle=True, verbose=1, validation_split=0.1)

#Find accuracy of our model
print("Test_Accuracy : (:.2f)%".format(model.evaluate(np.array(x_test), np.array(y_test))[1]*100))

#save our model
model.save('Denoising_autoencoder.model')

#Predict our model
no_noise_img = model.predict(x_test)

#Let's visualize our model
plt.imshow(no_noise_img[i].reshape(SIZE, SIZE), cmap="gray")

