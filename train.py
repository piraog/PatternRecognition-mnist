# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from network import Network
import matplotlib.pyplot as plt

import numpy as np
import random
import pickle
import cv2
import os
import time
from mnist import *

# Training parameters
EPOCHS = 10
INIT_LR = 1e-3
BS = 100
IMAGE_DIMS = (28, 28, 1)

#Initialisation
start1 = time.time()
print("Loading images...")

X, image_shape, Y = get_training_set('mnist')
X_test, (_, _), Y_test = get_test_set('mnist')

#input pixels between 0 and 1
X = X.astype(np.float64)
X = (X-X.mean())/263
X = X.reshape(X.shape[0],28,28,1)

X_test = X_test.astype(np.float64)
X_test = (X_test-X_test.mean())/263
X_test = X_test.reshape(X_test.shape[0],28,28,1)

# binarize the labels using scikit-learn's special multi-label binarizer implementation
Y = Y.reshape((len(Y), 1))
ohe_Y = one_hot_encode(Y)

Y_test = Y_test.reshape((len(Y_test), 1))
ohe_Y_test = one_hot_encode(Y_test)

# construct the image generator for data augmentation
trainDataAug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
trainDataAug.fit(X)
end1 = time.time()
print("Loading of images and labels took " + str(end1- start1) + " seconds")


# Creating the network
start2 = time.time()
print("Compiling model...")
model = Network.buildModel(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0])


#using adam instead of gradient descent
model.compile(loss="binary_crossentropy", optimizer= Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS),
	metrics=["accuracy"])

end2 = time.time()
print("Compilation of model took " + str(end2 - start2) + " seconds")

# train the network
start3 = time.time()
print("Training network...")
H = model.fit_generator(
	trainDataAug.flow(X, ohe_Y, batch_size=BS),
	validation_data=(X_test, ohe_Y_test),
	steps_per_epoch=len(X) // BS,
	epochs=EPOCHS, verbose=1)
end3 = time.time()
print("Loading of images took " + str(end3 - start3) + " seconds")

# save the model to disk
print("Serializing model...")
model.save('model')


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch No.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('CNNplot')
plt.show()

