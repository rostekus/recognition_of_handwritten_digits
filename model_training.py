#Import all Necessary Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Lambda, MaxPooling2D, Flatten, BatchNormalization, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import numpy as np
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# batch size and number of epochs
BATCH_SIZE = 32
EPOCHS = 5

#importing dataset, 28x28 images of digits 
mnist = fetch_openml('mnist_784')

#unpacking data
X , y = mnist.data, mnist.target

# converting string into int
y = y.astype(np.short)

# Reshape image in 3 dimensions
# canal = 1 for gray scale
X = X.reshape(-1,28,28,1)

# Scaling numbers [0,1], normalization
X = tf.keras.utils.normalize(X, axis = 1)

# Split the train and the test set
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.3, random_state = 42)

early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

# Sequential Model
model =tf.keras.models.Sequential()
model.add(Conv2D(filters=64, kernel_size=3, input_shape = (28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size = 3, activation='relu'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'], 
              callbacks=[early_stopping_monitor],)

# Training model
model = model.fit(X_train, y_train,batch_size=BATCH_SIZE,
          epochs=EPOCHS, validation_split=0.2)

#Saving model to json file
with open('model.h5', 'wb') as f:
    pickle.dump(model.history, f)