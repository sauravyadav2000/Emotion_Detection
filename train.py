from cgi import test
from ctypes.wintypes import tagRECT
# from msilib import sequence
from operator import mod
from pickletools import optimize
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D, BatchNormalization, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam


from glob import glob
import matplotlib.pyplot as plt

# from tensorflow.keras.optimizers import adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Rescaling the images
TrainingData = ImageDataGenerator(rescale = 1./255)
TestingData = ImageDataGenerator(rescale = 1./255)

#Preprocess all train data
train_generator = TrainingData.flow_from_directory(
    'Data/train',
    target_size = (48, 48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = "categorical"
)

#Preprocess all test data
test_generator = TestingData.flow_from_directory(
    'Data/test',
    target_size = (48, 48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = "categorical"
)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


#Training the model
model_info = model.fit_generator(
    train_generator,
    steps_per_epoch = 28709//64,
    epochs = 20,
    validation_data = test_generator,
    validation_steps = 7178//64
)


#Saving our model in json file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#saving trained model weights in .h5 file
model.save("models/emotion.h5")


