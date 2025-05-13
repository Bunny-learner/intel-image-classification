import pandas as pd
import seaborn as sns
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

print("training has started,")

TRAIN_DIR = 'train_set'
IMG_WIDTH, IMG_HEIGHT = 128, 128


#data augmentation tool->
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode='categorical',
    subset='training'
)


validation_generator = train_datagen.flow_from_directory(
    'train_set',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode='categorical',
    subset='validation'
)



from tensorflow.keras.layers import  BatchNormalization,Activation


model = Sequential([
    Conv2D(16, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),

    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64*14*14),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(32),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    epochs=20,
validation_data = validation_generator,
)
model.save("cnn_img_model.h5")
print("training has completed!!")