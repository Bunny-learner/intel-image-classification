import pandas as pd
import seaborn as sns
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

print("Model is being loaded!!")
loaded_model = load_model("cnn_img_model.h5")
print("Testing started:")

# Define constants
TEST_DIR = 'test_set'
IMG_WIDTH, IMG_HEIGHT = 200, 200
BATCH_SIZE = 32

# Create a test data generator without augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Create a generator for the test data
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'  # Use categorical labels for multi-class classification
)

# Evaluate the model on the test set
test_loss, test_acc = loaded_model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2f}')

# Predict labels on the test set
test_steps = test_generator.samples // BATCH_SIZE + 1 if test_generator.samples % BATCH_SIZE != 0 else test_generator.samples // BATCH_SIZE
predictions = loaded_model.predict(test_generator, steps=test_steps)

# Convert predictions to class labels (argmax for categorical labels)
predicted_labels = np.argmax(predictions, axis=1)

# Get actual labels from the generator
actual_labels = test_generator.classes

# Calculate confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)
print("Confusion Matrix:\n", cm)

# Calculate F1 score
f1 = f1_score(actual_labels, predicted_labels, average='macro')
print(f"F1 Score: {f1}")
