"""
This script demonstrates the process of building, training, and evaluating a road classifier model using TensorFlow and Keras.

It performs the following steps:
1. Imports necessary libraries, including TensorFlow, NumPy, Pandas, OpenCV, and scikit-learn.
2. Loads metadata from a CSV file containing image filenames and labels.
3. Loads and preprocesses images, resizing them to a specified size and normalizing pixel values.
4. Splits the data into training, validation, and test sets using scikit-learn's train_test_split function.
5. Constructs a Convolutional Neural Network (CNN) model using Keras Sequential API.
6. Compiles the model using the Nadam optimizer and binary cross-entropy loss.
7. Defines a learning rate scheduling function and associated callbacks for training.
8. Augments training data using image data augmentation techniques.
9. Fits the model to the training data using the augmented generator and monitors validation loss for early stopping.
10. Evaluates the trained model on the test set and prints the test loss and accuracy.
11. Saves the trained model to a file named 'road_classifier_model.keras'.

Author: [Your Name]

Usage:
1. Ensure the 'metadata.csv' file and image files are available.
2. Set the 'IMAGE_DIR' and 'METADATA_PATH' variables to appropriate paths.
3. Adjust hyperparameters and model architecture if needed.
4. Run the script to train the model and evaluate its performance.
"""
import os
import numpy as np
import pandas as pd
import cv2
from google.cloud import storage
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler

# GCP setup to grab from bucket storage
storage_client = storage.Client()
bucket_name = 'clean_dirty_road_ml_bucket'

bucket = storage_client.bucket(bucket_name)
blob = bucket.blob('metadata.csv')
blob.download_to_filename('test-pull-from-gcp.csv')
print('Pulled from GCP, check files')

# Load metadata CSV
METADATA_PATH = 'test-pull-from-gcp.csv'
df = pd.read_csv(METADATA_PATH)

# Directory containing all images
IMAGE_DIR = 'images/'

# Load and preprocess images
image_size = (128, 128)
images = []
labels = []

for index, row in df.iterrows():
    img_path = os.path.join(IMAGE_DIR, row['filename'])
    img = cv2.imread(img_path)
    img = cv2.resize(img, image_size)
    img = img / 255.0  # Normalize pixel values
    images.append(img)
    labels.append(row['label'])

images = np.array(images)
labels = np.array(labels)

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4-6: Choosing a Model, Model Architecture, and Compilation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

def lr_schedule(epoch):
    """
    Learning rate schedule function.
    This function computes a learning rate adjustment based on the epoch number,
    using a step-wise decay strategy. The learning rate is reduced by a fixed drop_rate
    after every epochs_drop epochs.

    Parameters:
        epoch (int): The current epoch number.

    Returns:
        float: The adjusted learning rate for the current epoch.
    """
    initial_lr = 0.001
    drop_rate = 0.65
    epochs_drop = 10
    lr = initial_lr * np.power(drop_rate, np.floor((1 + epoch) / epochs_drop))
    return lr

model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Learning rate scheduling callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Hyper param that makes model stop learning once val_loss has stopped decreasing.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Step 7: Training
batch_size = 8
epochs = 200

data_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
data_gen.fit(X_train)

model.fit(data_gen.flow(X_train, y_train, batch_size=batch_size),
          steps_per_epoch=len(X_train) // batch_size, epochs=epochs,
          validation_data=(X_val, y_val), callbacks=[lr_scheduler, early_stopping])

# Step 8: Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# Save the trained model
model.save('road_classifier_model.keras')
print("New model saves to road_classifier_model.keras")
