import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load metadata CSV
metadata_path = 'metadata.csv'
df = pd.read_csv(metadata_path)

# Directory containing all images
image_dir = 'images/'

# Load and preprocess images
image_size = (128, 128)
images = []
labels = []

for index, row in df.iterrows():
    img_path = os.path.join(image_dir, row['filename'])
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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Training
batch_size = 32
epochs = 100

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), 
          steps_per_epoch=len(X_train) // batch_size, epochs=epochs,
          validation_data=(X_val, y_val))

# Step 8: Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
