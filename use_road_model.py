import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the saved model
# TODO: NEED TO FIX FOR MODEL THAT CAN BUILD

loaded_model = load_model('road_classifier_model.keras')

# Load and preprocess new images
image_size = (128, 128)
new_image_paths = ['images/clean_1.jpg', 'images/dirty_45.jpg', 'images/clean_104.jpg']
new_images = []

for img_path in new_image_paths:
    img = cv2.imread(img_path)
    img = cv2.resize(img, image_size)
    img = img / 255.0  # Normalize pixel values
    new_images.append(img)

new_images = np.array(new_images)

# Make predictions
predictions = loaded_model.predict(new_images)

for i, pred in enumerate(predictions):
    if pred >= 0.5:
        print(f"Image {i+1}: Dirty ({pred[0]:.4f} probability)")
    else:
        print(f"Image {i+1}: Clean ({1-pred[0]:.4f} probability)")
