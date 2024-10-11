import os
import zipfile
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from google.colab import files

# Step 1: Upload kaggle.json file
uploaded = files.upload()

# Step 2: Move kaggle.json to the right directory
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/

# Step 3: Set permissions to the file
!chmod 600 ~/.kaggle/kaggle.json

# Step 4: Download the dataset from Kaggle
!kaggle datasets download -d gti-upm/leapgestrecog -p ./data

# Step 5: Unzip the downloaded dataset
with zipfile.ZipFile('./data/leapgestrecog.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')

# Prepare the dataset
def load_images_and_labels(base_path, img_size=(64, 64)):
    images = []
    labels = []
    for label in os.listdir(base_path):
        category_path = os.path.join(base_path, label)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(int(label))
    return np.array(images), np.array(labels)

base_path = './data/leapGestRecog/leapGestRecog'
X, y = load_images_and_labels(base_path)

# Normalize the images
X = X / 255.0

# Flatten the images for the SVM
X_flat = X.reshape(X.shape[0], -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Create and train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
