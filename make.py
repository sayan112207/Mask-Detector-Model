import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define the paths to your training and validation data
train_data_path = "C:/Users/KIIT/Project/Merger/train_data"
val_data_path = "C:/Users/KIIT/Project/Merger/validation_data"

# Define the size of your images
img_size = (224, 224)

# Define your labels
labels = {"with_mask": 1, "without_mask": 0}

# Load your data into memory
train_data = []
val_data = []

for folder in ["with_mask", "without_mask"]:
    for img in os.listdir(os.path.join(train_data_path, folder)):
        img_path = os.path.join(train_data_path, folder, img)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        label = labels[folder]
        train_data.append((img, label))

    for img in os.listdir(os.path.join(val_data_path, folder)):
        img_path = os.path.join(val_data_path, folder, img)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        label = labels[folder]
        val_data.append((img, label))

# Split your data into training and validation sets
train_data, train_labels = zip(*train_data)
val_data, val_labels = zip(*val_data)

train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Convert your data into arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
val_data = np.array(val_data)
val_labels = np.array(val_labels)
