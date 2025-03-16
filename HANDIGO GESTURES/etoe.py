import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import cv2
import random


DATASET_PATH = r"C:\Program Files\PortableGit\GESTURE\HANDIGO GESTURES\TEST"

# 🔹 Image parameters
IMG_SIZE = 128  # Resize images to 128x128
BATCH_SIZE = 32
EPOCHS = 20

# 🔹 Load dataset (Images grouped by gesture folders)
gesture_classes = sorted(os.listdir(DATASET_PATH))  # Folder names as labels
print("Classes:", gesture_classes)

# 🔹 Create training data
train_data = []
labels = []
for label_idx, gesture in enumerate(gesture_classes):
    gesture_path = os.path.join(DATASET_PATH, gesture)
    for file_name in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, file_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize
        image = image / 255.0  # Normalize (0-1)
        train_data.append(image)
        labels.append(label_idx)

# 🔹 Convert to NumPy arrays
train_data = np.array(train_data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# 🔹 Shuffle the dataset
combined = list(zip(train_data, labels))
random.shuffle(combined)
train_data, labels = zip(*combined)
train_data = np.array(train_data)
labels = np.array(labels)

# 🔹 Split into training & validation sets
split_index = int(0.8 * len(train_data))  # 80% training, 20% validation
x_train, x_val = train_data[:split_index], train_data[split_index:]
y_train, y_val = labels[:split_index], labels[split_index:]

# 🔹 Convert labels to categorical
num_classes = len(gesture_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# 🔹 Define CNN-LSTM model
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")  # Gesture classification
    ])
    
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# 🔹 Build and train the model
model = build_model()
model.summary()

history = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val),
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE)

# 🔹 Save the trained model
model.save("gesture_model.h5")
print("✅ Model training complete and saved as 'gesture_model.h5'")

