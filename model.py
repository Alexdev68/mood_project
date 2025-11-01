# model.py
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Path to your dataset
# Folder structure example:
# dataset/
# ├── happy/
# ├── sad/
# ├── angry/
# ├── neutral/
DATASET_DIR = "dataset"

# Initialize lists
data, labels = [], []

# Load images and labels
for emotion in os.listdir(DATASET_DIR):
    max_images_per_class = 100  # reduce to speed up training
    for i, img_file in enumerate(os.listdir(path)):
        if i >= max_images_per_class:
        break
    # ... load/process image ...

    folder = os.path.join(DATASET_DIR, emotion)
    if not os.path.isdir(folder):
        continue
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (48, 48))
        data.append(img.flatten())
        labels.append(emotion)

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

print("✅ Model trained successfully.")
print("Accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "emotion_model.pkl")
print("💾 Model saved as emotion_model.pkl")
