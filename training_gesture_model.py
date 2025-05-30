import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Constants
IMG_SIZE = (128, 128)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (16, 16),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# Dataset path
dataset_path = 'archive/leapGestRecog'

# Prepare dataset
X = []
y = []

# Loop through person folders 00 to 09
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    # Loop through gesture folders like 01_palm, 02_l, etc.
    for gesture_folder in os.listdir(person_path):
        gesture_path = os.path.join(person_path, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue

        label = gesture_folder  # label is folder name like "01_palm"

        for img_file in os.listdir(gesture_path):
            if img_file.endswith(".png"):
                img_path = os.path.join(gesture_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"⚠️ Could not read image: {img_path}")
                    continue

                img = cv2.resize(img, IMG_SIZE)
                features = hog(img, **HOG_PARAMS)
                X.append(features)
                y.append(label)

print("✅ Finished loading data. Total samples:", len(X))

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Test accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc)

# Save the model
joblib.dump(clf, "gesture_svm_model.pkl")
print("✅ Model saved as 'gesture_svm_model.pkl'")
