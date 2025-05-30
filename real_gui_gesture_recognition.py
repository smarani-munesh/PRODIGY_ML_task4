import cv2
import joblib
import numpy as np
from skimage.feature import hog

IMG_SIZE = (64, 64)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

clf = joblib.load('gesture_svm_model.pkl')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    roi = frame[100:300, 100:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    features = hog(gray, **HOG_PARAMS).reshape(1, -1)
    pred = clf.predict(features)[0]

    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    cv2.putText(frame, f'Gesture: {pred}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
