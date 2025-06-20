# run_sorter.py

# --- NEW: Add these lines at the VERY TOP to silence logs ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# --- END NEW ---

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import config

print("Loading waste sorter model...")
model = load_model('waste_sorter_model.keras')
print("Model loaded successfully!")

class_labels = {0: 'Non-Recyclable', 1: 'Recyclable'}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    img_resized = cv2.resize(frame, (config.IMG_WIDTH, config.IMG_HEIGHT))
    img_array = np.asarray(img_resized)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # --- CHANGED: Added 'verbose=0' to silence the prediction log ---
    prediction = model.predict(img_batch, verbose=0)
    # --- END CHANGED ---
    
    confidence = prediction[0][0]

    if confidence > 0.5:
        label_index = 1
        label_text = class_labels[label_index]
        confidence_percent = confidence * 100
        display_color = (0, 255, 0)
    else:
        label_index = 0
        label_text = class_labels[label_index]
        confidence_percent = (1 - confidence) * 100
        display_color = (0, 0, 255)

    display_text = f"{label_text}: {confidence_percent:.1f}%"
    
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (100, 100), (540, 380), display_color, 2)
    cv2.imshow('Smart Waste Sorter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program terminated.")