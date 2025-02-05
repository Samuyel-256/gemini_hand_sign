import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)  # Lower confidence
mp_draw = mp.solutions.drawing_utils

# Set dataset path
dataset_path = "D:/gemini hand sign/dataset"

# Create a CSV file to store landmarks
csv_file = os.path.join("D:/gemini hand sign", "hand_sign_data.csv")

# Write CSV header if file doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, 'w') as f:
        f.write("label," + ",".join([f"point_{i}" for i in range(63)]) + "\n")

# Process each folder (sign label)
for label in os.listdir(dataset_path):
    sign_folder = os.path.join(dataset_path, label)

    if not os.path.isdir(sign_folder):
        continue  # Skip if it's not a folder

    for image_name in os.listdir(sign_folder):
        image_path = os.path.join(sign_folder, image_name)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Skipping: {image_path} (Not a valid image)")
            continue

        # Convert image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_image)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract 21 landmarks (x, y, z)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                print(f"✅ Extracted {len(landmarks)} landmarks for {label}")

                # Save to CSV
                with open(csv_file, 'a') as f:
                    f.write(label + "," + ",".join(map(str, landmarks)) + "\n")

print("✅ Landmark extraction complete! Data saved in hand_sign_data.csv")
