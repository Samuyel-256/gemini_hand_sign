import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# Load dataset
dataset_path = "D:/gemini hand sign/hand_sign_data.csv"
df = pd.read_csv(dataset_path)

# Debugging: Print dataset size
print(f"📂 Dataset contains {df.shape[0]} samples and {df.shape[1]} columns.")

if df.empty:
    print("❌ Error: The dataset is empty! Run extract_landmarks.py again.")
    exit()

# Separate features (X) and labels (y)
X = df.iloc[:, 1:].values  # Landmark values
y = df.iloc[:, 0].values   # Labels (sign names)

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=3)  # Using 3 neighbors for better accuracy
model.fit(X_train, y_train)

# Save the trained model
model_path = "D:/gemini hand sign/hand_sign_model.pkl"
joblib.dump(model, model_path)

# Print model accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%")
print(f"📁 Model saved at: {model_path}")
