import cv2
import mediapipe as mp
import os

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)  # Lower confidence to detect hands better
mp_draw = mp.solutions.drawing_utils

# Set dataset path to D:\gemini hand sign
dataset_path = "D:/gemini hand sign/dataset"
os.makedirs(dataset_path, exist_ok=True)

# Ask for sign label
sign_label = input("Enter the label for this sign (e.g., 'Hello', 'Yes', 'OK'): ").strip()

# Create a folder for the sign inside the dataset
sign_folder = os.path.join(dataset_path, sign_label)
os.makedirs(sign_folder, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
count = 0

print("\n📸 Press 'S' to capture an image")
print("❌ Press 'Q' to quit and close the camera\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not detected!")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        print("✅ Hand detected!")

        # Save image when 's' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            image_path = os.path.join(sign_folder, f"{sign_label}_{count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"✅ Image saved: {image_path}")
            count += 1

    else:
        print("❌ No hand detected. Try adjusting lighting or position.")

    # Show webcam feed
    cv2.imshow("Hand Sign Data Collection", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("📌 Data collection stopped! Closing camera...")
        break

# Release resources properly
cap.release()
cv2.destroyAllWindows()
print("✅ Camera closed successfully!")
