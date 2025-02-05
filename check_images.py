import cv2
import os

dataset_path = "D:/gemini hand sign/dataset"

# Check each folder
for label in os.listdir(dataset_path):
    sign_folder = os.path.join(dataset_path, label)

    if not os.path.isdir(sign_folder):
        continue  # Skip if it's not a folder

    print(f"📂 Checking folder: {label}")

    # Check each image in the folder
    for image_name in os.listdir(sign_folder):
        image_path = os.path.join(sign_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ ERROR: Cannot read {image_name} in {label}. It may be corrupt or in an unsupported format.")
        else:
            print(f"✅ {image_name} in {label} is valid.")

print("📌 Image check complete!")
