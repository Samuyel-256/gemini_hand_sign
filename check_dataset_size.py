import os

dataset_path = "D:/gemini hand sign/dataset"

for label in os.listdir(dataset_path):
    sign_folder = os.path.join(dataset_path, label)

    if not os.path.isdir(sign_folder):
        continue  # Skip if not a folder

    num_images = len(os.listdir(sign_folder))
    print(f"📂 {label}: {num_images} images")

print("✅ Dataset check complete!")
