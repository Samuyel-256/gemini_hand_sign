import pandas as pd

dataset_path = "D:/gemini hand sign/hand_sign_data.csv"

try:
    df = pd.read_csv(dataset_path)
    print("✅ Dataset loaded successfully!")
    print(df.head())  # Print first few rows
    print(f"Total Samples: {len(df)}")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
