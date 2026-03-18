import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Đường dẫn
RAW_PATH = "data/raw/clean_dataset.csv"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(RAW_PATH)

# Xoá dòng null
df = df.dropna()
df = df.sample(frac=1, random_state=42)
# Bước 1: tách train (80%) và temp (20%)
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42
)

# Bước 2: tách temp thành valid (10%) và test (10%)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42
)

# Lưu file
train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
valid_df.to_csv(f"{OUTPUT_DIR}/valid.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

print("Done splitting dataset!")
print(f"Train: {len(train_df)}")
print(f"Valid: {len(valid_df)}")
print(f"Test: {len(test_df)}")