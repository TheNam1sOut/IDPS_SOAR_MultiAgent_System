import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "./data/labelled/NumberLabelled.csv"
OUTPUT_DIR = "./data/split"

TEST_SIZE = 0.2
RANDOM_STATE = 42

def split_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    if "Label" not in df.columns:
        raise ValueError("Label column not found")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Label"]
    )

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Dataset split completed")
    print(f"Train: {train_df.shape}")
    print(f"Test : {test_df.shape}")

if __name__ == "__main__":
    split_dataset()
