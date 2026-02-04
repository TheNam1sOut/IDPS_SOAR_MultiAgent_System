import os
import pandas as pd

def label_dataset(input_file, output_root):
    os.makedirs(output_root, exist_ok=True)
    output_file = os.path.join(output_root, "NumberLabelled.csv")

    print(f"[+] Reading {input_file}...")
    df = pd.read_csv(input_file)

    # LOGIC BINARY: Benign là 0, tất cả còn lại (Attack) là 1
    print("[+] Converting to Binary Labels (0: Normal, 1: Attack)...")
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    # In ra để kiểm tra
    print("Label distribution:")
    print(df["Label"].value_counts())

    df.to_csv(output_file, index=False)
    print(f"[+] Saved binary labeled data to {output_file}")

if __name__ == "__main__":
    label_dataset(
        input_file=r"./data/dataset/CleanlyLabelled_Dataset.csv",
        output_root=r"./data/labelled"
    )