import os
import pandas as pd

LABEL_MAPPING = {
    "BENIGN": 0,
    "DDoS": 1,
    "PortScan": 2,
    "Bot": 3,
    "Infiltration": 4,
    "Web Attack - Brute Force": 5,
    "Web Attack - XSS": 6,
    "Web Attack - Sql Injection": 7,
    "FTP-Patator": 8,
    "SSH-Patator": 9,
    "DoS slowloris": 10,
    "DoS Slowhttptest": 11,
    "DoS Hulk": 12,
    "DoS GoldenEye": 13,
    "Heartbleed": 14
}

def label_dataset(input_file, output_root):
    os.makedirs(output_root, exist_ok=True)
    output_file = os.path.join(output_root, "NumberLabelled.csv")

    df = pd.read_csv(input_file)

    df["Label"] = df["Label"].map(LABEL_MAPPING)

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    label_dataset(
        input_file=r"./data/dataset/CleanlyLabelled_Dataset.csv",
        output_root=r"./data/labelled"
    )
