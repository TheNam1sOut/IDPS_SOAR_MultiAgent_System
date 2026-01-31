"""
CIC-IDS-2017 Preprocessing Script
--------------------------------
Chức năng:
- Loại bỏ feature dư thừa / nhiễu (đã audit)
- Xử lý NaN / Infinity (Flow Bytes/s, Flow Packets/s)
- Scaling dữ liệu cho Deep Learning (LSTM)
- Xuất dữ liệu sạch ra file mới

Không bao gồm:
- Windowing / sequence building
- Huấn luyện model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


# =========================
# CONFIG
# =========================

INPUT_CSV  = "./data/split/train.csv"
OUTPUT_CSV = "./data/preprocessed/train_processed.csv"
INPUT_CSV_TEST  = "./data/split/test.csv"
OUTPUT_CSV_TEST = "./data/preprocessed/test_processed.csv"

LABEL_COL = "Label"

SCALER_PATH = "./models/standard_scaler.joblib"

# =========================
# DROP LIST (TỪ KẾT LUẬN AUDIT)
# =========================

# Zero-variance + Bulk features
DROP_COLUMNS = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",

    # Duplicate by value / artifact
    "Fwd Header Length.1",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Avg Bwd Segment Size"
]


# =========================
# FUNCTIONS
# =========================

def load_dataset(path):
    print("[+] Loading dataset...")
    df = pd.read_csv(path)
    print(f"[+] Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_unwanted_columns(df):
    print("[+] Dropping unwanted columns...")
    before = df.shape[1]
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    after = df.shape[1]
    print(f"    Dropped {before - after} columns")
    return df


def handle_inf_nan(df):
    print("[+] Handling NaN and Infinity...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Replace inf -> NaN
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Thống kê trước khi xử lý
    nan_before = df.isna().sum().sum()
    print(f"    NaN before handling: {nan_before}")

    # Với CIC-IDS-2017: NaN rất ít → drop row là an toàn
    df.fillna(0, inplace=True)

    nan_after = df.isna().sum().sum()
    print(f"    NaN after handling : {nan_after}")

    return df


def split_features_label(df):
    print("[+] Splitting features and label...")
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]
    return X, y


def scale_features(X):
    print("[+] Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, scaler


def save_output(X, y, path):
    print("[+] Saving preprocessed dataset...")
    df_out = pd.concat([X, y.reset_index(drop=True)], axis=1)
    df_out.to_csv(path, index=False)
    print(f"[+] Saved to: {path}")


# =========================
# MAIN PIPELINE
# =========================

def main():
    df = load_dataset(INPUT_CSV)
    df_test = load_dataset(INPUT_CSV_TEST)

    df = drop_unwanted_columns(df)
    df_test = drop_unwanted_columns(df_test)

    df = handle_inf_nan(df)
    df_test = handle_inf_nan(df_test)

    X, y = split_features_label(df)
    feature_columns = X.columns.tolist()

    X_test, y_test = split_features_label(df_test)
    X_test = X_test[feature_columns]

    X_scaled, scaler = scale_features(X)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[+] Scaler saved to {SCALER_PATH}")

    X_scaled_test = scaler.transform(X_test)
    X_scaled_test = pd.DataFrame(X_scaled_test, columns=X.columns)

    save_output(X_scaled, y, OUTPUT_CSV)
    save_output(X_scaled_test, y_test, OUTPUT_CSV_TEST)

    print("\nPreprocessing completed successfully.")
    print(f"    Final shape: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")


if __name__ == "__main__":
    main()
