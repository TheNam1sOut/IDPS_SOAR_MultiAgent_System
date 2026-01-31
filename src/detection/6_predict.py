"""
CIC-IDS-2017 LSTM Inference Script
---------------------------------
Chức năng:
- Load scaler + trained LSTM model
- Load preprocessed test dataset
- Build time-series sequences (windowing)
- Predict multi-class labels
- Xuất kết quả + đánh giá cơ bản

LƯU Ý:
- KHÔNG fit lại scaler
- KHÔNG shuffle dữ liệu
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# CONFIG
# =========================

TEST_CSV = "./data/preprocessed/test_processed.csv"

MODEL_PATH  = "./models/model.keras"
SCALER_PATH = "./models/standard_scaler.joblib"

LABEL_COL = "Label"

TIMESTEPS  = 10
BATCH_SIZE = 256

OUTPUT_PRED_CSV = "./results/test_predictions.csv"


# =========================
# FUNCTIONS
# =========================

def load_test_data(path):
    print("[+] Loading test dataset...")
    df = pd.read_csv(path)
    print(f"[+] Loaded: {df.shape[0]} rows")
    return df


def build_sequences(X, y, timesteps):
    """
    Chuyển dữ liệu tabular -> sequence cho LSTM
    """
    X_seq = []
    y_seq = []

    for i in range(len(X) - timesteps):
        X_seq.append(X.iloc[i:i + timesteps].values)
        y_seq.append(y.iloc[i + timesteps])

    return np.array(X_seq), np.array(y_seq)


def main():
    # =========================
    # LOAD MODEL & SCALER
    # =========================
    print("[+] Loading scaler...")
    scaler = joblib.load(SCALER_PATH)

    print("[+] Loading trained LSTM model...")
    model = load_model(MODEL_PATH)

    # =========================
    # LOAD TEST DATA
    # =========================
    df = load_test_data(TEST_CSV)

    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    # =========================
    # BUILD SEQUENCES
    # =========================
    print("[+] Building LSTM sequences...")
    X_seq, y_seq = build_sequences(X, y, TIMESTEPS)

    print(f"[+] Sequence shape: {X_seq.shape}")
    print(f"[+] Label shape   : {y_seq.shape}")

    # =========================
    # PREDICTION
    # =========================
    print("[+] Running inference...")
    y_prob = model.predict(
        X_seq,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    y_pred = np.argmax(y_prob, axis=1)

    # =========================
    # EVALUATION
    # =========================
    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(y_seq, y_pred, digits=4))

    print("========== CONFUSION MATRIX ==========")
    print(confusion_matrix(y_seq, y_pred))

    # =========================
    # SAVE RESULTS
    # =========================
    print("[+] Saving predictions...")
    result_df = pd.DataFrame({
        "y_true": y_seq,
        "y_pred": y_pred
    })

    result_df.to_csv(OUTPUT_PRED_CSV, index=False)
    print(f"[+] Predictions saved to {OUTPUT_PRED_CSV}")


if __name__ == "__main__":
    main()
