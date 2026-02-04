import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================
TEST_CSV = "./data/preprocessed/test_processed.csv"
MODEL_PATH  = "./models/binary_model.keras" # Đọc đúng model binary vừa train
SCALER_PATH = "./models/standard_scaler.joblib"
OUTPUT_PRED_CSV = "./results/binary_test_predictions.csv"

LABEL_COL = "Label"
TIMESTEPS  = 10
BATCH_SIZE = 256

def main():
    # Kiểm tra đường dẫn
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Không tìm thấy model tại {MODEL_PATH}")
        return

    print(f"[+] Đang load model từ {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    print("[+] Đang load dữ liệu test...")
    df = pd.read_csv(TEST_CSV)
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    # Tạo sequence cho LSTM
    print("[+] Tạo sequence...")
    X_seq = []
    y_seq = []
    for i in range(len(X) - TIMESTEPS):
        X_seq.append(X.iloc[i:i + TIMESTEPS].values)
        y_seq.append(y.iloc[i + TIMESTEPS])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Dự đoán
    print("[+] Đang dự đoán (Binary)...")
    y_prob = model.predict(X_seq, batch_size=BATCH_SIZE, verbose=1)
    
    # Ngưỡng 0.5: > 0.5 là Attack (1), <= 0.5 là Benign (0)
    y_pred = (y_prob > 0.5).astype(int).flatten()

    # Đánh giá
    print("\n========== KẾT QUẢ BINARY CLASSIFICATION ==========")
    print(classification_report(y_seq, y_pred, digits=4, target_names=["Benign", "Attack"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_seq, y_pred))

    # Lưu kết quả
    os.makedirs("./results", exist_ok=True)
    pd.DataFrame({"y_true": y_seq, "y_pred": y_pred}).to_csv(OUTPUT_PRED_CSV, index=False)
    print(f"[+] Đã lưu kết quả tại {OUTPUT_PRED_CSV}")

if __name__ == "__main__":
    main()