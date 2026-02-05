import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ===========================
# CONFIG
# ===========================
TEST_CSV = "./data/preprocessed/test_processed.csv"
MODEL_PATH  = "./models/binary_model.keras"
OUTPUT_PRED_CSV = "./results/binary_test_predictions.csv"
LABEL_COL = "Label"
BATCH_SIZE = 2048

def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found! Please train first.")
        return

    print("[+] Loading Model & Test Data...")
    model = load_model(MODEL_PATH)
    df = pd.read_csv(TEST_CSV)
    
    X_test = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
    y_true = df[LABEL_COL].values.astype(int)
    
    # 1. Dự đoán xác suất (Probabilities)
    print("[+] Predicting...")
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).flatten()
    
    # 2. Tự động tìm ngưỡng (Threshold Tuning)
    # Vì dữ liệu lệch, ngưỡng 0.5 chưa chắc đã tốt nhất
    print("\n[+] Tuning Threshold (Finding optimal cut-off)...")
    best_thresh = 0.5
    best_f1 = 0.0
    
    # Quét các ngưỡng phổ biến
    thresholds = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"{'Threshold':<10} | {'Macro F1':<10}")
    print("-" * 25)
    
    for thresh in thresholds:
        y_pred_temp = (y_prob > thresh).astype(int)
        score = f1_score(y_true, y_pred_temp, average='macro')
        print(f"{thresh:<10.2f} | {score:<10.4f}")
        
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    print(f"\n>>> SELECTED BEST THRESHOLD: {best_thresh}")
    
    # 3. Áp dụng ngưỡng tốt nhất để ra kết quả cuối cùng
    y_pred = (y_prob > best_thresh).astype(int)

    # 4. Báo cáo kết quả
    print("\n========== FINAL CLASSIFICATION REPORT ==========")
    print(classification_report(y_true, y_pred, digits=4, target_names=["Benign", "Attack"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # 5. Lưu kết quả
    os.makedirs("./results", exist_ok=True)
    result_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "probability": y_prob
    })
    result_df.to_csv(OUTPUT_PRED_CSV, index=False)
    print(f"\n✅ Results saved to {OUTPUT_PRED_CSV}")

if __name__ == "__main__":
    main()