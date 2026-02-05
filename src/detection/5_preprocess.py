import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import gc

# ===========================
# CONFIG
# ===========================
INPUT_CSV_TRAIN = "./data/split/train.csv"
OUTPUT_CSV_TRAIN = "./data/preprocessed/train_processed.csv"

INPUT_CSV_TEST  = "./data/split/test.csv"
OUTPUT_CSV_TEST = "./data/preprocessed/test_processed.csv"

LABEL_COL = "Label"
SCALER_PATH = "./models/standard_scaler.joblib"

# Các cột không có giá trị hoặc gây nhiễu cần loại bỏ
DROP_COLUMNS = [
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Fwd Header Length.1", "Subflow Fwd Packets", "Subflow Fwd Bytes", 
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Avg Bwd Segment Size"
]

def clean_data(df):
    # 1. Bỏ cột rác
    df.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)
    
    # 2. Xử lý vô cực và NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    
    # 3. Ép kiểu về float32 để giảm tải RAM
    for col in numeric_cols:
        if col != LABEL_COL:
            df[col] = df[col].astype(np.float32)
            
    return df

def main():
    os.makedirs("./data/preprocessed", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # --- XỬ LÝ TẬP TRAIN ---
    if os.path.exists(INPUT_CSV_TRAIN):
        print("[+] Processing Train Dataset...")
        df_train = pd.read_csv(INPUT_CSV_TRAIN)
        df_train = clean_data(df_train)

        # Tách Label
        y_train = df_train[LABEL_COL]
        X_train = df_train.drop(columns=[LABEL_COL])
        
        # Dọn RAM
        del df_train
        gc.collect()

        # Fit Scaler
        print("[+] Fitting Scaler on Train data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Lưu Scaler
        joblib.dump(scaler, SCALER_PATH)
        print(f"[+] Scaler saved to {SCALER_PATH}")

        # Lưu file đã xử lý
        print("[+] Saving train_processed.csv...")
        df_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        df_train_final[LABEL_COL] = y_train.values
        df_train_final.to_csv(OUTPUT_CSV_TRAIN, index=False)
        
        # Dọn dẹp
        del X_train, X_train_scaled, y_train, df_train_final
        gc.collect()

    # --- XỬ LÝ TẬP TEST ---
    if os.path.exists(INPUT_CSV_TEST):
        print("\n[+] Processing Test Dataset...")
        df_test = pd.read_csv(INPUT_CSV_TEST)
        df_test = clean_data(df_test)
        
        y_test = df_test[LABEL_COL]
        X_test = df_test.drop(columns=[LABEL_COL])
        
        del df_test
        gc.collect()

        # Chỉ Transform (dùng Scaler đã fit ở trên)
        print("[+] Transforming Test data...")
        X_test_scaled = scaler.transform(X_test)
        
        # Lưu file
        print("[+] Saving test_processed.csv...")
        df_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        df_test_final[LABEL_COL] = y_test.values
        df_test_final.to_csv(OUTPUT_CSV_TEST, index=False)
        print("✅ Preprocessing Completed!")

if __name__ == "__main__":
    main()