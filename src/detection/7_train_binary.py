import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# =========================
# CẤU HÌNH ĐƯỜNG DẪN (Tính từ Root Project)
# =========================
TRAIN_DATA_PATH = "./data/preprocessed/train_processed.csv"
# Lưu model ra folder models ở ngoài root
MODEL_SAVE_PATH = "./models/binary_model.keras" 

TIMESTEPS = 10
BATCH_SIZE = 1024 
EPOCHS = 20
LABEL_COL = "Label"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file {path}. Hãy chạy script từ thư mục gốc!")
        
    print(f"[+] Đang tải dữ liệu từ {path}...")
    df = pd.read_csv(path)
    X = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)
    return X, y

def sequence_generator(X, y, timesteps, batch_size):
    num_samples = len(X) - timesteps
    while True:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            X_batch = []
            y_batch = []
            for i in range(start, end):
                X_batch.append(X[i:i+timesteps])
                y_batch.append(y[i+timesteps])
            yield np.array(X_batch), np.array(y_batch)

def build_binary_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        # Lớp Output cho Binary: 1 Unit + Sigmoid
        Dense(1, activation="sigmoid") 
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    # Đảm bảo thư mục models tồn tại
    os.makedirs("./models", exist_ok=True)

    X, y = load_data(TRAIN_DATA_PATH)
    print(f"[+] Kích thước dữ liệu: {X.shape}")

    model = build_binary_model((TIMESTEPS, X.shape[1]))
    model.summary()

    steps_per_epoch = (len(X) - TIMESTEPS) // BATCH_SIZE
    
    callbacks = [
        EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="loss")
    ]

    print("[+] Bắt đầu Train Binary Classification...")
    model.fit(
        sequence_generator(X, y, TIMESTEPS, BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    print(f"[+] Model đã được lưu tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()