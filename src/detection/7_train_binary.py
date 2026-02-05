code_train_new = """
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import os

# CONFIG
TRAIN_DATA_PATH = "./data/preprocessed/train_processed.csv"
MODEL_SAVE_PATH = "./models/binary_model.keras" 
TIMESTEPS = 10
BATCH_SIZE = 1024 
EPOCHS = 20
LABEL_COL = "Label"

def load_data(path):
    print(f"[+] Loading dataset from {path}...")
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
        # Tăng độ phức tạp lên xíu để học tốt hơn
        LSTM(128, return_sequences=False), 
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid") 
    ])
    # Giảm learning rate xuống 0.0005 để học kỹ hơn
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    os.makedirs("./models", exist_ok=True)
    X, y = load_data(TRAIN_DATA_PATH)
    
    # --- PHẦN QUAN TRỌNG MỚI THÊM: TÍNH CLASS WEIGHT ---
    # Tự động tính toán để cân bằng giữa lớp 0 và 1
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"[+] Class Weights (Trọng số): {class_weights_dict}")
    # ---------------------------------------------------

    model = build_binary_model((TIMESTEPS, X.shape[1]))
    
    steps_per_epoch = (len(X) - TIMESTEPS) // BATCH_SIZE
    
    callbacks = [
        EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="loss")
    ]

    print("[+] Bắt đầu Train (có cân bằng dữ liệu)...")
    model.fit(
        sequence_generator(X, y, TIMESTEPS, BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights_dict, # Áp dụng trọng số
        verbose=1
    )
    model.save(MODEL_SAVE_PATH)
    print(f"[+] Model đã lưu tại {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
"""

# Ghi đè file
with open("src/detection/7_train_binary.py", "w", encoding="utf-8") as f:
    f.write(code_train_new)
print("✅ Đã cập nhật file train mới (Fix lỗi học vẹt)")