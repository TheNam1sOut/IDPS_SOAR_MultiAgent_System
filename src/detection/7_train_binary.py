import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import os

# ===========================
# CONFIG
# ===========================
TRAIN_DATA_PATH = "./data/preprocessed/train_processed.csv"
MODEL_SAVE_PATH = "./models/binary_model.keras" 
LABEL_COL = "Label"
BATCH_SIZE = 2048 # Dense Network nhẹ nên tăng Batch Size để train nhanh hơn
EPOCHS = 30

def load_data(path):
    print(f"[+] Loading data from {path}...")
    df = pd.read_csv(path)
    X = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)
    return X, y

def build_dense_model(input_dim):
    # Kiến trúc Dense Network (ANN) thay vì LSTM
    # Phù hợp hơn với dữ liệu dạng bảng (Tabular Data)
    model = Sequential([
        Input(shape=(input_dim,)),
        
        # Layer 1: Rộng để bắt đặc trưng
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 2: Co lại
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 3: Tinh chỉnh
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output: Binary (0 hoặc 1)
        Dense(1, activation="sigmoid") 
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    os.makedirs("./models", exist_ok=True)
    
    if not os.path.exists(TRAIN_DATA_PATH):
        print("❌ Error: Train data not found!")
        return

    # 1. Load Dữ liệu (Vào RAM)
    X, y = load_data(TRAIN_DATA_PATH)
    
    # 2. Tự động tính Class Weight (Cân bằng dữ liệu)
    # Giúp model chú ý vào lớp Attack (số lượng ít)
    unique_classes = np.unique(y)
    weights = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=y
    )
    class_weights_dict = dict(enumerate(weights))
    print(f"[+] Auto Class Weights computed: {class_weights_dict}")

    # 3. Xây dựng Model
    model = build_dense_model(X.shape[1])
    model.summary()
    
    # 4. Cấu hình Callbacks
    callbacks = [
        # Dừng nếu val_loss không giảm sau 5 epochs
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        # Lưu model tốt nhất
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss")
    ]

    # 5. Train
    print("\n[+] Starting Training...")
    history = model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights_dict, # Áp dụng trọng số tại đây
        validation_split=0.1,            # Dùng 10% tập train để kiểm thử chéo
        verbose=1
    )
    
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()