import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH ---
DATA_PATH = 'MP_Data_Augmented' 
SEQUENCE_LENGTH = 30             
FEATURE_DIM = 78 # Vector 78 chiều (63 tọa độ + 15 góc)

# Tự động quét các chữ cái có trong thư mục
actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
label_map = {label:num for num, label in enumerate(actions)}

# --- 2. LOAD DỮ LIỆU ---
sequences, labels = [], []
print(f"🚀 Đang tải mảng dữ liệu khổng lồ từ {DATA_PATH}...")

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for video_name in os.listdir(action_path):
        video_path = os.path.join(action_path, video_name)
        if not os.path.isdir(video_path): continue
        
        window = []
        frame_files = sorted(os.listdir(video_path), key=lambda x: int(os.path.splitext(x)[0]))
        
        # Chỉ lấy những video đủ 30 frame
        if len(frame_files) >= SEQUENCE_LENGTH:
            for frame_file in frame_files[:SEQUENCE_LENGTH]:
                res = np.load(os.path.join(video_path, frame_file))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"✅ Tổng số video chuẩn bị train: {X.shape[0]}")
print(f"✅ Kích thước Tensor đầu vào (Batch, Frames, Features): {X.shape}")

# Chia tập Train (80%) và tập Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. XÂY DỰNG KIẾN TRÚC MẠNG BiLSTM ---
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)))
model.add(Dropout(0.2)) # Tắt ngẫu nhiên 20% nơ-ron để chống học vẹt

model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(64, return_sequences=False, activation='relu')))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # Lớp xuất ra xác suất các chữ cái

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 4. CALLBACKS (Giám sát huấn luyện) ---
# Dừng sớm nếu sau 40 vòng mà không thông minh lên
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

# Chỉ lưu lại model tốt nhất (dựa trên điểm bài thi test val_categorical_accuracy)
checkpoint = ModelCheckpoint('action_bilstm_v2_best.h5', monitor='val_categorical_accuracy', 
                             save_best_only=True, mode='max', verbose=1)

# --- 5. BẮT ĐẦU HUẤN LUYỆN ---
print("\n🔥 BẮT ĐẦU ĐỐT CHÁY GPU/CPU ĐỂ HUẤN LUYỆN...")
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), # Thi thử ngay sau mỗi vòng
                    epochs=300, 
                    batch_size=64, # Gom 64 video học cùng lúc
                    callbacks=[early_stopping, checkpoint])

# --- 6. VẼ BIỂU ĐỒ BÁO CÁO ĐỒ ÁN ---
print("\n📊 Đang xuất biểu đồ báo cáo...")
plt.figure(figsize=(12, 5))

# Biểu đồ Accuracy (Độ chính xác)
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy', linewidth=2, linestyle='--')
plt.title('Biểu đồ Độ chính xác (Accuracy)')
plt.xlabel('Vòng lặp (Epochs)')
plt.ylabel('Độ chính xác')
plt.legend()
plt.grid(True)

# Biểu đồ Loss (Độ sai số)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
plt.title('Biểu đồ Độ sai số (Loss)')
plt.xlabel('Vòng lặp (Epochs)')
plt.ylabel('Sai số')
plt.legend()
plt.grid(True)

plt.savefig('Training_History_Chart.png', dpi=300, bbox_inches='tight')
print("🎉 XONG! Đã lưu model tốt nhất thành 'action_bilstm_v2_best.h5'")
print("📸 Đã lưu biểu đồ thành 'Training_History_Chart.png' (Mở lên xem ngay nhé!)")