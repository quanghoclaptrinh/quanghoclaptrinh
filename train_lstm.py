import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# --- CẤU HÌNH ---
DATA_PATH = 'MP_Data_Augmented'  # Trỏ vào thư mục dữ liệu đã nhân bản
SEQUENCE_LENGTH = 30             # Độ dài mỗi video (30 frame)

# 1. TỰ ĐỘNG QUÉT DANH SÁCH NHÃN (ACTIONS)
if not os.path.exists(DATA_PATH):
    print(f"❌ LỖI: Không tìm thấy thư mục '{DATA_PATH}'. Hãy chạy file augment trước!")
    exit()

# Tự động lấy tên thư mục làm nhãn (Xin_chao, Cam_on, So_1...)
actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
print(f"--> Tìm thấy {len(actions)} hành động để train: {actions}")

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

print("\n🚀 Đang tải dữ liệu lên RAM...")

# 2. LOAD DỮ LIỆU
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    
    # Lấy danh sách video (bao gồm video gốc và video nhân bản)
    video_list = os.listdir(action_path)
    print(f" - Đang tải hành động '{action}': {len(video_list)} videos")
    
    for video_name in video_list:
        video_path = os.path.join(action_path, video_name)
        
        # Bỏ qua nếu không phải thư mục
        if not os.path.isdir(video_path): continue
        
        window = []
        # Lấy danh sách file .npy và sắp xếp theo thứ tự 0, 1, 2...
        frame_files = sorted(os.listdir(video_path), key=lambda x: int(os.path.splitext(x)[0]))
        
        # Chỉ lấy những video đủ 30 frame (để tránh lỗi crash khi train)
        if len(frame_files) >= SEQUENCE_LENGTH:
            for frame_file in frame_files[:SEQUENCE_LENGTH]:
                res = np.load(os.path.join(video_path, frame_file))
                window.append(res)
            
            sequences.append(window)
            labels.append(label_map[action])

print(f"\n✅ Đã tải xong! Tổng cộng: {len(sequences)} mẫu dữ liệu.")

# 3. CHUẨN BỊ DỮ LIỆU (PRE-PROCESSING)
X = np.array(sequences)
y = to_categorical(labels).astype(int) # Chuyển nhãn sang dạng One-Hot Encoding

# Chia dữ liệu: 90% để học, 10% để kiểm tra (Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 4. THIẾT KẾ MÔ HÌNH LSTM (MODEL ARCHITECTURE)
model = Sequential()

# Layer 1: LSTM nhận chuỗi vào
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 63)))
# Layer 2: LSTM xử lý sâu hơn
model.add(LSTM(128, return_sequences=True, activation='relu'))
# Layer 3: LSTM nén thông tin
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Các lớp Dense để phân loại
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # Output layer = số lượng hành động (10, 20...)

# Compile mô hình
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Cấu hình "Dừng sớm" (Early Stopping): Nếu học 50 lần mà không khôn hơn thì dừng
early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

# 5. BẮT ĐẦU HUẤN LUYỆN (TRAINING)
print("\n💪 Bắt đầu huấn luyện...")
model.fit(X_train, y_train, epochs=1000, callbacks=[early_stopping])

# 6. LƯU KẾT QUẢ
model.save('action.h5')
print("\n🎉 XONG! Đã lưu mô hình vào file 'action.h5'.")

# Đánh giá nhanh
print("Kiểm tra thử độ chính xác trên tập Test:")
res = model.predict(X_test)
acc = np.sum(np.argmax(res, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Độ chính xác thực tế: {acc*100:.2f}%")