import numpy as np
import os
import shutil

# --- CẤU HÌNH ---
INPUT_DIR = 'MP_Data'              # Thư mục gốc chứa dữ liệu vừa quay
OUTPUT_DIR = 'MP_Data_Augmented'   # Thư mục mới sẽ được tạo ra

# 1. Dọn dẹp thư mục cũ (nếu có) để làm mới hoàn toàn
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# 2. Tự động lấy danh sách các hành động (Xin_chao, Cam_on...)
if not os.path.exists(INPUT_DIR):
    print(f"❌ LỖI: Không tìm thấy thư mục '{INPUT_DIR}'!")
    exit()

actions = [name for name in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, name))]
print(f"--> Tìm thấy {len(actions)} hành động: {actions}")

# --- CÁC HÀM BIẾN ĐỔI DỮ LIỆU ---
def add_noise(data):
    # Thêm nhiễu ngẫu nhiên (Giả lập camera rung, ánh sáng thay đổi)
    noise = np.random.normal(0, 0.02, data.shape)
    return data + noise

def scale_data(data, scale_factor):
    # Phóng to/Thu nhỏ biên độ (Giả lập người đứng gần/xa)
    return data * scale_factor

# --- BẮT ĐẦU NHÂN BẢN ---
print(f"\n🚀 Đang nhân bản dữ liệu từ '{INPUT_DIR}' sang '{OUTPUT_DIR}'...")
total_videos = 0

for action in actions:
    src_action_path = os.path.join(INPUT_DIR, action)
    dst_action_path = os.path.join(OUTPUT_DIR, action)
    os.makedirs(dst_action_path, exist_ok=True)
    
    # Lấy danh sách các video (folder con 0, 1, 2...)
    video_folders = [f for f in os.listdir(src_action_path) if os.path.isdir(os.path.join(src_action_path, f))]
    
    for video_folder in video_folders:
        src_video_path = os.path.join(src_action_path, video_folder)
        
        # Đọc 30 frame của video gốc
        frames = []
        file_list = sorted(os.listdir(src_video_path), key=lambda x: int(os.path.splitext(x)[0]))
        
        # Kiểm tra video lỗi (không đủ 30 frame)
        if len(file_list) < 30:
            continue

        # Load dữ liệu lên
        for frame_file in file_list:
            res = np.load(os.path.join(src_video_path, frame_file))
            frames.append(res)
        
        original_sequence = np.array(frames) # (30, 63)

        # --- TẠO 4 BIẾN THỂ ---
        variants = {
            'goc': original_sequence,                     # 1. Giữ nguyên
            'nhieu': add_noise(original_sequence),        # 2. Thêm nhiễu
            'phong_to': scale_data(original_sequence, 1.1), # 3. Phóng to 10%
            'thu_nho': scale_data(original_sequence, 0.9) # 4. Thu nhỏ 10%
        }

        # Lưu tất cả vào thư mục mới
        for var_name, data in variants.items():
            # Tên folder mới: ví dụ "0_goc", "0_nhieu"...
            new_video_name = f"{video_folder}_{var_name}"
            new_video_path = os.path.join(dst_action_path, new_video_name)
            os.makedirs(new_video_path, exist_ok=True)
            
            for i, frame_data in enumerate(data):
                np.save(os.path.join(new_video_path, f"{i}.npy"), frame_data)
            
            total_videos += 1

print(f"\n✅ HOÀN TẤT! Đã tạo ra tổng cộng {total_videos} video trong thư mục '{OUTPUT_DIR}'.")
print("👉 Bước tiếp theo: Chạy file 'train_lstm.py' (nhớ trỏ vào thư mục Augmented nhé).")