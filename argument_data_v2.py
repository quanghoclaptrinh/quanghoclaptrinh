import numpy as np
import os
import shutil

# --- CẤU HÌNH ---
INPUT_DIR = 'MP_Data'
OUTPUT_DIR = 'MP_Data_Augmented'

# Làm sạch thư mục cũ để tránh lẫn lộn
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

actions = [name for name in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, name))]
print(f"🚀 Bắt đầu nhân bản x10 dữ liệu cho các chữ cái: {actions}")

def generate_10x_variants(sequence):
    """
    Hàm này nhận vào 1 video (ma trận 30x78) 
    và trả về 10 phiên bản toán học khác nhau của nó.
    """
    variants = {}
    
    # Bản số 1: Giữ nguyên vẹn
    variants['00_goc'] = sequence
    
    # --- NHÓM 1: Bơm nhiễu ngẫu nhiên (Giả lập camera mờ, tay run) ---
    variants['01_nhieu_nhe'] = sequence + np.random.normal(0, 0.005, sequence.shape)
    variants['02_nhieu_vua'] = sequence + np.random.normal(0, 0.010, sequence.shape)
    variants['03_nhieu_manh'] = sequence + np.random.normal(0, 0.015, sequence.shape)
    
    # --- TÁCH LỚP DỮ LIỆU ---
    # Vì 63 số đầu là tọa độ, 15 số cuối là góc, ta phải xử lý riêng để không làm hỏng tính chất vật lý
    coords = sequence[:, :63] # Lấy 63 cột đầu
    angles = sequence[:, 63:] # Lấy 15 cột cuối
    
    # --- NHÓM 2: Phóng to/Thu nhỏ khung xương (Giả lập bàn tay người lớn/trẻ em) ---
    # Chỉ nhân tỷ lệ với tọa độ, giữ nguyên góc (vì bàn tay to hay nhỏ thì góc gập vẫn thế)
    variants['04_tay_to_nhe'] = np.concatenate([coords * 1.05, angles], axis=1)
    variants['05_tay_to_vua'] = np.concatenate([coords * 1.10, angles], axis=1)
    variants['06_tay_nho_nhe'] = np.concatenate([coords * 0.95, angles], axis=1)
    variants['07_tay_nho_vua'] = np.concatenate([coords * 0.90, angles], axis=1)
    
    # --- NHÓM 3: Độ mở của khớp (Giả lập người gập tay chặt/lỏng hơn một chút) ---
    # Chỉ cộng/trừ vào góc, giữ nguyên tọa độ. Giới hạn (clip) góc không vượt quá [0, 1]
    variants['08_gap_chat_hon'] = np.concatenate([coords, np.clip(angles + 0.02, 0, 1)], axis=1)
    variants['09_duoi_thang_hon'] = np.concatenate([coords, np.clip(angles - 0.02, 0, 1)], axis=1)
    
    return variants

# Bắt đầu duyệt qua từng thư mục để nhân bản
total_videos_created = 0

for action in actions:
    src_action_path = os.path.join(INPUT_DIR, action)
    dst_action_path = os.path.join(OUTPUT_DIR, action)
    os.makedirs(dst_action_path, exist_ok=True)
    
    video_folders = os.listdir(src_action_path)
    
    for video_folder in video_folders:
        src_video_path = os.path.join(src_action_path, video_folder)
        if not os.path.isdir(src_video_path): continue
        
        file_list = sorted(os.listdir(src_video_path), key=lambda x: int(os.path.splitext(x)[0]))
        if len(file_list) < 30: continue # Bỏ qua video bị lỗi rớt frame

        # Load 30 frame của 1 video lên
        frames = [np.load(os.path.join(src_video_path, f)) for f in file_list]
        original_sequence = np.array(frames) # Kích thước (30, 78)

        # Sinh ra 10 biến thể
        ten_variants = generate_10x_variants(original_sequence)

        # Lưu 10 biến thể này thành 10 thư mục video mới
        for var_name, data_matrix in ten_variants.items():
            new_video_name = f"{video_folder}_{var_name}"
            new_video_path = os.path.join(dst_action_path, new_video_name)
            os.makedirs(new_video_path, exist_ok=True)
            
            for frame_num in range(30):
                np.save(os.path.join(new_video_path, f"{frame_num}.npy"), data_matrix[frame_num])
            
            total_videos_created += 1

print(f"\n✅ HOÀN TẤT! Đã nhân bản thành công.")
print(f"📊 Tổng số video hiện có trong kho dữ liệu ({OUTPUT_DIR}): {total_videos_created} video")