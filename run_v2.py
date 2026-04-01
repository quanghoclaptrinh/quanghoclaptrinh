import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- 1. CẤU HÌNH HỆ THỐNG ---
DATA_PATH = 'MP_Data_Augmented' 
MODEL_NAME = 'action_bilstm_v2_best.h5' # Gọi đúng tên model xịn nhất vừa train
THRESHOLD = 0.85      # Độ tự tin tối thiểu (85%) AI mới dám nhận
CONFIRM_FRAMES = 15   # Giữ tay yên 15 frame (~0.5s) để chốt chữ

# Load danh sách nhãn (A, B, C...) và Mô hình AI
actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
print(f"🚀 Đang nạp bộ não AI: {MODEL_NAME}...")
model = load_model(MODEL_NAME)
print("✅ Nạp thành công! Chuẩn bị mở Camera...")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- 2. HÀM TRÍCH XUẤT 78 ĐẶC TRƯNG (Giữ nguyên y hệt lúc Train) ---
def compute_angles(image_points):
    angles = []
    angle_list = [(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),
                  (0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),
                  (0,17,18),(17,18,19),(18,19,20)]
    for (p1, p2, p3) in angle_list:
        v1 = image_points[p1] - image_points[p2]
        v2 = image_points[p3] - image_points[p2]
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)
    return angles

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        pts = np.array([[r.x, r.y, r.z] for r in results.multi_hand_landmarks[0].landmark])
        angles = compute_angles(pts)
        pts = pts - pts[0]
        max_val = np.max(np.abs(pts))
        if max_val > 0: pts = pts / max_val
        return np.concatenate([pts.flatten(), angles])
    else: 
        return np.zeros(78)

# --- 3. LOGIC CHẠY THỰC TẾ & BÀN PHÍM ĐÁNH VẦN ---
sequence = []
current_word = ""  
stable_count = 0   
final_text = ""    

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) # Lật gương cho dễ nhìn
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Vẽ khung xương lên màn hình
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            
        # Trích xuất 78 con số và nạp vào băng chuyền
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Chỉ giữ lại đúng 30 frame gần nhất
        
        # Khi băng chuyền đủ 30 frame -> Bắt đầu đoán
        if len(sequence) == 30:
            # Model nhận vào Tensor (1, 30, 78)
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_action = actions[np.argmax(res)]
            confidence = res[np.argmax(res)]
            
            # Logic Bàn phím: Chỉ xét khi AI cực kỳ tự tin (> 85%)
            if confidence > THRESHOLD:
                if predicted_action == current_word:
                    stable_count += 1
                else:
                    current_word = predicted_action
                    stable_count = 0
                
                # Giữ nguyên tay đủ 15 frame -> In chữ đó ra màn hình
                if stable_count == CONFIRM_FRAMES:
                    final_text += current_word + " "
                    stable_count = 0 # Reset đếm để gõ chữ tiếp theo
            else:
                # Nếu đưa tay bậy bạ, AI không tự tin -> Hủy đếm
                stable_count = 0
                
        # --- 4. GIAO DIỆN NGƯỜI DÙNG (UI) ---
        # Vẽ hộp màu đen mờ trên cùng để chứa Text
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Cau cua ban: {final_text}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Vẽ hướng dẫn dưới cùng
        cv2.putText(frame, "C: Xoa trang | Q: Thoat", (10, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Hiển thị độ tự tin của chữ đang xét (để bạn biết AI nhạy cỡ nào)
        if len(sequence) == 30:
            cv2.putText(frame, f"AI dang nhin thay: {actions[np.argmax(res)]} ({int(res[np.argmax(res)]*100)}%)", 
                        (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.imshow('Ban Phim Thu Ngu AI', frame)
        
        # Xử lý phím bấm
        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'): final_text = "" # Bấm C để xóa câu gõ lại từ đầu
        if key == ord('q'): break           # Bấm Q để tắt camera

cap.release()
cv2.destroyAllWindows()