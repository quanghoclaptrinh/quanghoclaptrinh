import cv2
import numpy as np
import os
import mediapipe as mp

# --- 1. CẤU HÌNH ---
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['A', 'B', 'C']) # Thay đổi chữ cái ở đây
no_sequences = 30 
sequence_length = 30 

for action in actions: 
    for sequence in range(no_sequences):
        try: os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except: pass

# --- 2. HÀM TOÁN HỌC TRÍCH XUẤT 78 ĐẶC TRƯNG ---
def compute_angles(image_points):
    angles = []
    angle_list = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),        
        (0, 5, 6), (5, 6, 7), (6, 7, 8),        
        (0, 9, 10), (9, 10, 11), (10, 11, 12),  
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        (0, 17, 18), (17, 18, 19), (18, 19, 20) 
    ]
    for (p1, p2, p3) in angle_list:
        v1 = image_points[p1] - image_points[p2]
        v2 = image_points[p3] - image_points[p2]
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle) * (180.0 / np.pi)
        angles.append(angle / 180.0) 
    return angles

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        points = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])
        angles = compute_angles(points) 
        
        points = points - points[0] 
        max_value = np.max(np.abs(points))
        if max_value > 0: points = points / max_value 
        
        relative_coords = points.flatten() 
        return np.concatenate([relative_coords, angles]) 
    else:
        return np.zeros(78)

# --- 3. KHỞI TẠO CAMERA & QUAY ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Cờ báo hiệu thoát chương trình
quit_flag = False 

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    for action in actions:
        if quit_flag: break # Thoát vòng lặp chữ cái
        
        for sequence in range(no_sequences):
            if quit_flag: break # Thoát vòng lặp video
            
            # --- MÀN HÌNH CHỜ ---
            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                
                # Cập nhật UI thông báo có nút Q
                cv2.putText(frame, f'SPACE: Quay | Q: Thoat (Chu: {action} - Video {sequence})', (10,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('Thu thap du lieu', frame)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord(' '): 
                    break # Bấm SPACE thì thoát vòng lặp chờ -> Bắt đầu quay
                elif key == ord('q'): 
                    quit_flag = True # Bấm Q thì bật cờ thoát
                    break
            
            if quit_flag: break # Thoát hẳn luôn không quay nữa
            
            # --- MÀN HÌNH QUAY DỮ LIỆU ---
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                
                if results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                
                cv2.putText(frame, 'DANG QUAY... (Q: Dung khan cap)', (15,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Thu thap du lieu', frame)
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                
                # Bấm Q lúc đang quay cũng thoát được
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    quit_flag = True
                    break
                    
            if quit_flag: break

# Đóng camera an toàn
cap.release()
cv2.destroyAllWindows()
if quit_flag:
    print("\n🛑 Đã dừng thu thập dữ liệu an toàn theo yêu cầu!")
else:
    print("\n✅ Tuyệt vời! Đã thu thập xong toàn bộ dữ liệu.")