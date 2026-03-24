import cv2
import numpy as np
import os
import mediapipe as mp

# --- CẤU HÌNH ---
DATA_PATH = os.path.join('MP_Data') 

# DANH SÁCH 10 KÝ HIỆU (Bạn hãy sửa lại tên cho đúng ý muốn)
# Lưu ý: Không dùng dấu cách, không dùng tiếng Việt có dấu
actions = np.array([
    'Xin_chao', 
    'Tam_biet', 
    'Toi_yeu_ban', 
    'Choi', 
    'Tra_da', 
    'Tra_sua', 
    'Nha_ve_sinh', 
    'Xin_loi', 
    'Xe_may', 
    'Trung'
])

no_sequences = 30 # Số lượng video cho mỗi hành động (30 video)
sequence_length = 30 # Độ dài mỗi video (30 frame)

# --- TẠO THƯ MỤC ---
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# --- KHỞI TẠO MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# --- HÀM TRÍCH XUẤT TOẠ ĐỘ ---
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        # Lấy tay đầu tiên
        hand_landmarks = results.multi_hand_landmarks[0]
        row = []
        for res in hand_landmarks.landmark:
            row.append(res.x)
            row.append(res.y)
            row.append(res.z)
        return np.array(row)
    else:
        return np.zeros(21*3)

# --- BẮT ĐẦU VÒNG LẶP CHÍNH ---
for action in actions:
    # Loop qua từng video (0 -> 29)
    for sequence in range(no_sequences):
        
        # --- GIAI ĐOẠN 1: CHỜ NGƯỜI DÙNG BẤM NÚT ---
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Hiển thị hướng dẫn
            cv2.putText(frame, 'Dang thu thap: {}'.format(action), (15,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Video so {}'.format(sequence), (15,60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Bam "SPACE" de quay...', (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Thu thap data', frame)
            
            # Nếu bấm SPACE thì thoát vòng lặp chờ, bắt đầu quay
            if cv2.waitKey(10) & 0xFF == ord(' '):
                break

        # --- GIAI ĐOẠN 2: QUAY 30 FRAME ---
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Xử lý MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            
            # Vẽ xương tay
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            # Thông báo đang quay (Màu đỏ)
            cv2.putText(frame, 'DANG QUAY...', (15,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Lưu dữ liệu
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            cv2.imshow('Thu thap data', frame)
            cv2.waitKey(1) # Quay nhanh nhất có thể

    print(f"--- Xong hành động {action} ---")

cap.release()
cv2.destroyAllWindows()