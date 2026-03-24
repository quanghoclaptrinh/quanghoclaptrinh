import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- CẤU HÌNH ---
DATA_PATH = 'MP_Data_Augmented' 
MODEL_NAME = 'action.h5'     # Hoặc action.h5
THRESHOLD = 0.85                # Độ tin cậy
DELAY_FRAMES = 30               # 30 frame ~ 1 giây (Muốn lâu hơn thì tăng lên 40, 50)

# --- 1. LOAD NHÃN ---
if os.path.exists(DATA_PATH):
    actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
    print(f"--> Actions: {actions}")
else:
    # Nếu không có data thì điền tay vào đây cho đúng thứ tự lúc train
    actions = np.array(['Xin_chao', 'Cam_on', 'So_1', 'So_2']) 

# --- 2. SINH MÀU NGẪU NHIÊN ---
colors = []
for i in range(len(actions)):
    colors.append((np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)))

# --- 3. LOAD MODEL ---
try:
    model = load_model(MODEL_NAME)
except:
    # Thử load h5 nếu keras không thấy
    model = load_model('action.h5')

# --- HÀM VẼ ---
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {int(prob*100)}%", (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# --- KHỞI TẠO ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

sequence = []
sentence = []

# --- BIẾN ĐỂ DELAY 1 GIÂY ---
current_action_check = "" # Hành động đang nghi ngờ
stability_counter = 0     # Đếm số frame ổn định

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # TRÍCH XUẤT KEYPOINTS
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Lấy toạ độ
        rh = results.multi_hand_landmarks[0]
        row = []
        for res in rh.landmark:
            row.append(res.x); row.append(res.y); row.append(res.z)
        keypoints = np.array(row)
    else:
        keypoints = np.zeros(21*3)

    # SLIDING WINDOW
    sequence.append(keypoints)
    sequence = sequence[-30:]

    # DỰ ĐOÁN
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        best_idx = np.argmax(res)
        predicted_action = actions[best_idx]
        
        # --- LOGIC DELAY 1 GIÂY (MỚI) ---
        
        # 1. Kiểm tra xem máy đoán có giống frame trước không?
        if predicted_action == current_action_check:
            stability_counter += 1 # Nếu giống thì cộng điểm uy tín
        else:
            current_action_check = predicted_action # Nếu khác thì reset, nghi ngờ cái mới
            stability_counter = 0
            
        # 2. Nếu điểm uy tín đủ 30 (tức là ổn định trong 1 giây)
        if stability_counter >= DELAY_FRAMES:
            
            # 3. Kiểm tra thêm độ tin cậy (Threshold)
            if res[best_idx] > THRESHOLD:
                
                # 4. Chỉ hiển thị nếu khác từ vừa nói xong (để không bị nói lắp)
                if len(sentence) > 0:
                    if current_action_check != sentence[-1]:
                        sentence.append(current_action_check)
                        # stability_counter = 0 # (Tuỳ chọn) Reset để bắt đầu chu trình mới
                else:
                    sentence.append(current_action_check)

        if len(sentence) > 5: sentence = sentence[-5:]
        
        # Vẽ thanh xác suất
        if results.multi_hand_landmarks:
            frame = prob_viz(res, actions, frame, colors)
        
        # Vẽ thanh loading (Trực quan hoá độ trễ cho đẹp)
        # Nếu đang đếm, vẽ thanh màu xanh dưới đáy để biết máy đang "nghĩ"
        if stability_counter > 0 and stability_counter < DELAY_FRAMES:
            loading_width = int((stability_counter / DELAY_FRAMES) * 640)
            cv2.rectangle(frame, (0, 460), (loading_width, 480), (0, 255, 255), -1) # Thanh màu vàng
            cv2.putText(frame, "Dang xac nhan...", (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Hiển thị
    cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(frame, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Nhan dien Thu ngu (Delay 1s)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()