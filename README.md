# 🤟 Real-Time Sign Language Recognition using Feature Fusion & BiLSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)

Hệ thống nhận diện Ngôn ngữ Ký hiệu (Sign Language Recognition - SLR) theo thời gian thực. Dự án này biến camera của bạn thành một "Bàn phím đánh vần thủ ngữ" thông qua việc phân tích chuỗi chuyển động Không gian - Thời gian (Spatio-Temporal Sequence).

> **💡 Điểm nhấn Kỹ thuật:** Thay vì nạp ảnh thô, dự án sử dụng thuật toán **Dung hợp Đặc trưng (Feature Fusion)** trích xuất Vector 78 chiều mang tính bất biến cao (chống dịch chuyển, phóng to, lật góc) kết hợp với kiến trúc mạng **Hồi quy Hai chiều (BiLSTM)** để tối ưu hóa độ chính xác.

---

## 🎥 Demo Thực tế
*(Bạn hãy dùng phần mềm quay màn hình, quay lại cảnh bạn đang giơ tay đánh vần thành một câu hoàn chỉnh, tạo thành file `.gif` và kéo thả vào đây nhé!)*

![Demo Nhận diện Thủ ngữ](link_den_file_gif_cua_ban.gif)

---

## 🚀 Các Tính Năng Nổi Bật (Key Features)

* **Trích xuất Đặc trưng Bất biến (Invariant Feature Extraction):** * Sử dụng MediaPipe bóc tách 21 điểm khớp xương (Landmarks).
  * Chuyển hóa thành mảng **78 chiều** bao gồm: **63 tọa độ tương đối** (đã dời gốc về cổ tay & chuẩn hóa Min-Max) và **15 góc khớp xương** (Joint Angles).
  * Hệ thống hoàn toàn miễn nhiễm với việc người dùng đứng xa/gần camera, để tay lệch khung hình hay xoay nghiêng bàn tay.
* **Tăng cường Dữ liệu Toán học (Data Augmentation x10):** * Tiêm nhiễu Gaussian (Noise Injection) và biến đổi tỷ lệ không gian (Scale/Angle Shifting) để nhân bản bộ dữ liệu lên gấp 10 lần, chống hiện tượng Overfitting.
* **Suy luận Chuỗi Thời gian (Time-Series Inference):** * Sử dụng cơ chế Cửa sổ trượt (Sliding Window) 30 frames.
  * Tích hợp bộ đếm ổn định (Stability Counter) để loại bỏ nhiễu chập chờn, tạo cảm giác gõ phím mượt mà như bàn phím thật.

---

## 🧠 Kiến trúc Hệ thống

1. **Input:** Camera luân chuyển khung hình (30 FPS).
2. **Feature Extraction:** `MediaPipe` -> Tính toán không gian -> `Vector(78)`.
3. **Data Structuring:** Gom 30 vectors thành một `Tensor(30, 78)`.
4. **Deep Learning Model:** * `Bidirectional LSTM (64)` -> `Dropout(0.2)`
   * `Bidirectional LSTM (128)` -> `Dropout(0.2)`
   * `Bidirectional LSTM (64)`
   * `Dense (64)` -> `Softmax`
5. **Output:** Ký tự thủ ngữ được dự đoán với độ tự tin (Confidence > 85%).

---

## ⚙️ Hướng dẫn Cài đặt (Installation)

1. Clone kho lưu trữ này về máy:
```bash
git clone [https://github.com/TenCuaBan/TenRepo.git](https://github.com/TenCuaBan/TenRepo.git)
cd TenRepo
