# Tổng quan về dữ liệu Augmented Data

Thư mục này chứa dữ liệu âm thanh đã được tăng cường (augmented) cho dự án nhận diện người nói dựa trên giọng nói.

## Cấu trúc thư mục

- **RIR Method/**: Dữ liệu tăng cường sử dụng phương pháp Room Impulse Response (RIR). Mô phỏng âm thanh trong các phòng khác nhau với các hệ số suy giảm khác nhau.
- **Simple Noise/**: Dữ liệu tăng cường sử dụng các kỹ thuật đơn giản như thêm nhiễu, thay đổi tốc độ, thay đổi cao độ, dịch chuyển và điều chỉnh âm lượng.

## Danh sách người nói

Mỗi phương pháp tăng cường chứa dữ liệu cho 10 người nói sau:
- Chau_Anh
- Danh_Son
- Huong_Ly
- Jenifer_Smith
- Long_Hai
- Minh_Tam
- Phuong_Anh
- Thomas_Williams
- Tran_Quyen
- Van_Anh

## Chi tiết dữ liệu

- **RIR Method**: Mỗi người nói có 20 tệp âm thanh WAV, được đặt tên theo định dạng `voice_rir_X_decay_Y.wav`, trong đó:
  - X: Số thứ tự từ 1 đến 20
  - Y: Hệ số suy giảm từ 0.02 đến 0.21

- **Simple Noise**: Mỗi người nói có 20 tệp âm thanh WAV, được đặt tên theo định dạng `aug_X_type.wav`, trong đó:
  - X: Số thứ tự từ 1 đến 20
  - Type: Các loại tăng cường như `noise`, `stretch`, `pitch`, `shift`, `volume`

## Tổng số tệp

- Tổng cộng: 400 tệp âm thanh (2 phương pháp × 10 người nói × 20 tệp mỗi người nói)

## Mục đích sử dụng

Dữ liệu này được sử dụng để huấn luyện và đánh giá mô hình nhận diện người nói, giúp cải thiện độ chính xác bằng cách tăng cường đa dạng dữ liệu âm thanh.