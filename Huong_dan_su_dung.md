# Hướng Dẫn Sử Dụng Dự Án Speaker Identification

## 1. Cài Đặt Môi Trường

### 1.1. Cài đặt Python
- Yêu cầu Python >= 3.8.
- Cài đặt Python từ [python.org](https://www.python.org/downloads/).

### 1.2. Tạo môi trường ảo
```bash
python -m venv .venv
```

### 1.3. Kích hoạt môi trường ảo
- **Windows**:
```bash
source .venv/Scripts/activate
```
- **Linux/MacOS**:
```bash
source .venv/bin/activate
```

### 1.4. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

## 2. Chuẩn Bị Dữ Liệu

### 2.1. Cấu trúc thư mục dữ liệu
- Dữ liệu thô cần được đặt trong thư mục `data_real/` với cấu trúc như sau:
```
data_real/
    Speaker_1/
        audio_1.mp3
        audio_2.mp3
    Speaker_2/
        audio_1.mp3
        audio_2.mp3
```

### 2.2. Tiền xử lý dữ liệu
Chạy lệnh sau để chuẩn hóa dữ liệu trước khi cho vào các model xử lý
```bash
python D:\Master DS\Intro_to_DS\src\data\pipeline_real.py
```

## 3. Trích Xuất Embedding
Chạy lệnh sau để trích xuất embedding từ các model:
```bash
python extract_embedding_real.py --model wavlm
```
- Thay `wavlm` bằng các model khác như `wav2vec2`, `hubert`, `unispeech` nếu cần.

## 4. Xây Dựng FAISS Index
Chạy lệnh sau để xây dựng FAISS index:
```bash
python build_faiss_real.py --model wavlm
```
- Thay `wavlm` bằng các model khác nếu cần.

## 5. Đánh Giá Model
Chạy lệnh sau để đánh giá model:
```bash
python evaluate_real.py --model wavlm
```
- Để đánh giá tất cả các model:
```bash
python evaluate_real.py --all
```

## 6. Chạy Ứng Dụng Streamlit
Chạy lệnh sau để khởi động ứng dụng Streamlit:
```bash
cd src
streamlit run app.py
```

Ứng dụng sẽ chạy tại địa chỉ: [http://localhost:8501](http://localhost:8501)

## 7. Lưu Ý
- Đảm bảo các file dữ liệu và model đã được chuẩn bị đúng cấu trúc.
- Kiểm tra các file log để xử lý lỗi nếu có.