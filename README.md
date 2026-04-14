# Identify Person by Voice

## Overview
This project focuses on researching, experimenting, and training machine learning models to identify and authenticate users based on voice characteristics.

## Objectives
- Research voice identification techniques and algorithms
- Experiment with different feature extraction methods
- Build and train speaker identification models
- Evaluate model performance and accuracy

## Key Topics
- Voice signal processing
- Feature extraction (MFCC, spectrograms, log-Mel)
- Speaker embedding models
- Deep learning architectures for voice recognition

## Dataset      // cần tìm thêm data, ưu tiên những audio có cùng nội dung để vector không bị phân tán bởi ngữ cảnh.
- Hugging Face dataset (100 speakers):
	https://huggingface.co/datasets/thucdangvan020999/speaker_identification_100_speakers_


| Package | Dùng để |
|---|---|
| datasets | load HF dataset |
| torchaudio | xử lý audio |
| librosa | preprocess |
| soundfile | đọc/ghi audio |
| numpy | array |
| scikit-learn | cosine similarity |
| transformers | WavLM |
| faiss-cpu | vector search |

## Pipeline
```text
[DATA]
	↓
Raw audio
	↓
[PREPROCESS]
(resample, trim, normalize, VAD)
	↓
[FEATURE / EMBEDDING]
WavLM / MFCC / ECAPA
	↓
[REGISTER PHASE]
User registration -> embedding -> vector DB
	↓
[INFERENCE PHASE]
New audio -> embedding -> similarity search
	↓
[OUTPUT]
Speaker ID + confidence score
```


librosa.util.normalize(S, norm=inf, axis=0, threshold=None, fill=None)

Chuẩn hóa theo norm
L∞ (mặc định, norm=inf)

Mỗi vector (theo trục axis) được chia cho giá trị tuyệt đối lớn nhất.

Kết quả: biên độ nằm trong [-1, 1].

Mục đích: giữ nguyên hình dạng tín hiệu nhưng giới hạn biên độ.

L1 (norm=1)

Mỗi vector được chia cho tổng giá trị tuyệt đối.

Kết quả: tổng biên độ = 1.

Mục đích: chuẩn hóa để so sánh phân bố năng lượng giữa các tín hiệu.

L2 (norm=2)

Mỗi vector được chia cho căn bậc hai của tổng bình phương.

Kết quả: độ dài vector = 1.

Mục đích: chuẩn hóa để dùng trong các mô hình học máy, đặc biệt khi cần dữ liệu có cùng “magnitude”.

2. Tham số axis
axis=0: chuẩn hóa theo cột (mỗi cột độc lập).

axis=1: chuẩn hóa theo hàng (mỗi hàng độc lập).

Mục đích: linh hoạt khi xử lý ma trận đặc trưng (ví dụ spectrogram).

3. Tham số threshold
Nếu norm của một vector nhỏ hơn ngưỡng threshold, có thể:

Giữ nguyên (không chuẩn hóa).

Đặt toàn bộ thành 0.

Hoặc điền giá trị đồng đều (fill).

Mục đích: tránh chia cho số quá nhỏ gây nhiễu hoặc lỗi.

4. Tham số fill
Nếu một vector quá nhỏ (dưới ngưỡng), có thể gán giá trị đồng đều để norm = 1.

Mục đích: đảm bảo dữ liệu không bị mất thông tin hoàn toàn.

Tóm lại
L∞: chuẩn hóa biên độ ([-1, 1]) → thường dùng cho tín hiệu âm thanh.

L1: chuẩn hóa tổng năng lượng = 1 → dùng khi so sánh phân bố năng lượng.

L2: chuẩn hóa độ dài vector = 1 → dùng trong học máy, giảm ảnh hưởng của cường độ.

threshold/fill: xử lý trường hợp tín hiệu quá nhỏ hoặc gần như im lặng.


## Project Structure
```
. 
├── data/
├── models/
├── notebooks/
├── src/
└── README.md
```

## Getting Started
1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt` (if provided).
4. Prepare dataset under `data/`.
5. Run preprocessing, training, and evaluation scripts from `src/`.

## References
- Add relevant papers, blog posts, and official documentation here.
