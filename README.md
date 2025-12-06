# TNGT_NEWS_IE
```
TNGT_NEWS_IE/
│
├── data/                       # Nơi chứa toàn bộ dữ liệu
│   ├── raw/                 # Dữ liệu thô vừa crawl về (CSV gốc)
│   ├── label_studio/        # File JSON để import/export từ Label Studio
│   └── processed/           # Dữ liệu đã gán nhãn xong & chia tập (train/dev/test)
│
├── models/                     # Nơi lưu các file model đã train (.pkl, .pt, .h5)
│   ├── ner/                    # Lưu model NER (CRF, SVM, BERT...)
│   └── re/                     # Lưu model RE (SVM, BERT...)
│
├── notebooks/                  # Nơi chạy thử nghiệm (Jupyter Notebooks)
│   ├── 1_data_exploration.ipynb
│   ├── 2_ner_ml_experiments.ipynb  # Chạy 3 model ML cho NER
│   ├── 3_ner_dl_experiments.ipynb  # Chạy model DL cho NER
│   ├── 4_re_ml_experiments.ipynb   # Chạy 3 model ML cho RE
│   └── 5_re_dl_experiments.ipynb   # Chạy model DL cho RE
│
├── src/                        # Mã nguồn chính (Core logic) - Dùng để tái sử dụng
│   ├── __init__.py
│   ├── preprocessing.py        # Chứa hàm `core_preprocessor` và regex (đã làm ở trên)
│   ├── features.py             # Feature Engineering cho ML (trích xuất đặc trưng từ câu)
│   ├── ner_models.py           # Class định nghĩa các model NER
│   └── re_models.py            # Class định nghĩa các model RE
│
├── reports/                    # Lưu kết quả so sánh
│   ├── figures/                # Biểu đồ so sánh F1-score, Confusion Matrix
│   └── logs/                   # Log huấn luyện
│
├── app/              # Thư mục chứa code UI Demo
│   └── app.py                  # File chạy chính
│
├── requirements.txt            # Các thư viện cần thiết (sklearn, transformers, spacy...)
└── README.md                   # Hướng dẫn chạy dự án
```

# Tạo môi trường ảo và cài đặt thư viện
```bash
python -m venv venv
venv\Scripts\activate  # Trên PowerShell dùng `.\venv\Scripts\activate
pip install -r requirements.txt
```