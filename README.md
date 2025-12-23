# Phân tích và Trích xuất Thông tin Tai nạn Giao thông (TNGT_NEWS_IE)

Tai nạn giao thông là vấn đề nhức nhối tại Việt Nam. Dự án này nhằm mục đích cấu trúc hóa dữ liệu phi cấu trúc từ báo chí để phục vụ công tác thống kê và phân tích. Hệ thống tập trung giải quyết hai bài toán chính:

1. **Named Entity Recognition (NER):** Xác định các thực thể như người lái xe, nạn nhân, phương tiện, nguyên nhân, hậu quả....
2. **Relation Extraction (RE):** Xác định mối quan hệ giữa các thực thể (ví dụ: Nguyên nhân dẫn đến Tai nạn).

## Dữ liệu (Dataset)

Dữ liệu được thu thập từ các trang báo điện tử uy tín tại Việt Nam thông qua kỹ thuật Web Crawling.

* **Công cụ gán nhãn:** Label Studio.
* **Quy mô dữ liệu:** 15.816 thực thể và 7.856 quan hệ.
* **Chia tập dữ liệu:** Train (80%) - Validation (10%) - Test (20%).

### Hệ thống nhãn (Label Scheme)

Dự án đề xuất hệ thống nhãn hướng "vai trò" (Role-oriented) đặc thù cho miền giao thông:

**Thực thể (Entities):**

* `PER_DRIVER`: Tài xế/Người điều khiển.
* `PER_VICTIM`: Nạn nhân.
* `VEH`: Phương tiện (xe máy, ô tô...).
* `CAUSE`: Nguyên nhân (say rượu, mất lái...).
* `CONSEQUENCE`: Hậu quả (tử vong, bị thương, hư hỏng).
* `EVENT`, `LOC`, `ORG`, `TIME`.

**Quan hệ (Relations):**

* `INVOLVED`: Quan hệ giữa sự kiện/người với phương tiện.
* `CAUSED_BY`: Quan hệ nhân quả.
* `HAS_CONSEQUENCE`: Quan hệ dẫn đến hậu quả.
* `LOCATED_AT`, `HAPPENED_ON`.

## Phương pháp & Mô hình

Dự án thực nghiệm so sánh giữa các phương pháp Học máy truyền thống và Học sâu (Deep Learning).

### 1. Nhận dạng thực thể (NER)

* **Baseline:** Logistic Regression, SVM, CRF (kết hợp embedding từ PhoBERT).
* **Deep Learning:** PhoBERT-base (Fine-tuned).

### 2. Trích xuất quan hệ (RE)

* **Phương pháp:** Sử dụng kỹ thuật **Typed Entity Markers** để đánh dấu thực thể trong câu.
* **Baseline:** SVM, Random Forest, Logistic Regression.
* **Deep Learning:** PhoBERT-base (Fine-tuned với Linear Classification Head).

## Cài đặt

```bash
# Clone repository
git clone https://github.com/Sura3607/TNGT_NEWS_IE.git
cd TNGT_NEWS_IE

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Chạy ứng dụng web
streamlit run app/app.py

```


