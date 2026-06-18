# 🚦 Nhận Diện Biển Báo Giao Thông

> Hệ thống phát hiện và phân loại biển báo giao thông sử dụng YOLOv8 và YOLOv11.
---

## 📋 Mục Lục

- [Tổng quan](#-tổng-quan)
- [Tính năng](#-tính-năng)
- [Các loại biển báo](#-các-loại-biển-báo)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Mô hình](#-mô-hình)
- [Bộ dữ liệu](#-bộ-dữ-liệu)
- [Kết quả](#-kết-quả)
- [Tác giả](#-tác-giả)

---

## 📖 Tổng Quan

Dự án được thực hiện trong khuôn khổ môn học **CS406** tại **Trường Đại học Công nghệ Thông tin (UIT)**. Hệ thống xây dựng pipeline phát hiện biển báo giao thông hoàn chỉnh bao gồm:

- Huấn luyện hai mô hình phát hiện đối tượng YOLO (YOLOv8l và YOLOv11l) trên bộ dữ liệu biển báo giao thông tùy chỉnh.
- So sánh song song hiệu năng của hai mô hình về **độ chính xác phát hiện** và **tốc độ suy luận**.
- Cung cấp **giao diện web demo** trực quan bằng Streamlit để người dùng tải ảnh lên và xem kết quả phát hiện ngay lập tức.

---

## ✨ Tính Năng

- 🔍 **Phát hiện biển báo** từ ảnh tải lên
- 🤖 **Chạy song song hai mô hình** — YOLOv8l và YOLOv11l cùng lúc
- ⏱️ **So sánh thời gian suy luận** giữa hai mô hình
- 📦 **Vẽ bounding box** kèm nhãn lớp và điểm tin cậy (confidence score)
- 🖥️ **Giao diện web tương tác** xây dựng bằng Streamlit

---

## 🚸 Các Loại Biển Báo

Mô hình nhận diện **4 loại** biển báo giao thông:

| ID | Loại biển báo | Mô tả |
|----|--------------|-------|
| 0 | 🚫 **Biển cấm** | Các biển cấm thực hiện hành động (ví dụ: cấm vào, giới hạn tốc độ) |
| 1 | ⚠️ **Biển cảnh báo** | Các biển cảnh báo nguy hiểm phía trước cho người lái xe |
| 2 | ✅ **Biển hiệu lệnh** | Các biển chỉ dẫn hành động bắt buộc (ví dụ: hướng rẽ) |
| 3 | ℹ️ **Biển thông tin** | Các biển cung cấp thông tin chung cho người tham gia giao thông |

---

## 📁 Cấu Trúc Dự Án

```
Do_An/
├── 22520270/
│   ├── README.md                        # Tệp này
│   ├── docs/
│   │   ├── 22520270.docx                # Báo cáo dự án
│   │   └── Traffic Sign Detection.pptx  # Slide thuyết trình
│   └── source/
│       └── demo.py                      # Ứng dụng Streamlit demo
└── dataset/
    ├── best_150epoch_v8l.pt             # Trọng số YOLOv8l (~87MB)
    ├── best_150epochs_v11l.pt           # Trọng số YOLOv11l (~51MB)
    ├── data.yaml                        # Cấu hình bộ dữ liệu
    ├── types.txt                        # Danh sách nhãn lớp
    └── labels/
        ├── train/                       # Nhãn tập huấn luyện
        └── val/                         # Nhãn tập kiểm định
```

---

## 🛠️ Yêu Cầu Hệ Thống

- Python **3.8+**
- pip

---

## ⚙️ Cài Đặt

**1. Clone repository**

```bash
git clone <đường-dẫn-repository>
cd Do_An
```

**2. Cài đặt các thư viện cần thiết**

```bash
pip install ultralytics streamlit opencv-python pillow numpy matplotlib
```

**3. Tải bộ dữ liệu và trọng số mô hình**

Tải bộ dữ liệu từ Kaggle:
👉 [https://www.kaggle.com/datasets/nhd1311/cs406-data](https://www.kaggle.com/datasets/nhd1311/cs406-data)

Đặt các tệp trọng số vào thư mục `dataset/`:
```
dataset/
├── best_150epoch_v8l.pt
└── best_150epochs_v11l.pt
```

**4. Cập nhật đường dẫn trọng số trong `demo.py`**

Mở tệp `source/demo.py` và cập nhật biến `weights_paths` cho đúng với đường dẫn trên máy của bạn:

```python
weights_paths = {
    "YOLOv8":  "đường/dẫn/đến/dataset/best_150epoch_v8l.pt",
    "YOLOv11": "đường/dẫn/đến/dataset/best_150epochs_v11l.pt",
}
```

---

## 🚀 Hướng Dẫn Sử Dụng

Di chuyển vào thư mục `source/` và chạy ứng dụng Streamlit:

```bash
cd 22520270/source
streamlit run demo.py
```

Sau đó mở trình duyệt tại `http://localhost:8501` và thực hiện:

1. **Tải ảnh lên** (định dạng `.jpg`, `.jpeg` hoặc `.png`)
2. **Xem kết quả** dự đoán từ YOLOv8 và YOLOv11 cạnh nhau
3. **So sánh** thời gian suy luận và chất lượng phát hiện giữa hai mô hình

---

## 🤖 Mô Hình

| Mô hình | Backbone | Số epoch | Kích thước trọng số |
|---------|----------|----------|---------------------|
| **YOLOv8l** | Large | 150 | ~87 MB |
| **YOLOv11l** | Large | 150 | ~51 MB |

Cả hai mô hình đều được huấn luyện trên cùng bộ dữ liệu với cấu hình định nghĩa trong `dataset/data.yaml`.

---

## 📊 Bộ Dữ Liệu

- **Nguồn:** [Kaggle - cs406-data](https://www.kaggle.com/datasets/nhd1311/cs406-data)
- **Số lớp:** 4 (Biển cấm, Biển cảnh báo, Biển hiệu lệnh, Biển thông tin)
- **Phân chia:** Tập huấn luyện (train) / Tập kiểm định (val)
- **Định dạng nhãn:** YOLO annotation format (tệp `.txt`)

---

## 📈 Kết Quả


| Mô hình | mAP@50 | mAP@50-95 |
|---------|--------|-----------|
| YOLOv8l | 0.953 | 0.828 |
| YOLOv11l | 0.954 | 0.82 | 

---
