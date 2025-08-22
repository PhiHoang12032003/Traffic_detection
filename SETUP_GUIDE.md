# Hướng dẫn Cài đặt - Traffic Red Light Running Violation Detection

## Yêu cầu hệ thống

- Python 3.8 hoặc cao hơn
- pip package manager
- Ít nhất 4GB RAM
- CUDA-compatible GPU (tuỳ chọn, để tăng tốc AI processing)

## Cài đặt nhanh

### 1. Clone repository (nếu chưa có)

```bash
git clone <repository-url>
cd python_project
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng

```bash
python app_server.py
```

Ứng dụng sẽ tự động mở trình duyệt tại `http://127.0.0.1:8000/`

## Chức năng có sẵn

### ✅ Hoạt động không cần MySQL:

- **Phát hiện vượt đèn đỏ** (/Hethongcamera3)
- **Phát hiện vi phạm mũ bảo hiểm** (/Hethongcamera2)
- **Phát hiện vi phạm làn đường** (/Hethongcamera1)
- **Xử lý video và tạo báo cáo PDF**
- **API endpoints cho thống kê cơ bản**

### 📊 Thống kê (MySQL đã được tắt):

- Routes `/test` và `/test1` vẫn hoạt động với dữ liệu mẫu
- Không cần cài đặt MySQL Server
- Dữ liệu thống kê sử dụng dữ liệu mẫu thay vì database

## Cấu trúc thư mục quan trọng

```
python_project/
├── app_server.py           # Main Flask application
├── requirements.txt        # Python dependencies
├── uploads/               # Uploaded videos
├── processed_videos/      # Processed output videos
├── templates/             # HTML templates
├── YoloWeights/          # AI model weights
├── best_new/             # Latest model weights
└── data_*/               # Generated violation images
```

## Troubleshooting

### Lỗi thiếu dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Lỗi CUDA (nếu có GPU):

```bash
# Cài đặt CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Lỗi OpenCV:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

## Tái kích hoạt MySQL (tuỳ chọn)

Nếu muốn sử dụng chức năng thống kê với database:

1. **Cài đặt MySQL Server**
2. **Tạo database tên 'datn'**
3. **Uncomment các dòng MySQL trong app_server.py:**
   ```python
   # Dòng 11: from flask_mysqldb import MySQL
   # Dòng 37-42: MySQL configuration
   # Và các try/except blocks trong functions
   ```
4. **Cài đặt MySQL dependencies:**
   ```bash
   pip install Flask-MySQLdb mysqlclient PyMySQL
   ```

## Liên hệ

Nếu gặp vấn đề trong quá trình cài đặt, vui lòng liên hệ để được hỗ trợ.
