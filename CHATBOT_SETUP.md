# 🤖 HƯỚNG DẪN TÍCH HỢP CHATBOT GEMINI AI

## 📋 Tổng quan

Chatbot AI được tích hợp vào hệ thống Traffic Monitoring để hỗ trợ cán bộ công an tra cứu thông tin, thống kê vi phạm giao thông một cách nhanh chóng và thông minh.

## ✨ Tính năng

### 1. Tra cứu & Thống kê vi phạm

- Thống kê vi phạm theo ngày/tuần/tháng
- So sánh xu hướng vi phạm
- Lọc theo loại vi phạm, biển số xe, thời gian
- Phân tích điểm nóng vi phạm

### 2. Theo dõi hệ thống & Camera

- Kiểm tra trạng thái camera
- Giám sát hàng đợi xử lý video
- Kiểm tra log và lỗi hệ thống

### 3. Tra cứu quy định & mức phạt

- Tra cứu mức phạt theo loại vi phạm
- Thông tin về Nghị định xử phạt
- Quy trình xử lý vi phạm

### 4. Hỗ trợ báo cáo

- Tạo báo cáo thống kê
- Xuất dữ liệu Excel/PDF
- Gửi báo cáo qua email

## 🚀 Cài đặt

### Bước 1: Cài đặt thư viện

```bash
pip install google-generativeai python-dotenv
```

Hoặc cài đặt tất cả dependencies:

```bash
pip install -r requirements.txt
```

### Bước 2: Lấy API Key từ Google

1. Truy cập: https://makersuite.google.com/app/apikey
2. Đăng nhập với tài khoản Google
3. Click **"Create API Key"** hoặc **"Get API Key"**
4. Copy API key

### Bước 3: Cấu hình API Key

1. Copy file `.env.example` thành `.env`:

```bash
copy .env.example .env
```

2. Mở file `.env` và thay thế API key:

```
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Bước 4: Khởi động server

```bash
python app_server.py
```

## 💡 Cách sử dụng

### Trên Web Interface

1. Mở trình duyệt: http://127.0.0.1:8000/
2. Click vào nút **robot icon** ở góc dưới bên phải
3. Nhập câu hỏi và nhấn Enter hoặc click gửi
4. Click vào các gợi ý (suggestions) để hỏi nhanh

### Ví dụ câu hỏi

#### Thống kê cơ bản:

```
- Hôm nay có bao nhiêu vi phạm?
- Thống kê vi phạm vượt đèn đỏ
- Có bao nhiêu trường hợp không đội mũ bảo hiểm trong tuần này?
```

#### So sánh & Xu hướng:

```
- So sánh số vi phạm giữa tuần này và tuần trước
- Xu hướng vi phạm theo khung giờ
- Loại vi phạm nào phổ biến nhất?
```

#### Tra cứu chi tiết:

```
- Tìm vi phạm của xe máy biển số 29-H1 12345
- Liệt kê vi phạm lấn làn từ 7h đến 9h sáng nay
- Cho xem 5 hình ảnh vi phạm mới nhất
```

#### Trạng thái hệ thống:

```
- Camera nào đang hoạt động?
- Có camera nào offline không?
- Tình trạng hệ thống hiện tại
```

#### Tra cứu quy định:

```
- Vượt đèn đỏ phạt bao nhiêu?
- Không đội mũ bảo hiểm phạt theo Nghị định nào?
- Quy trình xử lý vi phạm qua hình ảnh
```

## 🔧 API Endpoints

### 1. Chat với AI

```http
POST /api/chat
Content-Type: application/json

{
  "message": "Hôm nay có bao nhiêu vi phạm?"
}
```

**Response:**

```json
{
  "success": true,
  "response": "Theo dữ liệu hôm nay, hệ thống ghi nhận 45 vi phạm...",
  "timestamp": "2024-11-07T10:30:00"
}
```

### 2. Lấy gợi ý câu hỏi

```http
GET /api/chat/suggestions
```

**Response:**

```json
{
  "success": true,
  "suggestions": [
    "Hôm nay có bao nhiêu vi phạm?",
    "Thống kê vi phạm vượt đèn đỏ",
    "..."
  ]
}
```

### 3. Thống kê nhanh

```http
GET /api/chat/quick-stats
```

**Response:**

```json
{
  "success": true,
  "stats": {
    "total_today": 45,
    "total_all_time": 1234,
    "cameras_active": 3
  }
}
```

### 4. Kiểm tra trạng thái chatbot

```http
GET /api/chat/status
```

**Response:**

```json
{
  "available": true,
  "ready": true,
  "message": "Chatbot sẵn sàng"
}
```

## 📁 Cấu trúc File

```
python_project/
├── gemini_chatbot.py          # Service xử lý Gemini AI
├── app_server.py              # Flask server với API endpoints
├── templates/
│   └── index.html            # UI chatbot đã tích hợp
├── .env.example              # Template cấu hình
├── .env                      # Cấu hình thực (không commit)
├── requirements.txt          # Dependencies đã cập nhật
└── CHATBOT_SETUP.md         # File này
```

## 🎨 Giao diện Chatbot

- **Nút mở chatbot**: Icon robot màu tím ở góc dưới phải
- **Cửa sổ chat**: 400x600px, responsive trên mobile
- **Avatar**: Icon người dùng (xanh) và robot (tím)
- **Typing indicator**: Hiệu ứng 3 chấm động khi AI đang suy nghĩ
- **Suggestions**: 4 câu hỏi gợi ý ngay dưới khung chat
- **Theme**: Gradient tím hiện đại, phù hợp với giao diện hệ thống

## ⚙️ Cấu hình Nâng cao

### Thay đổi model Gemini

Trong file `gemini_chatbot.py`:

```python
# Mặc định sử dụng gemini-pro
self.model = genai.GenerativeModel('gemini-pro')

# Có thể thay đổi thành:
# self.model = genai.GenerativeModel('gemini-1.5-pro')
```

### Tùy chỉnh System Prompt

Chỉnh sửa hàm `build_system_prompt()` trong `gemini_chatbot.py` để thay đổi cách AI trả lời.

### Thêm quy định mới

Thêm vào hàm `get_traffic_regulations()`:

```python
"ten_loi_vi_pham": {
    "ten": "Tên lỗi vi phạm",
    "muc_phat": "Mức phạt",
    "tuoc_bang_lai": "Có/Không",
    "phương_tiện": "Loại xe",
    "nghi_dinh": "Nghị định"
}
```

## 🐛 Xử lý lỗi thường gặp

### Lỗi: "GEMINI_API_KEY not found"

**Nguyên nhân**: Chưa cấu hình API key

**Giải pháp**:

1. Tạo file `.env` từ `.env.example`
2. Thêm API key vào file `.env`
3. Khởi động lại server

### Lỗi: "Chatbot không khả dụng"

**Nguyên nhân**: Thiếu thư viện hoặc lỗi import

**Giải pháp**:

```bash
pip install google-generativeai python-dotenv
```

### Lỗi: "Cannot get statistics"

**Nguyên nhân**: Database chưa có dữ liệu hoặc chưa kết nối

**Giải pháp**:

- Kiểm tra kết nối MySQL
- Đảm bảo đã xử lý ít nhất 1 video để có dữ liệu
- AI vẫn hoạt động nhưng không có dữ liệu thống kê

## 📊 Giới hạn & Lưu ý

### Giới hạn API miễn phí

- **Gemini API Free Tier**: 60 requests/phút
- **Rate limit**: Nếu vượt quá sẽ bị tạm khóa vài giây

### Bảo mật

- ⚠️ **KHÔNG** commit file `.env` lên Git
- ⚠️ **KHÔNG** chia sẻ API key
- ✅ Sử dụng `.gitignore` để loại trừ `.env`

### Performance

- Response time: 2-5 giây (tùy độ phức tạp câu hỏi)
- Context window: ~8000 tokens
- Lịch sử chat được lưu trong session

## 🔄 Nâng cấp trong tương lai

### Đang phát triển:

- [ ] RAG (Retrieval-Augmented Generation) với vector database
- [ ] Xuất báo cáo Excel/PDF trực tiếp từ chat
- [ ] Gửi email báo cáo tự động
- [ ] Voice input/output (speech-to-text)
- [ ] Multi-language support (English/Vietnamese)
- [ ] Admin dashboard để theo dõi sử dụng chatbot

### Có thể mở rộng:

- Tích hợp với Telegram/Zalo bot
- Mobile app với chatbot
- Webhook để gửi thông báo real-time
- Analytics dashboard cho chatbot usage

## 📞 Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra console log trong browser (F12)
2. Kiểm tra terminal output của Flask server
3. Xem file log trong `logs/` (nếu có)
4. Đọc phần Xử lý lỗi ở trên

## 📚 Tài liệu tham khảo

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Python dotenv](https://github.com/theskumar/python-dotenv)

---

**Phát triển bởi**: Đội ngũ Traffic Monitoring System  
**Phiên bản**: 1.0.0  
**Ngày cập nhật**: November 2024
