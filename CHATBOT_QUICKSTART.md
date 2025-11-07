# 🚀 HƯỚNG DẪN NHANH - CHATBOT GEMINI AI

## ✅ ĐÃ CÀI ĐẶT

- ✅ Service `gemini_chatbot.py` - Xử lý AI
- ✅ API endpoints trong `app_server.py`
- ✅ Giao diện chatbot trong `templates/index.html`
- ✅ Thư viện `google-generativeai` và `python-dotenv`
- ✅ File cấu hình `.env` với API key
- ✅ File `.env.example` (template)

## 🎯 CÁCH SỬ DỤNG

### 1. Khởi động server

```bash
python app_server.py
```

### 2. Mở trình duyệt

```
http://127.0.0.1:8000/
```

### 3. Sử dụng chatbot

- Click icon **robot** (🤖) ở góc dưới bên phải
- Gõ câu hỏi và nhấn Enter
- Hoặc click vào các câu gợi ý

## 💬 VÍ DỤ CÂU HỎI

### Thống kê:

```
Hôm nay có bao nhiêu vi phạm?
Thống kê vi phạm vượt đèn đỏ
So sánh số vi phạm tuần này với tuần trước
```

### Trạng thái hệ thống:

```
Camera nào đang hoạt động?
Tình trạng hệ thống hiện tại
```

### Quy định pháp luật:

```
Vượt đèn đỏ phạt bao nhiêu?
Không đội mũ bảo hiểm phạt theo Nghị định nào?
```

## 🔧 CẤU TRÚC API

### POST /api/chat

Gửi tin nhắn đến chatbot

```json
{
  "message": "Hôm nay có bao nhiêu vi phạm?"
}
```

### GET /api/chat/suggestions

Lấy danh sách câu hỏi gợi ý

### GET /api/chat/status

Kiểm tra trạng thái chatbot

## 📚 TÀI LIỆU CHI TIẾT

Xem file `CHATBOT_SETUP.md` để biết:

- Hướng dẫn cài đặt chi tiết
- Danh sách tính năng đầy đủ
- Cấu hình nâng cao
- Xử lý lỗi
- API documentation

## ⚙️ KIỂM TRA HOẠT ĐỘNG

### Kiểm tra console log:

- Mở browser, nhấn F12
- Tab Console sẽ hiển thị:
  - `🤖 Initializing chatbot...`
  - `✅ Chatbot initialized`

### Kiểm tra terminal:

Server Flask sẽ hiển thị:

```
✅ Gemini AI Chatbot initialized
✅ Database initialized successfully
```

## 🐛 XỬ LÝ LỖI NHANH

### Lỗi: "Chatbot không khả dụng"

→ Kiểm tra file `.env` có GEMINI_API_KEY chưa

### Lỗi import google.generativeai

→ Chạy: `pip install google-generativeai python-dotenv`

### Chatbot không phản hồi

→ Kiểm tra console (F12) và terminal để xem lỗi

## 🎨 GIAO DIỆN

- **Nút chatbot**: Icon robot màu tím, góc dưới phải
- **Cửa sổ chat**: 400x600px, responsive
- **Giao diện**: Gradient tím, hiện đại
- **Typing indicator**: 3 chấm động khi AI đang suy nghĩ

## 📞 HỖ TRỢ

Nếu gặp vấn đề:

1. Đọc `CHATBOT_SETUP.md` (hướng dẫn đầy đủ)
2. Kiểm tra console log (F12 trong browser)
3. Kiểm tra terminal output
4. Xem phần Xử lý lỗi trong `CHATBOT_SETUP.md`

---

**Ready to use!** 🚀
Chatbot đã sẵn sàng hỗ trợ cán bộ công an.
