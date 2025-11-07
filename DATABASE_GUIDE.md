# Hướng Dẫn Sử Dụng Database

## 🚀 Cài Đặt Nhanh

### 1. Tạo Database

```powershell
# PowerShell
Get-Content database_setup.sql | mysql -u root -p
```

### 2. Chạy Server

```powershell
python app_server.py
```

Khi được hỏi password MySQL:

- Nhập password → Hệ thống dùng database
- Enter (bỏ qua) → Hệ thống chạy bình thường không dùng database

## 📊 Cấu Trúc Database

### 3 Cameras:

- **Camera 1**: Lane Detection (Làn đường)
- **Camera 2**: Helmet Detection (Mũ bảo hiểm)
- **Camera 3**: Red Light Detection (Đèn đỏ)

### Tables:

- `cameras` - Thông tin camera
- `videos` - Video đã upload
- `lane_violations` - Vi phạm Camera 1
- `helmet_violations` - Vi phạm Camera 2
- `red_light_violations` - Vi phạm Camera 3

## 🔧 API Mới

### Upload Video

```javascript
POST /api/upload_video_db
Form-data:
  - video: [file]
  - camera_id: 1, 2, or 3
```

### Thống Kê

```javascript
GET / api / stats / overall; // Tất cả camera
GET / api / stats / 1; // Camera 1 (Lane)
GET / api / stats / 2; // Camera 2 (Helmet)
GET / api / stats / 3; // Camera 3 (Red Light)
```

## 💾 Truy Vấn Database

```sql
-- Xem tất cả vi phạm
SELECT * FROM v_overall_stats;

-- Vi phạm làn đường
SELECT * FROM lane_violations ORDER BY detected_at DESC LIMIT 10;

-- Vi phạm mũ bảo hiểm (không mũ)
SELECT * FROM helmet_violations WHERE has_helmet = FALSE LIMIT 10;

-- Vi phạm đèn đỏ
SELECT * FROM red_light_violations ORDER BY detected_at DESC LIMIT 10;
```

## ⚠️ Lưu Ý

- Hệ thống **VẪN CHẠY BÌ NÒI** nếu không có database
- Database chỉ để lưu trữ và thống kê tốt hơn
- Tất cả chức năng cũ vẫn hoạt động
- Trang chủ tự động hiển thị số liệu từ database nếu có

## 🔍 Kiểm Tra

Mở trình duyệt: http://localhost:8000

- Thống kê trang chủ tự động load từ database
- API: http://localhost:8000/api/stats/overall
