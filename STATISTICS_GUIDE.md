# 📊 Hướng Dẫn Sử Dụng Trang Thống Kê

## Tổng Quan

Trang thống kê cung cấp giao diện trực quan để xem và phân tích dữ liệu vi phạm giao thông từ hệ thống giám sát. Tất cả dữ liệu được lấy trực tiếp từ database MySQL.

## Các Tính Năng Chính

### 1. 📈 Thống Kê Tổng Quan

**7 Card Thống Kê Chính:**

- **Tổng Vi Phạm**: Tổng số vi phạm từ trước đến nay
- **Vi Phạm Hôm Nay**: Số vi phạm được ghi nhận trong ngày
- **Vi Phạm Làn Đường**: Tổng vi phạm từ Camera 1
- **Không Đội Mũ Bảo Hiểm**: Tổng vi phạm từ Camera 2
- **Vi Phạm Đèn Đỏ**: Tổng vi phạm từ Camera 3
- **Tỷ Lệ Tuân Thủ**: Phần trăm tuân thủ luật giao thông
- **Trung Bình Mỗi Ngày**: Số vi phạm trung bình trong 7 ngày qua

### 2. 📊 Các Biểu Đồ

#### a) Biểu Đồ Xu Hướng Vi Phạm (Line Chart)

- Hiển thị xu hướng tổng số vi phạm theo thời gian
- Dữ liệu: 7 ngày qua
- Cập nhật: Real-time từ database

#### b) Phân Bố Loại Vi Phạm (Pie Chart)

- Hiển thị tỷ lệ phần trăm từng loại vi phạm
- 3 loại: Làn Đường, Mũ Bảo Hiểm, Đèn Đỏ
- Có tooltip hiển thị số lượng và phần trăm

#### c) So Sánh Vi Phạm Theo Camera (Bar Chart)

- So sánh số vi phạm giữa 3 camera
- 2 dataset: Hôm Nay vs Tổng Cộng
- Dễ dàng nhận biết camera nào vi phạm nhiều nhất

#### d) Vi Phạm Theo Giờ Trong Ngày (Line Chart)

- Hiển thị phân bố vi phạm theo khung giờ
- Chia thành 8 khung giờ (mỗi 3 tiếng)
- Giúp xác định giờ cao điểm vi phạm

#### e) Xu Hướng Vi Phạm 7 Ngày Qua (Multi-Line Chart)

- 3 đường line riêng biệt cho từng loại vi phạm
- Dễ dàng so sánh xu hướng giữa các loại
- Hiển thị ngày tháng cụ thể

### 3. 🔍 Bộ Lọc Thời Gian

**5 Tùy Chọn:**

- **Hôm Nay**: Chỉ hiển thị dữ liệu trong ngày
- **Tuần Này**: 7 ngày gần nhất
- **Tháng Này**: 30 ngày gần nhất
- **Năm Nay**: Dữ liệu trong năm hiện tại
- **Tất Cả**: Toàn bộ dữ liệu từ trước đến nay

**Cách sử dụng:**

1. Click vào nút thời gian mong muốn
2. Tất cả biểu đồ và số liệu sẽ tự động cập nhật
3. Nút đang chọn sẽ có màu cam

### 4. 📥 Xuất Báo Cáo

**3 Định Dạng:**

#### PDF

- File báo cáo định dạng chuyên nghiệp
- Bao gồm bảng thống kê đầy đủ
- Tiêu đề, ngày giờ xuất, tổng kết
- Thư viện: `reportlab`

#### Excel

- File .xlsx có thể chỉnh sửa
- Bảng dữ liệu có định dạng màu sắc
- Dễ dàng phân tích thêm trong Excel
- Thư viện: `openpyxl`

#### CSV

- File văn bản thuần túy
- Dễ dàng import vào các hệ thống khác
- Không cần thư viện bổ sung

**Cách xuất:**

1. Click vào nút xuất tương ứng
2. File sẽ tự động download
3. Tên file có timestamp để dễ quản lý

## API Endpoints

### Lấy Thống Kê Tổng Quan

```http
GET /api/stats/overall
```

**Response:**

```json
{
  "success": true,
  "summary": {
    "total_violations_all_time": 150,
    "total_violations_today": 25
  },
  "cameras": [
    {
      "camera_id": 1,
      "camera_name": "Camera 1",
      "camera_type": "lane",
      "total_all_time": 60,
      "total_today": 10
    },
    ...
  ]
}
```

### Lấy Dữ Liệu Xu Hướng

```http
GET /api/stats/trend/<period>
```

**Parameters:**

- `period`: today, week, month, year, all

**Response:**

```json
{
  "success": true,
  "period": "week",
  "labels": ["01/11", "02/11", "03/11", ...],
  "data": [12, 15, 18, ...]
}
```

### Lấy Chi Tiết Theo Camera

```http
GET /api/stats/breakdown
```

**Response:**

```json
{
  "success": true,
  "cameras": [
    {
      "camera_id": 1,
      "camera_name": "Camera 1 (Làn Đường)",
      "total_all_time": 60,
      "total_today": 10
    },
    ...
  ]
}
```

### Lấy Xu Hướng 7 Ngày

```http
GET /api/stats/week-trend
```

**Response:**

```json
{
  "success": true,
  "labels": ["01/11", "02/11", ...],
  "datasets": {
    "lane": [5, 8, 6, ...],
    "helmet": [3, 5, 4, ...],
    "redlight": [2, 3, 3, ...]
  }
}
```

### Xuất Báo Cáo

```http
GET /api/export/csv
GET /api/export/excel
GET /api/export/pdf
```

## Yêu Cầu Cài Đặt

### Thư Viện Python Cần Thiết

```bash
# Cơ bản (đã có)
pip install flask mysql-connector-python

# Cho xuất Excel
pip install openpyxl

# Cho xuất PDF
pip install reportlab
```

### Cấu Hình Database

Đảm bảo MySQL đang chạy và có:

- Database: `traffic_monitoring`
- Tables: `lane_violations`, `helmet_violations`, `red_light_violations`
- View: `v_overall_stats`

## Tính Năng Kỹ Thuật

### 1. Real-time Updates

- Dữ liệu load từ database mỗi khi mở trang
- Không cache, luôn là dữ liệu mới nhất

### 2. Chart.js Integration

- Sử dụng Chart.js v3 cho biểu đồ
- Responsive, đẹp mắt, interactive
- Tooltips chi tiết khi hover

### 3. Smooth Animations

- Số liệu có animation đếm lên
- Transitions mượt mà giữa các filter
- Loading states rõ ràng

### 4. Error Handling

- Fallback data nếu database lỗi
- Error messages user-friendly
- Console logs chi tiết cho debug

## Tips & Tricks

### 1. Tối Ưu Hiệu Suất

- Database queries đã được optimize
- Chỉ query dữ liệu cần thiết
- Index đúng các cột

### 2. Customize Biểu Đồ

Có thể tùy chỉnh màu sắc trong code:

```javascript
backgroundColor: ["#ff6b6b", "#4facfe", "#ef4444"];
```

### 3. Thêm Loại Vi Phạm Mới

1. Thêm camera mới vào database
2. Update API endpoints
3. Update biểu đồ với màu sắc mới

## Troubleshooting

### Biểu đồ không hiển thị

- Check console log xem có lỗi API không
- Verify Chart.js đã load đúng
- Kiểm tra database có dữ liệu không

### Export không hoạt động

- PDF/Excel: Cài đặt thư viện tương ứng
- CSV: Check browser không block download
- Verify API endpoint trả về đúng

### Số liệu sai

- Verify database connection
- Check query SQL trong backend
- Xem log console để debug

## Support

Nếu gặp vấn đề:

1. Check console log (F12)
2. Xem terminal log của Flask
3. Verify database đang chạy
4. Check cấu hình trong `db_config.py`

## Future Enhancements

Có thể thêm:

- [ ] Filter theo khoảng thời gian tùy chỉnh
- [ ] Xuất báo cáo theo email
- [ ] Dashboard real-time (WebSocket)
- [ ] Notification khi vi phạm cao điểm
- [ ] Phân tích AI/ML cho dự đoán
- [ ] Mobile responsive tốt hơn
- [ ] Dark mode
- [ ] Multi-language support

---

**Version:** 1.0  
**Last Updated:** November 2024  
**Author:** Traffic Monitoring System Team
