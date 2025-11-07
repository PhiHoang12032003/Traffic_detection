from db_config import DatabaseConfig
from gemini_chatbot import GeminiChatbot

# Kết nối database
db = DatabaseConfig(password='12345678')
db.connect()

# Khởi tạo chatbot
bot = GeminiChatbot(db)

# Lấy context với dữ liệu thật
ctx = bot.get_system_context()

print("=" * 60)
print("📊 THỐNG KÊ VI PHẠM TỪ DATABASE")
print("=" * 60)
print(f"\n✅ Tổng số vi phạm: {ctx['statistics']['total_violations']}")
print(f"   - Vượt đèn đỏ: {ctx['statistics']['red_light_violations']} trường hợp")
print(f"   - Không đội mũ: {ctx['statistics']['helmet_violations']} trường hợp")
print(f"   - Lấn làn: {ctx['statistics']['lane_violations']} trường hợp")

print("\n" + "=" * 60)
print("🚦 CHI TIẾT VI PHẠM VƯỢT ĐÈN ĐỎ")
print("=" * 60)
for i, v in enumerate(ctx['violation_details']['red_light'], 1):
    print(f"\n{i}. Biển số: {v['license_plate']}")
    print(f"   Thời gian: {v['violation_time']}")
    print(f"   Camera: {v['camera_id']}")
    print(f"   Độ tin cậy: {v['confidence']}")

print("\n✅ Dữ liệu đã được lấy từ database thành công!")
