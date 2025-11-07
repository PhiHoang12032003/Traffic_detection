"""
Gemini AI Chatbot Service for Traffic Monitoring System
Hỗ trợ cán bộ công an tra cứu thông tin, thống kê vi phạm
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
from db_config import DatabaseConfig, StatisticsDatabase

load_dotenv()

class GeminiChatbot:
    """Chatbot AI sử dụng Google Gemini để hỗ trợ cán bộ công an"""
    
    def __init__(self, db_connection=None):
        """
        Khởi tạo Gemini AI Chatbot
        
        Args:
            db_connection: DatabaseConfig instance
        """
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        # Sử dụng gemini-2.5-flash (model mới nhất, ổn định)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.db = db_connection
        self.stats_db = StatisticsDatabase(db_connection) if db_connection else None
        
        # Chat history để context
        self.chat_history = []
        
    def get_traffic_regulations(self):
        """Quy định xử phạt giao thông Việt Nam"""
        return {
            "vuot_den_do": {
                "ten": "Vi phạm vượt đèn đỏ",
                "muc_phat": "4,000,000 - 6,000,000 VNĐ",
                "tuoc_bang_lai": "2-4 tháng",
                "phương_tiện": "Ô tô, xe máy",
                "nghi_dinh": "Nghị định 100/2019/NĐ-CP"
            },
            "khong_doi_mu": {
                "ten": "Không đội mũ bảo hiểm",
                "muc_phat": "200,000 - 300,000 VNĐ",
                "tuoc_bang_lai": "Không",
                "phương_tiện": "Xe máy, xe đạp điện",
                "nghi_dinh": "Nghị định 100/2019/NĐ-CP"
            },
            "lan_lan": {
                "ten": "Lấn làn, đi sai làn đường",
                "muc_phat": "800,000 - 1,000,000 VNĐ (xe máy), 1,000,000 - 2,000,000 VNĐ (ô tô)",
                "tuoc_bang_lai": "Không (thường)",
                "phương_tiện": "Ô tô, xe máy",
                "nghi_dinh": "Nghị định 100/2019/NĐ-CP"
            },
            "vuot_toc_do": {
                "ten": "Vượt quá tốc độ quy định",
                "muc_phat": "400,000 - 8,000,000 VNĐ (tùy mức độ)",
                "tuoc_bang_lai": "2-4 tháng (nếu vượt > 35 km/h)",
                "phương_tiện": "Ô tô, xe máy",
                "nghi_dinh": "Nghị định 100/2019/NĐ-CP"
            }
        }
    
    def get_system_context(self):
        """Lấy context từ database để cung cấp cho AI"""
        try:
            context = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_status": "online",
                "statistics": {
                    "lane_violations": 0,
                    "helmet_violations": 0,
                    "red_light_violations": 0,
                    "total_violations": 0
                },
                "violation_details": {
                    "lane": [],
                    "helmet": [],
                    "red_light": []
                },
                "cameras": [
                    {
                        "camera_id": 1, 
                        "name": "Lane Detection Camera", 
                        "type": "lane",
                        "location": "Đường Nguyễn Trãi - Thanh Xuân, Hà Nội",
                        "description": "Phát hiện lấn làn đường"
                    },
                    {
                        "camera_id": 2, 
                        "name": "Helmet Detection Camera", 
                        "type": "helmet",
                        "location": "Giao lộ Láng - Cầu Giấy, Đống Đa, Hà Nội",
                        "description": "Phát hiện không đội mũ bảo hiểm"
                    },
                    {
                        "camera_id": 3, 
                        "name": "Red Light Detection Camera", 
                        "type": "red_light",
                        "location": "Ngã tư Trần Duy Hưng - Hoàng Đạo Thúy, Cầu Giấy, Hà Nội",
                        "description": "Phát hiện vượt đèn đỏ"
                    }
                ],
                "regulations": self.get_traffic_regulations()
            }
            
            # Lấy thống kê và chi tiết nếu có database
            if self.stats_db:
                try:
                    # Lấy chi tiết vi phạm lấn làn
                    lane_details = self.get_lane_violation_details()
                    if lane_details:
                        context["violation_details"]["lane"] = lane_details
                        context["statistics"]["lane_violations"] = len(lane_details)
                    
                    # Lấy chi tiết vi phạm mũ bảo hiểm
                    helmet_details = self.get_helmet_violation_details()
                    if helmet_details:
                        context["violation_details"]["helmet"] = helmet_details
                        context["statistics"]["helmet_violations"] = len(helmet_details)
                    
                    # Lấy chi tiết vi phạm vượt đèn đỏ
                    red_light_details = self.get_red_light_violation_details()
                    if red_light_details:
                        context["violation_details"]["red_light"] = red_light_details
                        context["statistics"]["red_light_violations"] = len(red_light_details)
                    
                    context["statistics"]["total_violations"] = (
                        context["statistics"]["lane_violations"] +
                        context["statistics"]["helmet_violations"] +
                        context["statistics"]["red_light_violations"]
                    )
                except Exception as e:
                    print(f"⚠️ Cannot get statistics: {e}")
            
            return context
        except Exception as e:
            print(f"❌ Error getting context: {e}")
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_status": "limited",
                "regulations": self.get_traffic_regulations()
            }
    
    def get_lane_violation_details(self):
        """Lấy chi tiết vi phạm lấn làn với thông tin xe và thời gian"""
        try:
            query = """
                SELECT 
                    violation_id,
                    vehicle_type,
                    violation_type,
                    DATE_FORMAT(detected_at, '%Y-%m-%d %H:%i:%s') as violation_time,
                    camera_id,
                    frame_number,
                    time_in_video,
                    confidence
                FROM lane_violations
                ORDER BY detected_at DESC
                LIMIT 50
            """
            results = self.db.execute_query(query, fetch=True)
            return results if results else []
        except Exception as e:
            print(f"⚠️ Error getting lane details: {e}")
            return []
    
    def get_helmet_violation_details(self):
        """Lấy chi tiết vi phạm mũ bảo hiểm với biển số xe và thời gian"""
        try:
            query = """
                SELECT 
                    violation_id,
                    license_plate,
                    has_helmet,
                    DATE_FORMAT(detected_at, '%Y-%m-%d %H:%i:%s') as violation_time,
                    camera_id,
                    frame_number,
                    time_in_video,
                    confidence
                FROM helmet_violations
                WHERE has_helmet = 0
                ORDER BY detected_at DESC
                LIMIT 50
            """
            results = self.db.execute_query(query, fetch=True)
            return results if results else []
        except Exception as e:
            print(f"⚠️ Error getting helmet details: {e}")
            return []
    
    def get_red_light_violation_details(self):
        """Lấy chi tiết vi phạm vượt đèn đỏ với biển số xe và thời gian"""
        try:
            query = """
                SELECT 
                    violation_id,
                    license_plate,
                    DATE_FORMAT(detected_at, '%Y-%m-%d %H:%i:%s') as violation_time,
                    camera_id,
                    frame_number,
                    time_in_video,
                    confidence
                FROM red_light_violations
                ORDER BY detected_at DESC
                LIMIT 50
            """
            results = self.db.execute_query(query, fetch=True)
            return results if results else []
        except Exception as e:
            print(f"⚠️ Error getting red light details: {e}")
            return []
    
    def convert_decimals(self, obj):
        """Convert Decimal objects to float for JSON serialization"""
        from decimal import Decimal
        if isinstance(obj, list):
            return [self.convert_decimals(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_decimals(value) for key, value in obj.items()}
        elif isinstance(obj, Decimal):
            return float(obj)
        else:
            return obj
    
    def build_system_prompt(self, context):
        """Xây dựng system prompt với context"""
        # Convert Decimal to float before JSON serialization
        context = self.convert_decimals(context)
        
        prompt = f"""
Bạn là trợ lý AI thông minh hỗ trợ cán bộ CSGT quản lý hệ thống giám sát giao thông.

NHIỆM VỤ CỦA BẠN:
1. Trả lời câu hỏi về thống kê vi phạm giao thông
2. Tra cứu thông tin camera và trạng thái hệ thống
3. Cung cấp thông tin về quy định và mức phạt
4. Phân tích xu hướng và so sánh dữ liệu
5. Hỗ trợ tạo báo cáo chi tiết

FORMAT TRẢ LỜI YÊU CẦU:
1. **Lời chào**: Bắt đầu bằng "Chào cán bộ," hoặc "Kính báo cáo,"
2. **Nội dung chi tiết**:
   - Số liệu cụ thể với đơn vị rõ ràng
   - Địa điểm: Luôn nêu rõ tên đường/ngã tư/quận (ví dụ: "Ngã tư Láng - Cầu Giấy", "Đường Nguyễn Trãi, Quận Thanh Xuân")
   - Thời gian: Nêu rõ khung giờ nếu có
   - Phân tích xu hướng nếu phù hợp
3. **Đề xuất/Kiến nghị**: Kết thúc với đề xuất hành động cụ thể

PHONG CÁCH VIẾT:
- Trang trọng, chuyên nghiệp (văn phong công an)
- Chi tiết, đầy đủ thông tin
- Sử dụng các thuật ngữ chuyên ngành: "trường hợp vi phạm", "giao lộ trọng điểm", "giám sát chặt chẽ"
- Format Markdown: dùng **in đậm**, dấu đầu dòng (*), số thứ tự
- Độ dài: 5-10 dòng, không quá ngắn gọn

VÍ DỤ TRẢ LỜI MẪU (với dữ liệu thực từ database):
```
Chào cán bộ,

Báo cáo thống kê vi phạm vượt đèn đỏ:

* **Tổng số vi phạm**: 2 trường hợp
* **Địa điểm**: Ngã tư Trần Duy Hưng - Hoàng Đạo Thúy, Cầu Giấy, Hà Nội

**Chi tiết vi phạm:**

1. **Biển số**: 29A-12345 (Xe máy)
   - Thời gian: 2024-11-07 08:15:23
   - Địa điểm: Ngã tư Trần Duy Hưng - Hoàng Đạo Thúy
   - Mức phạt: 4,000,000 - 6,000,000 VNĐ + tước bằng lái 2-4 tháng

2. **Biển số**: 30G-67890 (Ô tô)
   - Thời gian: 2024-11-07 17:45:10
   - Địa điểm: Ngã tư Trần Duy Hưng - Hoàng Đạo Thúy
   - Mức phạt: 4,000,000 - 6,000,000 VNĐ + tước bằng lái 2-4 tháng

* **Phân tích**: Vi phạm xảy ra cả giờ sáng và chiều, cần tăng cường giám sát

**Đề xuất**: 
- Gửi thông báo phạt nguội cho 2 phương tiện vi phạm
- Tăng cường biển báo đèn tín hiệu tại ngã tư
- Giám sát chặt chẽ trong khung giờ cao điểm (7h-9h, 17h-19h)
```

DỮ LIỆU HỆ THỐNG HIỆN TẠI:
{json.dumps(context, ensure_ascii=False, indent=2)}

DANH SÁCH CAMERA VÀ ĐỊA ĐIỂM:
- Camera 1 (Lane Detection): Đường Nguyễn Trãi - Thanh Xuân, Hà Nội
- Camera 2 (Helmet Detection): Giao lộ Láng - Cầu Giấy, Đống Đa, Hà Nội  
- Camera 3 (Red Light Detection): Ngã tư Trần Duy Hưng - Hoàng Đạo Thúy, Cầu Giấy, Hà Nội

LƯU Ý QUAN TRỌNG:
- **BẮT BUỘC**: Khi có dữ liệu trong violation_details, PHẢI liệt kê chi tiết TỪNG vi phạm bao gồm:
  + Biển số xe (license_plate)
  + Thời gian vi phạm chính xác (violation_time) - định dạng "YYYY-MM-DD HH:MM:SS"
  + Địa điểm cụ thể theo camera
  + Loại phương tiện (nếu có)
- Nếu không có dữ liệu trong violation_details, thông báo rõ ràng "Hiện chưa có dữ liệu vi phạm được ghi nhận"
- Luôn nêu rõ địa điểm cụ thể (tên đường, quận) theo thông tin camera
- Khi được hỏi về quy định, trích dẫn đầy đủ Nghị định và điều khoản
- Đối với thống kê, luôn phân tích và đưa ra nhận xét về xu hướng
- Kết thúc bằng đề xuất hành động cụ thể, khả thi

CÁCH SỬ DỤNG DỮ LIỆU VIOLATION_DETAILS:
- violation_details.red_light: Chứa danh sách vi phạm vượt đèn đỏ với license_plate, violation_time, vehicle_type
- violation_details.helmet: Chứa danh sách vi phạm không đội mũ với license_plate, violation_time
- violation_details.lane: Chứa danh sách vi phạm lấn làn với license_plate, violation_time, violation_type
- LUÔN hiển thị biển số xe và thời gian cho MỖI vi phạm
"""
        return prompt
    
    def chat(self, user_message):
        """
        Xử lý tin nhắn từ user
        
        Args:
            user_message: Câu hỏi từ user
            
        Returns:
            dict: {'success': bool, 'response': str, 'error': str}
        """
        try:
            # Lấy context hệ thống
            context = self.get_system_context()
            
            # Build prompt với context
            system_prompt = self.build_system_prompt(context)
            
            # Tạo full prompt
            full_prompt = f"{system_prompt}\n\nCÂU HỎI CỦA CÁN BỘ: {user_message}\n\nTRẢ LỜI:"
            
            # Gọi Gemini API
            response = self.model.generate_content(full_prompt)
            
            if not response or not response.text:
                return {
                    'success': False,
                    'response': '',
                    'error': 'Không nhận được phản hồi từ AI'
                }
            
            answer = response.text.strip()
            
            # Lưu vào history
            self.chat_history.append({
                'timestamp': datetime.now().isoformat(),
                'user': user_message,
                'assistant': answer
            })
            
            return {
                'success': True,
                'response': answer,
                'error': None
            }
            
        except Exception as e:
            error_msg = f"Lỗi xử lý: {str(e)}"
            print(f"❌ Chatbot error: {error_msg}")
            return {
                'success': False,
                'response': '',
                'error': error_msg
            }
    
    def get_quick_stats(self):
        """Lấy thống kê nhanh để trả lời câu hỏi cơ bản"""
        try:
            if not self.stats_db:
                return {
                    'total_today': 0,
                    'total_all_time': 0,
                    'cameras_active': 3
                }
            
            # Lấy stats từng loại
            lane_stats = self.stats_db.get_lane_stats(camera_id=1)
            helmet_stats = self.stats_db.get_helmet_stats(camera_id=2)
            red_light_stats = self.stats_db.get_red_light_stats(camera_id=3)
            
            total = 0
            if lane_stats:
                total += len(lane_stats)
            if helmet_stats:
                total += len(helmet_stats)
            if red_light_stats:
                total += len(red_light_stats)
            
            return {
                'total_today': total,
                'total_all_time': total,
                'cameras_active': 3
            }
        except Exception as e:
            print(f"❌ Error getting quick stats: {e}")
            return {
                'total_today': 0,
                'total_all_time': 0,
                'cameras_active': 3
            }
    
    def suggest_questions(self):
        """Đề xuất các câu hỏi mẫu cho user"""
        return [
            "Hôm nay có bao nhiêu vi phạm?",
            "Thống kê vi phạm vượt đèn đỏ",
            "Camera nào đang hoạt động?",
            "Vượt đèn đỏ phạt bao nhiêu?",
            "Không đội mũ bảo hiểm phạt thế nào?",
            "So sánh số vi phạm hôm nay với hôm qua",
            "Loại vi phạm nào nhiều nhất?",
            "Tình trạng hệ thống hiện tại"
        ]
    
    def clear_history(self):
        """Xóa lịch sử chat"""
        self.chat_history = []
        print("🗑️ Chat history cleared")


# Hàm tiện ích để test
def test_chatbot():
    """Test chatbot functionality"""
    print("🤖 Testing Gemini Chatbot...")
    
    try:
        # Khởi tạo database
        db_config = DatabaseConfig(password='')
        if db_config.connect():
            chatbot = GeminiChatbot(db_config)
        else:
            print("⚠️ Testing without database connection")
            chatbot = GeminiChatbot(None)
        
        # Test questions
        test_questions = [
            "Hôm nay có bao nhiêu vi phạm?",
            "Vượt đèn đỏ phạt bao nhiêu?",
            "Giải thích quy định đội mũ bảo hiểm"
        ]
        
        for q in test_questions:
            print(f"\n❓ Q: {q}")
            result = chatbot.chat(q)
            if result['success']:
                print(f"✅ A: {result['response'][:200]}...")
            else:
                print(f"❌ Error: {result['error']}")
        
        print("\n✅ Chatbot test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_chatbot()
