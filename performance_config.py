"""
Performance Configuration for Traffic Detection System
Cấu hình hiệu suất cho hệ thống phát hiện giao thông

Điều chỉnh các tham số này để tối ưu hóa hiệu suất phù hợp với phần cứng của bạn:
- LOW: Phần cứng yếu (CPU i3, 4GB RAM)
- MEDIUM: Phần cứng trung bình (CPU i5, 8GB RAM) 
- HIGH: Phần cứng mạnh (CPU i7+, 16GB+ RAM, GPU)
"""

class PerformanceConfig:
    def __init__(self, mode="MEDIUM"):
        """
        mode: "LOW", "MEDIUM", "HIGH"
        """
        self.mode = mode.upper()
        self._setup_config()
    
    def _setup_config(self):
        if self.mode == "LOW":
            # Cấu hình cho phần cứng yếu - Ưu tiên tốc độ (SMOOTH STREAMING)
            self.video_resize_factor = 0.6  # Tăng resolution một chút để đẹp hơn
            self.frame_skip = 2  # Giảm skip để mượt mà hơn
            self.yolo_conf_threshold = 0.7  # Confidence cao để giảm false positives
            self.yolo_img_size = 320  # Kích thước ảnh nhỏ cho YOLO
            self.ocr_frequency = 12  # OCR ít hơn để tăng tốc
            self.batch_size = 1  # Batch size nhỏ
            self.queue_maxsize = 10  # Queue nhỏ để tiết kiệm RAM
            self.save_evidence_ratio = 10  # Chỉ lưu 1/10 violations
            self.enable_license_detection = False  # Tắt OCR để tăng tốc
            self.thread_count = 2  # Ít threads
            
        elif self.mode == "MEDIUM":
            # Cấu hình cân bằng - Mặc định (OPTIMIZED cho smooth streaming)
            self.video_resize_factor = 0.75  # Giảm resolution ít hơn
            self.frame_skip = 1  # Chỉ skip 1 frame để mượt mà hơn
            self.yolo_conf_threshold = 0.6
            self.yolo_img_size = 480  # Giảm size YOLO để tăng tốc
            self.ocr_frequency = 8  # OCR ít thường xuyên hơn
            self.batch_size = 4
            self.queue_maxsize = 32
            self.save_evidence_ratio = 5  # Lưu 1/5 violations
            self.enable_license_detection = True
            self.thread_count = 3
            
        elif self.mode == "HIGH":
            # Cấu hình cho phần cứng mạnh - Ưu tiên chất lượng
            self.video_resize_factor = 0.9  # Giữ gần như full resolution
            self.frame_skip = 1  # Xử lý hầu hết frames
            self.yolo_conf_threshold = 0.5
            self.yolo_img_size = 1024
            self.ocr_frequency = 3  # OCR thường xuyên hơn
            self.batch_size = 8
            self.queue_maxsize = 64
            self.save_evidence_ratio = 3  # Lưu nhiều violations
            self.enable_license_detection = True
            self.thread_count = 4
        
        # Cài đặt chung
        self.display_info_frequency = max(30, 30 * self.frame_skip)  # Hiển thị thông tin ít hơn
        self.cleanup_frequency = 1000  # Cleanup memory mỗi 1000 frames
        
    def get_video_dimensions(self, original_width, original_height):
        """Tính toán kích thước video sau khi resize"""
        new_width = int(original_width * self.video_resize_factor)
        new_height = int(original_height * self.video_resize_factor)
        # Đảm bảo kích thước chia hết cho 2 (requirement cho video encoding)
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        return new_width, new_height
    
    def should_process_frame(self, frame_number):
        """Kiểm tra có nên xử lý frame này không"""
        return frame_number % (self.frame_skip + 1) == 0
    
    def should_run_ocr(self, frame_number):
        """Kiểm tra có nên chạy OCR cho frame này không"""
        return self.enable_license_detection and (frame_number % self.ocr_frequency == 0)
    
    def should_save_evidence(self, violation_count):
        """Kiểm tra có nên lưu evidence cho violation này không"""
        return violation_count % self.save_evidence_ratio == 1
    
    def get_info_text(self):
        """Lấy thông tin cấu hình hiện tại"""
        return f"Performance: {self.mode} | Resize: {self.video_resize_factor} | Skip: {self.frame_skip} | OCR: {'ON' if self.enable_license_detection else 'OFF'}"

# Tạo các instance có sẵn
PERFORMANCE_LOW = PerformanceConfig("LOW")
PERFORMANCE_MEDIUM = PerformanceConfig("MEDIUM") 
PERFORMANCE_HIGH = PerformanceConfig("HIGH")

# Auto-detect performance level based on system (optional)
def auto_detect_performance():
    """Tự động phát hiện mức hiệu suất phù hợp dựa trên hệ thống"""
    try:
        import psutil
        import platform
        
        # Kiểm tra RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Kiểm tra CPU cores
        cpu_count = psutil.cpu_count()
        
        # Phân loại đơn giản
        if ram_gb >= 12 and cpu_count >= 6:
            return PERFORMANCE_HIGH
        elif ram_gb >= 6 and cpu_count >= 4:
            return PERFORMANCE_MEDIUM
        else:
            return PERFORMANCE_LOW
            
    except ImportError:
        print("psutil not installed, using MEDIUM performance")
        return PERFORMANCE_MEDIUM
    except Exception:
        print("Auto-detection failed, using MEDIUM performance")
        return PERFORMANCE_MEDIUM

if __name__ == "__main__":
    # Test configurations
    for mode in ["LOW", "MEDIUM", "HIGH"]:
        config = PerformanceConfig(mode)
        print(f"\n{mode} Configuration:")
        print(f"  Video resize: {config.video_resize_factor}")
        print(f"  Frame skip: {config.frame_skip}")
        print(f"  YOLO confidence: {config.yolo_conf_threshold}")
        print(f"  OCR enabled: {config.enable_license_detection}")
        print(f"  Info: {config.get_info_text()}")