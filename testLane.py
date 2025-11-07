import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import createBB
import os
from datetime import datetime
import time
import pandas as pd
import json
from performance_config import PerformanceConfig, auto_detect_performance

# --- Cấu hình đường dẫn ---
# Lấy đường dẫn của thư mục chứa file script này
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Tạo một thư mục chính để chứa tất cả dữ liệu của dự án
PROJECT_DATA_DIR = os.path.join(BASE_DIR, "project_data")

# Định nghĩa tất cả các thư mục con cần thiết
DIRS = {
    'videos': os.path.join(PROJECT_DATA_DIR, "Videos"),
    'weights': os.path.join(BASE_DIR, "YolWeights"),
    'motor_violations': os.path.join(PROJECT_DATA_DIR, "data_xe_may_vi_pham"),
    'car_violations': os.path.join(PROJECT_DATA_DIR, "data_oto_vi_pham"),
    'motor_reports': os.path.join(PROJECT_DATA_DIR, "BienBanNopPhatXeMay"),
    'car_reports': os.path.join(PROJECT_DATA_DIR, "BienBanNopPhatXeOTo")
}

# Tự động tạo tất cả các thư mục nếu chúng chưa tồn tại
for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# Tạo thư mục output cho video và CSV
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Các hàm tiện ích ---

def draw_text(img, text, pos=(0, 0), font_scale=0.7, text_color=(255, 255, 255), text_color_bg=(0, 0, 0)):
    """Vẽ chữ có nền để dễ đọc hơn."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    # Vẽ hình chữ nhật làm nền
    cv2.rectangle(img, pos, (x + text_w + 5, y + text_h + 5), text_color_bg, -1)
    # Vẽ chữ lên trên
    cv2.putText(img, text, (x, y + text_h + 3), font, font_scale, text_color, thickness)

def imageMotorViolate(frame, r1, r2, c1, c2, stt):
    """Lưu ảnh xe máy vi phạm."""
    try:
        cropped_region = frame[r1:r2, c1:c2].copy()
        save_path = os.path.join(DIRS['motor_violations'], f"{stt}.jpg")
        cv2.imwrite(save_path, cropped_region)
        return save_path  # Trả về đường dẫn đầy đủ của ảnh đã lưu
    except Exception as e:
        print(f"Lỗi khi lưu ảnh xe máy vi phạm: {e}")
        return None

def imageCTBViolate(frame, r1, r2, c1, c2, stt):
    """Lưu ảnh ô tô vi phạm."""
    try:
        cropped_region = frame[r1:r2, c1:c2].copy()
        save_path = os.path.join(DIRS['car_violations'], f"{stt}.jpg")
        cv2.imwrite(save_path, cropped_region)
        return save_path  # Trả về đường dẫn đầy đủ của ảnh đã lưu
    except Exception as e:
        print(f"Lỗi khi lưu ảnh ô tô vi phạm: {e}")
        return None

# --- Hàm chính ---

if __name__ == '__main__':
    cap = None
    try:
        # --- Khởi tạo Performance Config ---
        print("🔧 Đang khởi tạo cấu hình hiệu suất...")
        
        # Tự động phát hiện hoặc cho phép user chọn
        performance_config = auto_detect_performance()
        print(f"✅ Sử dụng cấu hình: {performance_config.mode}")
        print(f"📊 {performance_config.get_info_text()}")
        
        # Cho phép user override bằng environment variable
        import os
        if 'PERFORMANCE_MODE' in os.environ:
            mode = os.environ['PERFORMANCE_MODE'].upper()
            if mode in ['LOW', 'MEDIUM', 'HIGH']:
                performance_config = PerformanceConfig(mode)
                print(f"🎯 Override: Sử dụng {mode} mode từ environment variable")
        
        # --- Khởi tạo Video ---
        video_path = os.path.join(DIRS['videos'], "main.mp4")
        weights_path = os.path.join(DIRS['weights'], 'best.pt')

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy video tại: {video_path}. Vui lòng đặt video vào thư mục {DIRS['videos']}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Không tìm thấy model weights tại: {weights_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Không thể mở file video")

        # Lấy thông tin video gốc
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tính toán kích thước mới
        new_width, new_height = performance_config.get_video_dimensions(original_width, original_height)
        
        print(f"📹 Video gốc: {original_width}x{original_height} @ {original_fps}fps, {total_frames} frames")
        print(f"📹 Video xử lý: {new_width}x{new_height} (resize factor: {performance_config.video_resize_factor})")
        print(f"⚡ Frame skip: {performance_config.frame_skip} (xử lý ~{100/(performance_config.frame_skip+1):.1f}% frames)")

        # Khởi tạo YOLO model với cấu hình tối ưu
        model = YOLO(weights_path)
        
        stt_m = 0
        stt_ctb = 0
        examBB = createBB.infoObject()
        
        # Khởi tạo video writer cho output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(OUTPUT_DIR, f"lane_violations_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (new_width, new_height))
        
        # Danh sách lưu trữ thông tin vi phạm cho CSV
        violations_data = []
        
        print(f"📼 Video output sẽ được lưu tại: {output_video_path}")
        
        # Performance tracking
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        last_fps_time = start_time
        fps_counter = 0

        print("\n" + "="*70)
        print("🚀 BẮT ĐẦU PHÂN TÍCH TỐI ƯU VÀ GHI VIDEO")
        print("="*70)
        print("🎮 HƯỚNG DẪN ĐIỀU KHIỂN:")
        print("   🔴 Nhấn 'q' - DỪNG và XUẤT KẾT QUẢ (Video + CSV)")
        print("   ⏸️  Nhấn 'p' - Tạm dừng")
        print("   👁️  Theo dõi real-time trong cửa sổ video")
        print("="*70)
        print("⏳ Đang bắt đầu phân tích...")

        print("🚀 Bắt đầu phân tích tối ưu... Nhấn 'q' để thoát và xuất kết quả, 'p' để tạm dừng.")

        # --- Vòng lặp xử lý video tối ưu ---
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("🏁 Kết thúc video.")
                break

            frame_count += 1
            
            # Skip frames theo cấu hình performance
            if not performance_config.should_process_frame(frame_count):
                continue
                
            processed_count += 1
            
            # Resize frame theo cấu hình performance
            frame = cv2.resize(frame, (new_width, new_height))
            h, w, _ = frame.shape

            # Chạy model YOLO với cấu hình tối ưu
            results = model(frame, 
                          conf=performance_config.yolo_conf_threshold,
                          imgsz=performance_config.yolo_img_size,
                          verbose=False)  # Tắt verbose để giảm I/O

            # --- Vẽ các vùng quan tâm (ROI) và làn đường ---
            roi_start = (0, int(0.2 * h))
            roi_end = (w, int(0.8 * h))
            cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2) # ROI chính

            start_line_motor = (0, int(0.2 * h))
            end_line_motor = (int(0.525 * w), int(0.8 * h))
            cv2.rectangle(frame, start_line_motor, end_line_motor, (255, 0, 255), 2) # Làn xe máy

            start_line_car = (int(0.55 * w), int(0.2 * h))
            end_line_car = (w, int(0.8 * h))
            cv2.rectangle(frame, start_line_car, end_line_car, (0, 255, 0), 2) # Làn ô tô

            # --- Xử lý các đối tượng được phát hiện ---
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Chỉ xử lý các đối tượng trong vùng ROI
                        if not (roi_start[1] < center_y < roi_end[1]):
                            continue

                        # --- Kiểm tra vi phạm trước ---
                        is_violation = False
                        
                        # Xe máy đi vào làn ô tô
                        if cls == 1 and start_line_car[0] < center_x < end_line_car[0]:
                            is_violation = True
                            stt_m += 1
                            
                            # Lưu thông tin vi phạm vào danh sách
                            violation_info = {
                                'violation_id': stt_m,
                                'type': 'xe_may_vi_pham_lan_oto',
                                'frame_number': frame_count,
                                'time_seconds': frame_count / original_fps,
                                'time_formatted': f"{int((frame_count / original_fps) // 60):02d}:{int((frame_count / original_fps) % 60):02d}",
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'center': [center_x, center_y],
                                'vehicle_class': result.names[cls],
                                'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            violations_data.append(violation_info)
                            
                            # Chỉ lưu evidence theo tỉ lệ cấu hình để giảm I/O
                            if performance_config.should_save_evidence(stt_m):
                                violation_img_path = imageMotorViolate(frame, roi_start[1], roi_end[1], roi_start[0], roi_end[0], stt_m)
                                if violation_img_path:
                                    report_path = os.path.join(DIRS['motor_reports'], f"{stt_m}.pdf")
                                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
                                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(temp_img.name)
                                        createBB.bienBanNopPhat(examBB, temp_img.name, violation_img_path, report_path)

                        # Ô tô đi vào làn xe máy
                        elif cls in [0, 3, 4] and start_line_motor[0] < center_x < end_line_motor[0]:
                            is_violation = True
                            stt_ctb += 1
                            
                            # Lưu thông tin vi phạm vào danh sách
                            violation_info = {
                                'violation_id': stt_ctb,
                                'type': 'oto_vi_pham_lan_xe_may',
                                'frame_number': frame_count,
                                'time_seconds': frame_count / original_fps,
                                'time_formatted': f"{int((frame_count / original_fps) // 60):02d}:{int((frame_count / original_fps) % 60):02d}",
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'center': [center_x, center_y],
                                'vehicle_class': result.names[cls],
                                'detected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            violations_data.append(violation_info)
                            
                            # Chỉ lưu evidence theo tỉ lệ cấu hình để giảm I/O
                            if performance_config.should_save_evidence(stt_ctb):
                                violation_img_path = imageCTBViolate(frame, roi_start[1], roi_end[1], roi_start[0], roi_end[0], stt_ctb)
                                if violation_img_path:
                                    report_path = os.path.join(DIRS['car_reports'], f"{stt_ctb}.pdf")
                                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img:
                                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(temp_img.name)
                                        createBB.bienBanNopPhat(examBB, temp_img.name, violation_img_path, report_path)

                        # --- Vẽ bounding box sau khi đã kiểm tra vi phạm ---
                        if is_violation:
                            # VẼ BOUNDING BOX MÀU ĐỎ CHO XE VI PHẠM
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f"{result.names[cls]} {conf:.2f}"
                            draw_text(frame, label, pos=(x1, y1 - 20), font_scale=0.5, text_color_bg=(0, 0, 255))
                            draw_text(frame, "VI PHAM", (x1, y1 - 40), text_color=(255, 255, 255), text_color_bg=(255, 0, 0))
                        else:
                            # VẼ BOUNDING BOX XANH LÁ CHO XE KHÔNG VI PHẠM
                            label = f"{result.names[cls]} {conf:.2f}"
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            draw_text(frame, label, pos=(x1, y1 - 20), font_scale=0.5, text_color_bg=color)

                    except Exception as e:
                        print(f"Lỗi khi xử lý đối tượng: {e}")
                        continue
            
            # Performance tracking và hiển thị thông tin
            fps_counter += 1
            current_time = time.time()
            
            # Tính FPS thực tế mỗi 30 frames
            if fps_counter % 30 == 0:
                elapsed = current_time - last_fps_time
                current_fps = 30 / elapsed if elapsed > 0 else 0
                last_fps_time = current_time
                
                # Hiển thị progress less frequently
                if processed_count % performance_config.display_info_frequency == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"⚡ Progress: {progress:.1f}% | Processed: {processed_count}/{frame_count} frames | FPS: {current_fps:.1f} | Violations: M={stt_m}, C={stt_ctb} | 🔴 Nhấn 'q' để dừng và xuất kết quả")
            
            # Memory cleanup định kỳ
            if processed_count % performance_config.cleanup_frequency == 0:
                import gc
                gc.collect()
            
            # Hiển thị thông tin thống kê với performance info
            draw_text(frame, f"Vi pham xe may: {stt_m}", (10, 30), text_color_bg=(0,0,0))
            draw_text(frame, f"Vi pham o to: {stt_ctb}", (10, 60), text_color_bg=(0,0,0))
            draw_text(frame, f"Performance: {performance_config.mode}", (10, 90), text_color_bg=(0,0,0))
            draw_text(frame, f"Frame: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", (10, 120), text_color_bg=(0,0,0))
            draw_text(frame, f"Recording: ON", (10, 150), text_color=(0, 255, 0), text_color_bg=(0,0,0))
            
            # Thêm hướng dẫn điều khiển
            draw_text(frame, f"Nhan 'q' de dung va xuat ket qua", (10, new_height - 60), text_color=(255, 255, 0), text_color_bg=(0,0,0))
            draw_text(frame, f"Nhan 'p' de tam dung", (10, new_height - 30), text_color=(255, 255, 255), text_color_bg=(0,0,0))

            # Ghi frame vào video output
            out.write(frame)

            # Hiển thị frame
            cv2.imshow("Traffic Analysis - Optimized", frame)

            # Xử lý phím bấm với delay tối ưu
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n" + "="*60)
                print("🛑 ĐANG DỪNG PHÂN TÍCH VÀ XUẤT KẾT QUẢ...")
                print("="*60)
                print("⏹️  Dừng thu thập dữ liệu...")
                print("💾 Chuẩn bị xuất video và file thống kê...")
                print("⏳ Vui lòng đợi trong giây lát...")
                break
            elif key == ord('p'):
                print("⏸️ Tạm dừng - Nhấn phím bất kỳ để tiếp tục...")
                cv2.waitKey(-1) # Đợi phím bất kỳ để tiếp tục
                print("▶️ Tiếp tục phân tích...")

    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("🛑 NHẬN TÍNH HIỆU DỪNG (Ctrl+C)")
        print("="*60)
        print("💾 Đang xuất kết quả hiện tại...")
    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi: {e}")
        print("💾 Đang cố gắng xuất kết quả hiện tại...")
        import traceback
        traceback.print_exc()
    finally:
        # --- Dọn dẹp ---
        print("🧹 Đang dọn dẹp và xuất kết quả...")
        
        # Đóng video capture và display window trước
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        print("✅ Đã dừng video capture")
        print("💾 Đang hoàn thiện file video...")
        
        # Đóng video writer
        if out:
            out.release()
        
        print("📼 Video đã được lưu thành công!")
        print("📊 Đang tạo file CSV thống kê...")
        
        # Xuất file CSV thống kê vi phạm
        if violations_data:
            csv_filename = os.path.join(OUTPUT_DIR, f"lane_violations_stats_{timestamp}.csv")
            df = pd.DataFrame(violations_data)
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"✅ File CSV thống kê đã được xuất: {os.path.basename(csv_filename)}")
            
            print("📄 Đang tạo file JSON chi tiết...")
            # Xuất file JSON chi tiết (optional)
            json_filename = os.path.join(OUTPUT_DIR, f"lane_violations_details_{timestamp}.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_info': {
                        'input_path': video_path,
                        'original_resolution': f"{original_width}x{original_height}",
                        'processed_resolution': f"{new_width}x{new_height}",
                        'fps': original_fps,
                        'total_frames': total_frames,
                        'processed_frames': processed_count,
                        'performance_mode': performance_config.mode
                    },
                    'violations': violations_data,
                    'summary': {
                        'total_violations': len(violations_data),
                        'motor_violations': stt_m,
                        'car_violations': stt_ctb,
                        'processing_time': time.time() - start_time
                    }
                }, f, indent=2, ensure_ascii=False)
            print(f"✅ File JSON chi tiết đã được xuất: {os.path.basename(json_filename)}")
        else:
            print("ℹ️ Không có vi phạm nào được phát hiện, không xuất file CSV.")
        
        # Thống kê cuối cùng
        total_time = time.time() - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0
        processing_ratio = processed_count / frame_count if frame_count > 0 else 0
        
        print("\n" + "="*70)
        print("🎉 XUẤT KẾT QUẢ HOÀN TẤT!")
        print("="*70)
        print(f"📊 THỐNG KÊ PHÂN TÍCH:")
        print(f"   🎬 Tổng frames: {frame_count:,}")
        print(f"   ⚡ Đã xử lý: {processed_count:,} ({processing_ratio*100:.1f}%)")
        print(f"   ⏱️  Thời gian: {total_time:.1f}s")
        print(f"   🚀 FPS trung bình: {avg_fps:.1f}")
        print(f"   🏍️  Vi phạm xe máy: {stt_m}")
        print(f"   🚗 Vi phạm ô tô: {stt_ctb}")
        print(f"   ⚙️  Performance mode: {performance_config.mode}")
        
        print(f"\n📁 CÁC FILE ĐÃ XUẤT:")
        print(f"   📼 Video: {os.path.basename(output_video_path)}")
        if violations_data:
            print(f"   📊 CSV: {os.path.basename(csv_filename)}")
            print(f"   📄 JSON: {os.path.basename(json_filename)}")
        
        print(f"\n📂 Vị trí lưu file: {OUTPUT_DIR}")
        
        # Mở thư mục output
        try:
            import subprocess
            print("📂 Đang mở thư mục kết quả...")
            subprocess.run(['explorer', OUTPUT_DIR], shell=True)
            print("✅ Đã mở thư mục output")
        except Exception as e:
            print(f"⚠️ Không thể mở thư mục tự động: {e}")
            print(f"📂 Vui lòng mở thủ công: {OUTPUT_DIR}")
        
        print("="*70)
        print("✨ HOÀN THÀNH! Cảm ơn bạn đã sử dụng hệ thống phân tích giao thông.")
        print("="*70)
