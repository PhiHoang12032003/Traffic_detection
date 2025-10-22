import datetime
import webbrowser
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import json
import time

from flask import Flask, jsonify, url_for, request, session, send_file
from flask import render_template, Response
from flask_cors import CORS
# from flask_mysqldb import MySQL  # Commented out for easier setup
# Note: testHelmetNew functions are replaced with processHelmetVideo functionality
from processHelmetVideo import process_helmet_video_complete
import threading
import uuid
from testLane import *
# from testRedLight import video_detect_red_light  # Removed - using new red_light_main system
from red_light_main import process_red_light_video_complete, generate_frames_red_light_new
import createBB
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, static_folder='static')
CORS(app)
app.secret_key = 'your-secret-key-here-change-in-production'

# Global variables for lane detection control
lane_detection_active = False
lane_detection_thread = None
lane_detection_data = {
    'violations': [],
    'motor_violations': 0,
    'car_violations': 0,
    'start_time': None,
    'video_writer': None,
    'output_path': None,
    'timestamp': None,
    'tracked_vehicles': {},  # Tracking để tránh duplicate
    'violation_cooldown': {},  # Cooldown để tránh spam detection
    'violation_frames': [],  # Lưu frames vi phạm để xuất video
    'frame_count': 0
}

# Configure upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create upload folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# MySQL Configuration - Commented out for easier setup
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = '12345678'
# app.config['MYSQL_DB'] = 'datn'
# mysql = MySQL(app)


# Apply Flask CORSx`
# CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
#
@app.route('/test', methods=['GET'])
def get_violate():
    # MySQL functionality disabled for easier setup
    # try:
    #     cur = mysql.connection.cursor()
    #     cur.execute(
    #         "SELECT nametransportation.vh_name, MAX(transportationviolation.date_violate) as date_violate, COUNT(*) AS total_violate FROM transportationviolation INNER JOIN nametransportation ON transportationviolation.id_name = nametransportation.id_name GROUP BY nametransportation.id_name, nametransportation.vh_name;")
    #     users = cur.fetchall()
    #     cur.close()
    #     return jsonify(users)
    # except Exception as e:
    #     # Fallback data when database is not available
    #     print(f"Database error: {e}")
    sample_data = [
        ["OTO", "2024-01-15", 12],
        ["Xe May", "2024-01-15", 25], 
        ["Xe Dap", "2024-01-15", 3],
        ["Xe Tai", "2024-01-15", 8],
        ["Xe Bus", "2024-01-15", 2]
    ]
    return jsonify(sample_data)


@app.route('/test1', methods=['GET'])
def get_violate_current():
    # MySQL functionality disabled for easier setup
    # try:
    #     cur = mysql.connection.cursor()
    #     cur.execute(
    #         "SELECT nametransportation.vh_name, MAX(transportationviolation.date_violate) as date_violate, COUNT(*) AS total_violate FROM transportationviolation INNER JOIN nametransportation ON transportationviolation.id_name = nametransportation.id_name WHERE transportationviolation.date_violate = curdate() GROUP BY nametransportation.id_name, nametransportation.vh_name;")
    #     users = cur.fetchall()
    #     cur.close()
    #     return jsonify(users)
    # except Exception as e:
    #     # Fallback data for current day when database is not available
    #     print(f"Database error: {e}")
    sample_data = [
        ["OTO", "2024-01-15", 3],
        ["Xe May", "2024-01-15", 7], 
        ["Xe Tai", "2024-01-15", 2],
        ["Xe Bus", "2024-01-15", 1]
    ]
    return jsonify(sample_data)


# MySQL database insert function - Disabled for easier setup
# @app.route('/create', methods=['GET'])
def create(cls):
    # MySQL functionality disabled
    # with app.app_context():
    #     cur = mysql.connection.cursor()
    #     ngay_hien_tai = datetime.date.today()
    #     cur.execute("insert into transportationviolation(id_name , date_violate) values (%s, %s)",
    #                 (cls + 1, ngay_hien_tai))
    #     mysql.connection.commit()
    #     cur.close()
    #     return jsonify({'message': 'User created successfully'})
    
    # Simple logging instead of database insert
    ngay_hien_tai = datetime.date.today()
    print(f"Violation detected: Class {cls + 1} on {ngay_hien_tai}")
    return None


def call_route(cls):
    url_for('create', cls=cls)
    # return redirect(url_for('create', cls=cls))


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


def video_detection_web(path_x=""):
    """Video detection có thể điều khiển từ web với xuất kết quả"""
    global lane_detection_active, lane_detection_data
    
    try:
        from performance_config import PerformanceConfig, auto_detect_performance
        
        # Khởi tạo performance config
        performance_config = auto_detect_performance()
        
        cap = cv2.VideoCapture(path_x)
        if not cap.isOpened():
            print("❌ Không thể mở video")
            return
            
        model = YOLO('best_new/vehicle.pt')
        examBB = createBB.infoObject()
        
        # Get video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate new dimensions
        new_width, new_height = performance_config.get_video_dimensions(original_width, original_height)
        
        # Initialize video writer
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"lane_violations_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (new_width, new_height))
        
        # Update global data
        lane_detection_data.update({
            'start_time': time.time(),
            'video_writer': out,
            'output_path': output_path,
            'timestamp': timestamp,
            'original_fps': original_fps,
            'output_size': (new_width, new_height),
            'violations': [],
            'motor_violations': 0,
            'car_violations': 0,
            'tracked_vehicles': {},
            'violation_cooldown': {},
            'violation_frames': [],  # Reset violation frames
            'frame_count': 0
        })
        
        frame_count = 0
        processed_count = 0
        
        print(f"🚀 Bắt đầu phân tích web - Performance: {performance_config.mode}")
        
        while lane_detection_active and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # Resize frame immediately so output video uses consistent dimensions
            frame = cv2.resize(frame, (new_width, new_height))
            h, w, _ = frame.shape

            # Update global frame count for status/debug
            lane_detection_data['frame_count'] = frame_count

            # Vẽ UI elements cơ bản cho mọi frame để tránh chớp chớp
            
            # Draw ROI và lanes cho mọi frame (consistent UI)
            roi_start = (0, int(0.2 * h))
            roi_end = (w, int(0.8 * h))
            cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)
            
            # Làn xe máy: 0% - 50% width
            start_line_motor = (0, int(0.2 * h))
            end_line_motor = (int(0.50 * w), int(0.8 * h))
            cv2.rectangle(frame, start_line_motor, end_line_motor, (255, 0, 255), 2)
            draw_text(frame, "LANE XE MAY", (10, int(0.25 * h)), text_color=(255, 255, 255), text_color_bg=(255, 0, 255))
            
            # Làn ô tô: 50% - 100% width  
            start_line_car = (int(0.50 * w), int(0.2 * h))
            end_line_car = (w, int(0.8 * h))
            cv2.rectangle(frame, start_line_car, end_line_car, (0, 255, 0), 2)
            draw_text(frame, "LANE O TO", (int(0.52 * w), int(0.25 * h)), text_color=(255, 255, 255), text_color_bg=(0, 255, 0))
            
            # If this frame should be skipped for heavy processing, still write and stream
            # the frame with basic UI so the exported video keeps full length/duration.
            if not performance_config.should_process_frame(frame_count):
                # Vẽ thống kê cơ bản cho skipped frame
                draw_text(frame, f"XE MÁY VI PHẠM: {lane_detection_data['motor_violations']}", 
                         (10, 30), text_color=(255, 255, 255), text_color_bg=(255, 0, 0))
                draw_text(frame, f"Ô TÔ VI PHẠM: {lane_detection_data['car_violations']}", 
                         (10, 60), text_color=(255, 255, 255), text_color_bg=(0, 0, 255))
                draw_text(frame, f"Frame: {frame_count} (SKIPPED)", (10, 90), text_color_bg=(128,128,128))
                
                try:
                    out.write(frame)
                except Exception:
                    pass

                # Yield the frame with consistent UI for streaming
                yield frame
                continue

            processed_count += 1
            
            # Run YOLO cho processed frames
            results = model(frame, 
                          conf=performance_config.yolo_conf_threshold,
                          imgsz=performance_config.yolo_img_size,
                          verbose=False)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        try:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Chỉ xử lý trong ROI
                            if not (roi_start[1] < center_y < roi_end[1]):
                                continue
                            
                            # Vẽ bounding box
                            label = f"{result.names[cls]} {conf:.2f}"
                            color = (0, 255, 0) if cls == 1 else (0, 255, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            draw_text(frame, label, pos=(x1, y1 - 20), font_scale=0.5, text_color_bg=color)
                            
                            # LOGIC VI PHẠM SIÊU CHÍNH XÁC - CHỈ ĐẾM ĐÚNG 1 LẦN
                            violation_detected = False
                            violation_type = ""
                            
                            # Ranh giới làn đường: chính xác 50%
                            lane_boundary = w // 2
                            
                            # Tạo unique key dựa trên vùng SIÊU LỚN để đảm bảo 1 xe = 1 key
                            # Grid size = 400 pixels (rất lớn) để tránh duplicate key
                            grid_size = 400  
                            grid_x = center_x // grid_size
                            grid_y = center_y // grid_size
                            
                            # Debug info (comment out for less noise)
                            # print(f"🚗 Vehicle detected: cls={cls}, center=({center_x},{center_y}), grid=({grid_x},{grid_y}), lane_boundary={lane_boundary}")
                            
                            # CASE 1: XE MÁY (cls=1) VI PHẠM VÀO LÀN Ô TÔ (>= 50%)
                            if cls == 1 and center_x >= lane_boundary:
                                # Tạo key duy nhất cho xe này tại vị trí này
                                violation_key = f"MOTOR_{grid_x}_{grid_y}_{cls}"
                                
                                # Log để debug (comment out for less noise)
                                # print(f"🏍️ Motor check: center_x={center_x} >= boundary={lane_boundary}, key={violation_key}")
                                
                                # Chỉ đếm nếu key chưa tồn tại
                                if violation_key not in lane_detection_data['violation_cooldown']:
                                    # VI PHẠM MỚI - INCREMENT COUNTER
                                    old_count = lane_detection_data['motor_violations']
                                    lane_detection_data['motor_violations'] += 1
                                    new_count = lane_detection_data['motor_violations']
                                    
                                    # Mark key để tránh recount
                                    lane_detection_data['violation_cooldown'][violation_key] = frame_count
                                    violation_detected = True
                                    violation_type = "xe_may_vi_pham_lan_oto"
                                    
                                    # Visual feedback
                                    draw_text(frame, "VI PHAM XE MAY!", (x1, y1 - 40), 
                                            text_color=(255, 255, 255), text_color_bg=(255, 0, 0))
                                    
                                    # CRITICAL LOG
                                    print(f"🚨🚨 [MOTOR INCREMENT] {old_count} -> {new_count} at frame {frame_count}")
                                    print(f"🚨🚨 [KEY] {violation_key} added to cooldown")
                                    print(f"�🚨 [TOTALS] Motor={new_count}, Car={lane_detection_data['car_violations']}")
                                else:
                                    # Key đã tồn tại - KHÔNG increment
                                    draw_text(frame, "ĐÃ GHI NHẬN", (x1, y1 - 40), 
                                            text_color=(255, 255, 0), text_color_bg=(128, 128, 128))
                                    print(f"⏭️ Motor skip: key {violation_key} already exists")
                            
                            # CASE 2: Ô TÔ (cls=0,3,4) VI PHẠM VÀO LÀN XE MÁY (< 50%)
                            elif cls in [0, 3, 4] and center_x < lane_boundary:
                                violation_key = f"CAR_{grid_x}_{grid_y}_{cls}"
                                
                                # Log để debug (comment out for less noise)
                                # print(f"🚗 Car check: center_x={center_x} < boundary={lane_boundary}, key={violation_key}")
                                
                                # Kiểm tra đã vi phạm chưa
                                if violation_key not in lane_detection_data['violation_cooldown']:
                                    # VI PHẠM MỚI - INCREMENT COUNTER
                                    old_count = lane_detection_data['car_violations']
                                    lane_detection_data['car_violations'] += 1
                                    new_count = lane_detection_data['car_violations']
                                    
                                    # Mark key để tránh recount
                                    lane_detection_data['violation_cooldown'][violation_key] = frame_count
                                    violation_detected = True
                                    violation_type = "oto_vi_pham_lan_xe_may"
                                    
                                    # Visual feedback
                                    draw_text(frame, "VI PHAM Ô TÔ!", (x1, y1 - 40),
                                            text_color=(255, 255, 255), text_color_bg=(255, 0, 0))
                                    
                                    # CRITICAL LOG
                                    print(f"🚨🚨 [CAR INCREMENT] {old_count} -> {new_count} at frame {frame_count}")
                                    print(f"🚨🚨 [KEY] {violation_key} added to cooldown")
                                    print(f"🚨🚨 [TOTALS] Motor={lane_detection_data['motor_violations']}, Car={new_count}")
                                    
                                    # Log chi tiết  
                                    print(f"� [CAR] Vi phạm #{lane_detection_data['car_violations']} - Frame {frame_count} - Key: {violation_key}")
                                    print(f"📊 Current totals: Motor={lane_detection_data['motor_violations']}, Car={lane_detection_data['car_violations']}")
                                else:
                                    # Đã ghi nhận - không tăng counter
                                    draw_text(frame, "ĐÃ GHI NHẬN", (x1, y1 - 40),
                                            text_color=(255, 255, 0), text_color_bg=(128, 128, 128))
                            
                            # Lưu thông tin vi phạm nếu có vi phạm mới
                            if violation_detected:
                                violation_info = {
                                    'violation_id': len(lane_detection_data['violations']) + 1,
                                    'type': violation_type,
                                    'frame_number': frame_count,
                                    'time_seconds': frame_count / original_fps,
                                    'time_formatted': f"{int((frame_count / original_fps) // 60):02d}:{int((frame_count / original_fps) % 60):02d}",
                                    'confidence': conf,
                                    'bbox': [x1, y1, x2, y2],
                                    'center': [center_x, center_y],
                                    'vehicle_class': result.names[cls],
                                    'detected_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                lane_detection_data['violations'].append(violation_info)
                                
                                # Lưu frame vi phạm để xuất video
                                lane_detection_data['violation_frames'].append({
                                    'frame': frame.copy(),
                                    'frame_number': frame_count,
                                    'violation_type': violation_type
                                })
                                
                                print(f"📊 TOTAL VIOLATIONS: Motor={lane_detection_data['motor_violations']}, Car={lane_detection_data['car_violations']}")
                                
                                
                        except Exception as e:
                            print(f"Lỗi xử lý detection: {e}")
                            continue
            
            # Debug logging mỗi 60 frames để track counter changes (reduced frequency)
            if frame_count % 60 == 0:
                active_keys = len(lane_detection_data['violation_cooldown'])
                print(f"🔍 [FRAME {frame_count}] Status: Motor={lane_detection_data['motor_violations']}, Car={lane_detection_data['car_violations']}, Cooldowns={active_keys}")
                # if active_keys > 0:
                #     print(f"   Keys: {list(lane_detection_data['violation_cooldown'].keys())[:3]}")  # Show first 3 keys only
            
            # Làm sạch violation_cooldown mỗi 150 frames để tránh memory leak  
            if frame_count % 150 == 0:
                # Xóa tracked vehicles cũ hơn 5 giây (150 frames) 
                expired_keys = [k for k, v in lane_detection_data['violation_cooldown'].items() 
                              if frame_count - v > 150]
                for k in expired_keys:
                    del lane_detection_data['violation_cooldown'][k]
                    
                if expired_keys:
                    print(f"🧹 Cleaned {len(expired_keys)} old tracking entries at frame {frame_count}")
                    print(f"🔢 After cleanup: Motor={lane_detection_data['motor_violations']}, Car={lane_detection_data['car_violations']}")
            
            # Hiển thị thống kê trực quan với màu sắc rõ ràng
            draw_text(frame, f"XE MÁY VI PHẠM: {lane_detection_data['motor_violations']}", 
                     (10, 30), text_color=(255, 255, 255), text_color_bg=(255, 0, 0))
            draw_text(frame, f"Ô TÔ VI PHẠM: {lane_detection_data['car_violations']}", 
                     (10, 60), text_color=(255, 255, 255), text_color_bg=(0, 0, 255))
            draw_text(frame, f"Frame: {frame_count}/{total_frames} ({(frame_count/total_frames*100):.1f}%)", 
                     (10, 90), text_color_bg=(0,0,0))
            draw_text(frame, f"Tổng vi phạm: {len(lane_detection_data['violations'])}", 
                     (10, 120), text_color_bg=(0,0,0))
            draw_text(frame, f"Đang tracking: {len(lane_detection_data['violation_cooldown'])} xe", 
                     (10, 150), text_color_bg=(0,0,0))
            draw_text(frame, f"Chế độ: {performance_config.mode} - FPS: {original_fps:.1f}", 
                     (10, 180), text_color_bg=(0,0,0))
            
            # Ghi frame
            out.write(frame)
            
            # Throttle để tránh stream quá nhanh gây lag
            time.sleep(0.033)  # ~30fps throttle để mượt mà hơn
            
            yield frame
            
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Video detection hoàn thành - {processed_count} frames")
        
    except Exception as e:
        print(f"❌ Lỗi trong video_detection_web: {e}")
        import traceback
        traceback.print_exc()


def video_detection(path_x=""):
    """Hàm cũ - giữ lại để tương thích"""
    cap = cv2.VideoCapture(path_x)
    model = YOLO('best_new/vehicle.pt')
    stt_m = 0
    stt_ctb = 0
    examBB = createBB.infoObject()
    dataBienBan_M = 'BienBanNopPhatXeMay/'
    dataBienBan_CTB = 'BienBanNopPhatXeOTo/'

    # results = model.track(source="Videos/test4.mp4", show=True, stream=True)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            #  Dự đoán
            results = model(frame)

            # lấy ra frame sau khi đc gắn nhãn
            annotated_frame = results[0].plot()

            # lấy kích thước (height , width , _ )
            # print("kích thước frame : ", annotated_frame.shape)

            # Hiển thị lên
            # cv2.imshow("Display ", annotated_frame)
            # results = model.track(source="Videos/test4.mp4", show=True, tracker="bytetrack.yaml", stream=True)
            for result in results:
                boxes = result.boxes.numpy()

                # Lấy tên class
                name = result.names

                # lấy tất cả các thông số trong một list tọa độ các đối tượng (x0 ,y0, x1, y1, )
                # print("list 1 ", boxes.xyxy)
                list_2 = []

                # Lấy tất các các thông số của nhiều đối tượng (x0, y0 , x1 , y1 , id ,độ chính xác , loại class)
                # print("Boxes ", boxes)

                for box in boxes:
                    # lấy tên class tương ứng bounding box trong model đã custom
                    # print("Class : ", box.cls)

                    # lấy tọa độ của bounding box đối tượng (x0y0 , x1y1)
                    print("xyxy : ", box.xyxy[0])

                    # Lấy độ chính xác của bounding box đối tượng
                    # print("Độ chính xác : ", box.conf)

                    print("ID------------------- ", box.id)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # box.xyxy trả về ma trận 2 chiều dạng [[x0, y0 , x1 ,y1]]
                    # đó là tọa độ bounding box
                    print("box.xyxy", box.xyxy)
                    # org (Tọa độ cần vẽ lên bounding box (x,y) )
                    # thêm int để lấy số nguyên (nghĩa là lấy x0 , y0 để vẽ lên bounding box)
                    org = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))

                    # fontScale (Độ lớn của chữ)
                    fontScale = 0.5

                    # Blue color in RGB (Màu sắc của chữ)
                    color = ()

                    # Line thickness of 2px (Độ dày của chữ )
                    thickness = 2

                    # Lấy tọa độ bounding box
                    x = int(box.xyxy[0][0])
                    y = int(box.xyxy[0][1])
                    w = int(box.xyxy[0][2])
                    h = int(box.xyxy[0][3])

                    text = str(name[box.cls[0]] + " ") + str(round(box.conf[0], 2))

                    #####################################################################
                    # Xe OTO vi pham lane XE MAY
                    start_line_motor = (0 * int(frame.shape[1] / 10), int((2 * frame.shape[0] / 10)))
                    # 11/20 = 5.5 / 10
                    end_line_motor = (11 * int(frame.shape[1] / 20), int(8 * frame.shape[0] / 10))
                    canh_bao_vi_pham_lane_xe_may = start_line_motor[0] < box.xyxy[0][0] < end_line_motor[0] and \
                                                   start_line_motor[1] < box.xyxy[0][
                                                       1] < end_line_motor[1]
                    #####################################################################

                    # ##################################################################
                    # Xe máy vi pham lane OTO
                    # lane xe ô tô (trục y phải khớp với vùng roi)
                    # trục x lấy 6/10 , trục y lấy 3/10
                    start_line_car = (22 * int(frame.shape[1] / 40), int((2 * frame.shape[0] / 10)))

                    # lấy từ 6/10 đến hết trục X , trục y lấy 8/10
                    end_line_car = (int(frame.shape[1]), int(8 * frame.shape[0] / 10))

                    canh_bao_vi_pham_lane_oto = start_line_car[0] < box.xyxy[0][0] < end_line_car[0] and \
                                                start_line_car[1] < box.xyxy[0][
                                                    1] < end_line_car[1]
                    # filterDataViolate(frame, (0, int(5 * frame.shape[0] / 10)),
                    #                   (int(frame.shape[1]), int(55 * frame.shape[0] / 10)))
                    center_x = (x + w) // 2
                    center_y = (y + h) // 2
                    filterData = 0 <= center_x <= (int(frame.shape[1])) and int(
                        5 * frame.shape[0] / 10) <= center_y <= int(
                        52 * frame.shape[0] / 100)
                    #####################################################################

                    # vẽ ra vùng lane xe máy và oto
                    # image = cv2.rectangle(frame, start_line_car, end_line_car
                    #                       , (0, 0, 255), thickness)
                    image = cv2.rectangle(frame, start_line_motor, end_line_motor
                                          , (255, 0, 255), thickness)

                    # xét vùng roi theo trục Y
                    if int((2 * frame.shape[0]) / 10) < int(box.xyxy[0][1]) < int((8 * frame.shape[0]) / 10):
                        cv2.rectangle(frame, (x, y), (w, h), (36, 255, 12), 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        if box.cls[0] == 1:
                            if canh_bao_vi_pham_lane_oto:
                                draw_text(frame, name[box.cls[0]] + " warning", font_scale=0.5,
                                          pos=(int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                          text_color_bg=(0, 0, 0))
                                print("tọa độ xe máy vi phạm : ", box.xyxy[0])
                                # cắt hình ảnh xe máy
                                # cropped_frame = frame[round(y, 1) - 100:round(y + h, 2) + 100,
                                #                 round(x, 1) - 100: round(x + w, 1) + 100]

                                # Cắt hình làn ô tô
                                # cropped_frame = frame[int((3 * frame.shape[0]) / 10):int((8 * frame.shape[0]) / 10),
                                #                 6 * int(frame.shape[1] / 10):int(frame.shape[1])]
                                if filterData:
                                    stt_m += 1
                                    imageMotorViolate(frame, int((2 * frame.shape[0]) / 10),
                                                      int((8 * frame.shape[0]) / 10), 2 * int(frame.shape[1] / 10),
                                                      int(frame.shape[1]), stt_m)
                                    stt_BB_m = dataBienBan_M + str(stt_m) + '.pdf'
                                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    # Tạo tệp tạm thời và lưu ảnh PIL vào đó
                                    temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                                    frame_pil.save(temp_image.name)
                                    create(box.cls[0])  # Log violation (MySQL disabled)
                                    createBB.bienBanNopPhat(examBB,
                                                            temp_image.name,
                                                            "data_xe_may_vi_pham/" + str(
                                                                stt_m) + '.jpg',
                                                            stt_BB_m)
                                    temp_image.close()

                                    # cv2.imwrite("F:\python_project\data_xe_may_vi_pham\ " + str(count) + ".xe_may_lan_lan.jpg",
                                    #             cropped_frame)
                                    # frame = cv2.putText(frame, name[box.cls[0]] + " warning", org, font, fontScale, (0, 0, 255),
                                    #                     thickness, cv2.LINE_AA)
                            else:
                                draw_text(frame, text, font_scale=0.5,
                                          pos=(int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                          text_color=(255, 255, 255), text_color_bg=(78, 235, 133))
                                # frame = cv2.putText(frame, text, org, font, fontScale,
                                #                     generate_random_color(int(box.cls[0])), thickness,
                                #                     cv2.LINE_AA)
                        if box.cls[0] == 0 or box.cls[0] == 3 or box.cls[0] == 4:
                            if canh_bao_vi_pham_lane_xe_may:
                                draw_text(frame, name[box.cls[0]] + " warning", font_scale=0.5,
                                          pos=(int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                          text_color_bg=(0, 0, 0))
                                # Cắt hình làn ô tô
                                if filterData:
                                    stt_ctb += 1
                                    cropped_frame = frame[
                                                    int((3 * frame.shape[0]) / 10):int((8 * frame.shape[0]) / 10),
                                                    6 * int(frame.shape[1] / 10):int(frame.shape[1])]
                                    imageCTBViolate(frame, int((2 * frame.shape[0]) / 10),
                                                    int((8 * frame.shape[0]) / 10), 0 * int(frame.shape[1] / 10),
                                                    6 *
                                                    int(frame.shape[1] / 10), stt_ctb)

                                    stt_BB_CTB = dataBienBan_CTB + str(stt_ctb) + '.pdf'
                                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    # Tạo tệp tạm thời và lưu ảnh PIL vào đó
                                    temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                                    frame_pil.save(temp_image.name)
                                    create(box.cls[0])  # Log violation (MySQL disabled)
                                    createBB.bienBanNopPhat(examBB,
                                                            temp_image.name,
                                                            "data_oto_vi_pham/" + str(
                                                                stt_ctb) + '.jpg',
                                                            stt_BB_CTB)
                                    temp_image.close()
                            else:
                                draw_text(frame, text, font_scale=0.5,
                                          pos=(int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                          text_color=(255, 255, 255), text_color_bg=(77, 229, 26))

                    # muốn lấy 5/10 phần của height tính từ trên xuống
                    start_point = (0, int((2 * frame.shape[0]) / 10))
                    # vẽ hết chiều rộng và chiểu cao lấy 9/10
                    end_point = (int(frame.shape[1]), int((8 * frame.shape[0]) / 10))
                    color = (255, 0, 0)
                    thickness = 2

                    # vẽ ra cái ROI
                    image = cv2.rectangle(frame, start_point, end_point, color, thickness)

                    # scale_percent = 30
                    # width = int(image.shape[1] * scale_percent / 100)
                    # height = int(image.shape[0] * scale_percent / 100)
                    # dim = (width, height)

                    # resize Image
                    # resize = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                    # cv2.imshow("Roi ", image)
                    yield image
        else:
            break
    cv2.destroyAllWindows()


def reset_lane_detection_data():
    """Reset toàn bộ dữ liệu lane detection cho phiên mới"""
    global lane_detection_data
    
    print("🔄 RESET lane detection data for new session...")
    
    lane_detection_data.update({
        'violations': [],
        'motor_violations': 0,
        'car_violations': 0,
        'start_time': datetime.datetime.now(),
        'video_writer': None,
        'output_path': None,
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        'tracked_vehicles': {},
        'violation_cooldown': {},  # QUAN TRỌNG: Reset tracking để tránh duplicate
        'violation_frames': [],
        'frame_count': 0
    })
    
    print(f"✅ Lane detection data reset complete - Timestamp: {lane_detection_data['timestamp']}")


def generate_frames_lane(path_x):
    """Generate frames cho lane detection với khả năng điều khiển từ web"""
    global lane_detection_active
    
    # RESET DỮ LIỆU KHI BẮT ĐẦU PHIÊN MỚI
    reset_lane_detection_data()
    lane_detection_active = True
    
    print(f"🎬 Starting new lane detection session for: {path_x}")
    
    try:
        yolo_output = video_detection_web(path_x)
        for detection_ in yolo_output:
            if not lane_detection_active:
                print("🛑 Lane detection stopped by user request")
                break
            
            try:
                # Encode với quality cao để tránh artifacts gây chớp chớp
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # Chất lượng cao hơn
                ref, buffer = cv2.imencode('.jpg', detection_, encode_params)
                
                if ref:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    print("⚠️ Failed to encode frame, skipping...")
                    
            except Exception as e:
                print(f"⚠️ Error encoding frame: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error in video detection: {e}")
        import traceback
        traceback.print_exc()


def generate_frames(path_x):
    """Hàm cũ - giữ lại để tương thích"""
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_helmet(path_x):
    # Defensive: video_detect_helmet_with_plate may exist in another module. If not, fallback to placeholder stream
    # Try to obtain function dynamically to avoid static NameError
    func = globals().get('video_detect_helmet_with_plate')
    if func is None:
        # Fallback: return a simple placeholder stream
        import numpy as np
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Helmet detection not available", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            ref, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    yolo_output = func(path_x)

    try:
        for detection_ in yolo_output:
            ref, buffer = cv2.imencode('.jpg', detection_)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except GeneratorExit:
        # Clean up when client disconnects
        pass
    finally:
        # Ensure video is properly closed
        cv2.destroyAllWindows()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/Hethongcamera2")
def camera_2():
    return render_template("HelmetViolate.html")


@app.route("/bb")
def bb():
    return render_template("bb.html")


@app.route("/thongke")
def tk():
    return render_template("thongke.html")


@app.route("/Hethongcamera1")
def camera_1():
    return render_template("LaneViolate.html")


@app.route("/camera1")
def video():
    # Check if video has been uploaded for lane detection
    uploaded_video = session.get('uploaded_video_lane')
    if not uploaded_video:
        return "No video uploaded for lane detection", 400
    
    return Response(generate_frames_lane(path_x=uploaded_video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/camera2")
def video_2():
    # Check if processed video is available
    processing_info = session.get('helmet_processing', {})
    
    # If processing is complete, serve the processed video
    if processing_info.get('status') == 'completed':
        processed_video = processing_info.get('output_path')
        if processed_video and os.path.exists(processed_video):
            # Serve the processed video file directly
            return send_file(processed_video, mimetype='video/mp4')
    
    # Otherwise show no video message
    return "No processed video available", 400


# Helper function to check allowed file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route for red light violation detection page
@app.route("/test_upload")
def test_upload():
    return send_file("test_upload.html")

@app.route("/Hethongcamera3")
def camera_3():
    return render_template("RedLightViolate.html")


# Route for red light video stream
@app.route("/camera3")
def video_3():
    # Debug session info
    print(f"DEBUG camera3: Session keys: {list(session.keys())}")
    print(f"DEBUG camera3: uploaded_video = {session.get('uploaded_video', 'NOT_FOUND')}")
    
    # Check if there's an uploaded video
    uploaded_video = session.get('uploaded_video', None)
    if not uploaded_video:
        print("DEBUG: No uploaded video in session, showing placeholder")
        # Instead of returning 400 error, return a placeholder stream
        return Response(generate_placeholder_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Check if file exists
    if not os.path.exists(uploaded_video):
        print(f"DEBUG: File {uploaded_video} does not exist, showing placeholder")
        return Response(generate_placeholder_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    print(f"DEBUG: Starting video stream for {uploaded_video}")
    
    # Check if user wants advanced detection
    use_advanced = session.get('red_light_advanced', False)
    
    if use_advanced:
        # Use advanced streaming with license plate detection
        return Response(generate_frames_red_light_new(uploaded_video),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Use basic streaming
        return Response(generate_frames_red_light(path_x=uploaded_video),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


# Route to process red light video and export results
@app.route("/process_red_light_video", methods=['POST'])
def process_red_light_video():
    """Process uploaded video and export analysis results"""
    print("DEBUG process_red_light_video: Processing request received")
    
    # Check if there's an uploaded video
    uploaded_video = session.get('uploaded_video', None)
    if not uploaded_video:
        return jsonify({'error': 'No video uploaded'}), 400
    
    # Check if file exists
    if not os.path.exists(uploaded_video):
        return jsonify({'error': 'Video file not found'}), 400
    
    try:
        # Process the video using the new system
        output_video, violation_count = process_red_light_video_complete(uploaded_video)
        
        # Get the CSV file (should be the latest one in output directory)
        output_dir = "output"
        csv_files = [f for f in os.listdir(output_dir) if f.startswith('violations_data_') and f.endswith('.csv')]
        
        if csv_files:
            # Get the most recent CSV file
            latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
            csv_path = os.path.join(output_dir, latest_csv)
        else:
            csv_path = None
        
        return jsonify({
            'success': True,
            'output_video': output_video,
            'violation_count': violation_count,
            'csv_path': csv_path,
            'message': f'Video processed successfully. Found {violation_count} violations.'
        })
        
    except Exception as e:
        print(f"ERROR processing video: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

# Route to download processed video
@app.route("/download_processed_video")
def download_processed_video():
    """Download the processed video file"""
    output_dir = "output"
    video_files = [f for f in os.listdir(output_dir) if f.startswith('processed_video_') and f.endswith('.mp4')]
    
    if video_files:
        # Get the most recent video file
        latest_video = max(video_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
        video_path = os.path.join(output_dir, latest_video)
        return send_file(video_path, as_attachment=True, download_name=latest_video)
    else:
        return "No processed video found", 404

# Route to download violations CSV
@app.route("/download_violations_csv")
def download_violations_csv():
    """Download the violations CSV file"""
    output_dir = "output"
    csv_files = [f for f in os.listdir(output_dir) if f.startswith('violations_data_') and f.endswith('.csv')]
    
    if csv_files:
        # Get the most recent CSV file
        latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
        csv_path = os.path.join(output_dir, latest_csv)
        return send_file(csv_path, as_attachment=True, download_name=latest_csv)
    else:
        return "No violations data found", 404

# Route to upload video for analysis
@app.route("/upload_video", methods=['POST'])
def upload_video():
    print("DEBUG upload_video: Upload request received")
    
    if 'video' not in request.files:
        print("DEBUG upload_video: No video in request files")
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        print("DEBUG upload_video: Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    print(f"DEBUG upload_video: File received: {file.filename}")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"DEBUG upload_video: Saving to: {filepath}")
        
        file.save(filepath)
        
        # Store in session
        session['uploaded_video'] = filepath
        session['red_light_advanced'] = True
        
        print(f"DEBUG upload_video: Session updated - uploaded_video = {session.get('uploaded_video')}")
        print(f"DEBUG upload_video: Session keys: {list(session.keys())}")
        
        # Always use advanced method (since we removed basic option)
        detection_method = 'advanced'
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'detection_method': detection_method,
            'filepath': filepath,  # Add this for debugging
            'message': f'Video uploaded successfully! Using {detection_method} detection method.'
        })
    
    print(f"DEBUG upload_video: Invalid file type for {file.filename}")
    return jsonify({'error': 'Invalid file type'}), 400


# Route to upload video for helmet detection
@app.route("/upload_video_helmet", methods=['POST'])
def upload_video_helmet():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'helmet_' + filename)
        file.save(filepath)
        
        # Generate unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        # Create output path
        os.makedirs('processed_videos', exist_ok=True)
        output_path = os.path.join('processed_videos', f'{job_id}_processed.mp4')
        
        # Store processing info in session
        session['helmet_processing'] = {
            'job_id': job_id,
            'status': 'processing',
            'input_path': filepath,
            'output_path': output_path
        }
        
        # For demonstration, process synchronously
        # In production, use a task queue like Celery for async processing
        session['helmet_processing']['message'] = 'Processing video, please wait...'
        
        # Return immediately to show processing status
        response = jsonify({
            'success': True, 
            'filename': filename,
            'job_id': job_id,
            'message': 'Video đang được xử lý. Vui lòng đợi...',
            'processing_url': '/process_helmet_now'
        })
        
        # Get detection method from form
        detection_method = request.form.get('detection_method', 'original')
        use_advanced = detection_method == 'advanced'
        
        # Store filepath for processing
        session['pending_helmet_process'] = {
            'input': filepath,
            'output': output_path,
            'job_id': job_id,
            'use_advanced': use_advanced
        }
        
        return response
    
    return jsonify({'error': 'Invalid file type'}), 400


# Route to actually process the helmet video
@app.route("/process_helmet_now", methods=['GET'])
def process_helmet_now():
    pending = session.get('pending_helmet_process')
    if not pending:
        return jsonify({'error': 'No pending process'}), 400
    
    try:
        # Process the video using selected detection method
        use_advanced = pending.get('use_advanced', False)
        result_path, stats = process_helmet_video_complete(
            pending['input'], 
            pending['output'],
            use_improved_detection=use_advanced
        )
        
        # Update session
        session['helmet_processing']['status'] = 'completed'
        session['helmet_processing']['stats'] = stats
        session['helmet_processing']['output_path'] = result_path
        session['uploaded_video_helmet'] = result_path
        
        # Clear pending
        session.pop('pending_helmet_process', None)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'message': 'Processing completed!'
        })
    except Exception as e:
        session['helmet_processing']['status'] = 'error'
        session['helmet_processing']['error'] = str(e)
        return jsonify({'error': str(e)}), 500


# Route to upload video for lane detection
@app.route("/upload_video_lane", methods=['POST'])
def upload_video_lane():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'lane_' + filename)
        file.save(filepath)
        
        # Store in session
        session['uploaded_video_lane'] = filepath
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'message': 'Video uploaded successfully for lane detection!'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


# API để dừng lane detection và xuất kết quả
@app.route("/stop_lane_detection", methods=['POST'])
def stop_lane_detection():
    """Dừng lane detection và xuất kết quả"""
    global lane_detection_active, lane_detection_data
    
    if not lane_detection_active:
        return jsonify({'error': 'Lane detection is not running'}), 400
    
    print("🛑 Nhận yêu cầu dừng từ web...")
    lane_detection_active = False
    
    # Đợi một chút để video detection dừng hoàn toàn
    time.sleep(2)
    
    try:
        # Đóng video writer nếu còn mở
        if lane_detection_data.get('video_writer'):
            lane_detection_data['video_writer'].release()
            lane_detection_data['video_writer'] = None
        
        violations_data = lane_detection_data.get('violations', [])
        timestamp = lane_detection_data.get('timestamp', datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_path = lane_detection_data.get('output_path', '')
        
        # Xuất CSV nếu có vi phạm
        csv_filename = None
        json_filename = None
        
        if violations_data:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Xuất CSV
            csv_filename = os.path.join(output_dir, f"lane_violations_stats_{timestamp}.csv")
            df = pd.DataFrame(violations_data)
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            
            # Xuất JSON chi tiết
            json_filename = os.path.join(output_dir, f"lane_violations_details_{timestamp}.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_info': {
                        'input_path': session.get('uploaded_video_lane', ''),
                        'output_path': output_path,
                        'timestamp': timestamp
                    },
                    'violations': violations_data,
                    'summary': {
                        'total_violations': len(violations_data),
                        'motor_violations': lane_detection_data.get('motor_violations', 0),
                        'car_violations': lane_detection_data.get('car_violations', 0),
                        'processing_time': time.time() - lane_detection_data.get('start_time', time.time())
                    }
                }, f, indent=2, ensure_ascii=False)
        
        # Inspect output video to report exact frame count and duration
        actual_video_info = None
        try:
            if output_path and os.path.exists(output_path):
                cap_out = cv2.VideoCapture(output_path)
                out_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
                out_fps = cap_out.get(cv2.CAP_PROP_FPS) or lane_detection_data.get('original_fps')
                out_duration = out_frames / (out_fps if out_fps > 0 else 1)
                cap_out.release()
                actual_video_info = {
                    'frames': out_frames,
                    'fps': out_fps,
                    'duration_seconds': out_duration,
                    'path': output_path
                }
        except Exception as e:
            print(f"⚠️ Unable to inspect output video: {e}")

        # Tạo response
        response_data = {
            'success': True,
            'message': 'Lane detection stopped and results exported successfully',
            'summary': {
                'total_violations': len(violations_data),
                'motor_violations': lane_detection_data.get('motor_violations', 0),
                'car_violations': lane_detection_data.get('car_violations', 0),
                'processing_time': time.time() - lane_detection_data.get('start_time', time.time()),
                'frames_processed_in_session': lane_detection_data.get('frame_count', 0),
                'expected_duration_seconds': lane_detection_data.get('frame_count', 0) / (lane_detection_data.get('original_fps', 1) or 1)
            },
            'files': {
                'video': os.path.basename(output_path) if output_path else None,
                'csv': os.path.basename(csv_filename) if csv_filename else None,
                'json': os.path.basename(json_filename) if json_filename else None
            },
            'video_info': actual_video_info
        }
        
        print(f"✅ Đã xuất kết quả thành công:")
        print(f"   - Video: {output_path}")
        if csv_filename:
            print(f"   - CSV: {csv_filename}")
            print(f"   - JSON: {json_filename}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Lỗi khi xuất kết quả: {e}")
        return jsonify({'error': f'Error exporting results: {str(e)}'}), 500


# API để lấy trạng thái hiện tại
@app.route("/lane_detection_status", methods=['GET'])
def get_lane_detection_status():
    """Lấy trạng thái hiện tại của lane detection với debug info chi tiết"""
    global lane_detection_active, lane_detection_data
    
    # Tính runtime
    if lane_detection_data.get('start_time') and isinstance(lane_detection_data['start_time'], datetime.datetime):
        current_time = datetime.datetime.now()
        runtime_delta = current_time - lane_detection_data['start_time']
        runtime_seconds = runtime_delta.total_seconds()
    else:
        runtime_seconds = 0
    
    # Debug info để kiểm tra accuracy
    debug_info = {
        'active_cooldowns': len(lane_detection_data.get('violation_cooldown', {})),
        'frame_count': lane_detection_data.get('frame_count', 0),
        'timestamp': lane_detection_data.get('timestamp', 'N/A'),
        'start_time': str(lane_detection_data.get('start_time', 'N/A')),
        'violations_list_length': len(lane_detection_data.get('violations', [])),
        'motor_count_direct': lane_detection_data.get('motor_violations', 0),
        'car_count_direct': lane_detection_data.get('car_violations', 0)
    }
    
    return jsonify({
        'active': lane_detection_active,
        'motor_violations': lane_detection_data.get('motor_violations', 0),
        'car_violations': lane_detection_data.get('car_violations', 0), 
        'total_violations': len(lane_detection_data.get('violations', [])),
        'runtime': runtime_seconds,
        'runtime_formatted': f"{int(runtime_seconds//3600):02d}:{int((runtime_seconds%3600)//60):02d}:{int(runtime_seconds%60):02d}",
        'debug': debug_info,
        'last_updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


# Route để download file đã xuất
@app.route("/download_lane_results/<file_type>")
def download_lane_results(file_type):
    """Download file kết quả (video, csv, json)"""
    try:
        output_dir = "output"
        timestamp = lane_detection_data.get('timestamp')
        
        if not timestamp:
            return "No results available", 404
        
        if file_type == "video":
            filename = f"lane_violations_{timestamp}.mp4"
        elif file_type == "csv":
            filename = f"lane_violations_stats_{timestamp}.csv"
        elif file_type == "json":
            filename = f"lane_violations_details_{timestamp}.json"
        else:
            return "Invalid file type", 400
        
        file_path = os.path.join(output_dir, filename)
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return "File not found", 404
            
    except Exception as e:
        return f"Error downloading file: {str(e)}", 500


# Generate frames for red light detection (basic streaming)
def generate_placeholder_stream():
    """Generate a placeholder stream when no video is uploaded"""
    import numpy as np
    
    while True:
        # Create a black frame with text
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add text overlay
        cv2.putText(frame, "CHUA CO VIDEO", (180, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, "Vui long upload video", (150, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, "de bat dau phat hien", (150, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, "vi pham vuot den do", (150, 310), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Add loading animation (simple rotating circle)
        import time
        angle = int(time.time() * 100) % 360
        center = (320, 380)
        radius = 30
        end_point = (int(center[0] + radius * np.cos(np.radians(angle))), 
                    int(center[1] + radius * np.sin(np.radians(angle))))
        cv2.circle(frame, center, radius, (100, 100, 100), 2)
        cv2.line(frame, center, end_point, (0, 255, 0), 3)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small delay to prevent high CPU usage
        time.sleep(0.1)


def generate_frames_red_light(path_x):
    """Basic red light detection - fallback when advanced detection is not used"""
    cap = cv2.VideoCapture(path_x)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Simple frame processing - just add text overlay
        cv2.putText(frame, "Basic Red Light Detection", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# Generate frames for red light detection with processing (advanced streaming)
def generate_frames_red_light_advanced(path_x):
    """Real-time streaming với advanced processing và license plate detection"""
    from red_light_main import process_red_light_video_complete, generate_frames_red_light_new
    import tempfile
    import threading
    import queue
    import time
    
    # Create frame queue for streaming
    frame_queue = queue.Queue(maxsize=30)  # Buffer 30 frames
    processing_active = True
    
    def process_video_thread():
        """Background thread để process video và đưa frames vào queue"""
        nonlocal processing_active
        try:
            # Tạo temporary output file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_output = temp_file.name
            
            # Process video với custom frame callback
            cap = cv2.VideoCapture(path_x)
            if not cap.isOpened():
                return
                
            # Import các function cần thiết từ processRedLightVideo
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            # Import helper functions (manually implement here for web streaming)
            def draw_simulated_traffic_light_web(frame, current_light):
                """Draw simulated traffic light for web streaming"""
                height, width = frame.shape[:2]
                
                # Traffic light position (top-right corner)
                light_width = 60
                light_height = 150
                light_x = width - light_width - 20
                light_y = 20
                
                # Draw traffic light background
                cv2.rectangle(frame, (light_x, light_y), (light_x + light_width, light_y + light_height), (0, 0, 0), -1)
                cv2.rectangle(frame, (light_x, light_y), (light_x + light_width, light_y + light_height), (255, 255, 255), 2)
                
                # Light positions
                circle_radius = 18
                circle_x = light_x + light_width // 2
                red_y = light_y + 30
                yellow_y = light_y + 75
                green_y = light_y + 120
                
                # Draw inactive lights
                cv2.circle(frame, (circle_x, red_y), circle_radius, (50, 50, 50), -1)
                cv2.circle(frame, (circle_x, yellow_y), circle_radius, (50, 50, 50), -1)
                cv2.circle(frame, (circle_x, green_y), circle_radius, (50, 50, 50), -1)
                
                # Draw active light
                if current_light == "red":
                    cv2.circle(frame, (circle_x, red_y), circle_radius, (0, 0, 255), -1)
                elif current_light == "yellow":
                    cv2.circle(frame, (circle_x, yellow_y), circle_radius, (0, 255, 255), -1)
                elif current_light == "green":
                    cv2.circle(frame, (circle_x, green_y), circle_radius, (0, 255, 0), -1)
                
                # Add light status text
                cv2.putText(frame, f"Light: {current_light.upper()}", 
                           (light_x - 50, light_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            def extract_license_plate_web(frame, bbox, reader):
                """Simple license plate extraction for web streaming"""
                if reader is None:
                    return None, None
                    
                try:
                    x1, y1, x2, y2 = bbox
                    vehicle_width = x2 - x1
                    vehicle_height = y2 - y1
                    
                    if vehicle_width < 40 or vehicle_height < 30:
                        return None, None
                    
                    # Focus on bottom part of vehicle  
                    crop_h = y2 - y1
                    crop_y1 = y1 + int(crop_h * 0.6)
                    license_region = frame[crop_y1:y2, x1:x2]
                    
                    if license_region.size == 0 or license_region.shape[0] < 10:
                        return None, None
                    
                    # Simple resize and OCR
                    scale_factor = 2.0 if vehicle_width < 80 else 1.5
                    enhanced = cv2.resize(license_region, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                    
                    if len(enhanced.shape) == 3:
                        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                    
                    # Simple OCR call
                    results = reader.readtext(enhanced, paragraph=False)
                    
                    best_text = None
                    best_confidence = 0
                    best_bbox = None
                    
                    for (bbox_coords, text, prob) in results:
                        if prob > 0.2:
                            cleaned_text = ''.join(c for c in text if c.isalnum())
                            
                            if len(cleaned_text) >= 4 and len(cleaned_text) <= 12:
                                has_letters = any(c.isalpha() for c in cleaned_text)
                                has_numbers = any(c.isdigit() for c in cleaned_text)
                                
                                if (has_letters and has_numbers) or (has_numbers and len(cleaned_text) >= 4):
                                    if prob > best_confidence:
                                        best_confidence = prob
                                        best_text = cleaned_text.upper()
                                        
                                        # Convert bbox back to original frame coordinates
                                        # bbox_coords is [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                                        ocr_x1 = int(min([p[0] for p in bbox_coords]) / scale_factor) + x1
                                        ocr_y1 = int(min([p[1] for p in bbox_coords]) / scale_factor) + crop_y1
                                        ocr_x2 = int(max([p[0] for p in bbox_coords]) / scale_factor) + x1
                                        ocr_y2 = int(max([p[1] for p in bbox_coords]) / scale_factor) + crop_y1
                                        
                                        best_bbox = (ocr_x1, ocr_y1, ocr_x2, ocr_y2)
                    
                    return best_text, best_bbox
                    
                except Exception as e:
                    return None, None
            from ultralytics import YOLO
            import easyocr
            import numpy as np
            from collections import defaultdict
            
            # Initialize models như trong processRedLightVideo
            model = YOLO('YoloWeights/yolov8n.pt')
            vehicle_model = YOLO('best_new/vehicle.pt')
            
            try:
                reader = easyocr.Reader(['vi', 'en'])
            except:
                reader = None
            
            # Initialize variables
            frame_count = 0
            current_light = "red"
            violation_count = 0
            processed_vehicles = set()
            frames_since_light_change = 0
            light_cycle_duration = 90
            light_patterns = ["red", "red", "yellow", "green", "red", "red"]
            current_light_index = 0
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Detection lines
            detection_lines = [{
                "start": {"x": int(width * 0.0), "y": int(height * 0.85)},
                "end": {"x": int(width * 0.6), "y": int(height * 0.90)}
            }]
            
            while processing_active:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Traffic light simulation
                frames_since_light_change += 1
                if frames_since_light_change >= light_cycle_duration:
                    frames_since_light_change = 0
                    current_light_index = (current_light_index + 1) % len(light_patterns)
                    current_light = light_patterns[current_light_index]
                
                # Detect vehicles and violations (simplified for streaming)
                vehicle_results = vehicle_model(frame)
                
                # Draw simulated traffic light
                draw_simulated_traffic_light_web(frame, current_light)
                
                # Process vehicles và license plates
                vehicles_with_plates = []
                for r in vehicle_results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if cls in [0, 1, 2, 3, 4] and conf > 0.3:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # Extract license plate (simplified for speed)
                                license_text = None
                                license_bbox = None
                                if reader and frame_count % 10 == 0:  # Every 10 frames
                                    try:
                                        license_text, license_bbox = extract_license_plate_web(frame, [x1, y1, x2, y2], reader)
                                    except:
                                        pass
                                
                                # Check violation
                                vehicle_id = f"{x1}_{y1}_{x2}_{y2}"
                                center_y = (y1 + y2) // 2
                                
                                is_violation = (current_light == "red" and 
                                              center_y > height * 0.85 and 
                                              vehicle_id not in processed_vehicles)
                                
                                if is_violation:
                                    violation_count += 1
                                    processed_vehicles.add(vehicle_id)
                                    
                                    # Draw violation
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.putText(frame, "VI PHAM!", (x1, y1 - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                else:
                                    # Normal vehicle
                                    color = (0, 255, 0) if current_light != "red" else (255, 255, 0)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Draw license plate if detected
                                if license_text and license_bbox:
                                    # Draw license plate bounding box
                                    lp_x1, lp_y1, lp_x2, lp_y2 = license_bbox
                                    cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 255, 0), 2)
                                    
                                    # Draw license plate text above the license plate box
                                    label = f"LP: {license_text}"
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    
                                    # Position above the license plate box
                                    text_x = max(0, lp_x1)
                                    text_y = max(label_size[1] + 5, lp_y1 - 5)
                                    
                                    # Ensure text stays within frame
                                    if text_x + label_size[0] > width:
                                        text_x = width - label_size[0] - 5
                                    if text_y < label_size[1] + 5:
                                        text_y = lp_y2 + label_size[1] + 10
                                    
                                    # Draw background rectangle for better visibility
                                    cv2.rectangle(frame, 
                                                (text_x, text_y - label_size[1] - 2), 
                                                (text_x + label_size[0] + 4, text_y + 2), 
                                                (0, 0, 0), -1)
                                    
                                    # Draw text
                                    cv2.putText(frame, label, (text_x + 2, text_y), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                elif license_text:
                                    # Fallback: Draw text above vehicle if no license bbox
                                    label = f"LP: {license_text}"
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    
                                    # Position above the vehicle box
                                    text_x = max(0, x1)
                                    text_y = max(label_size[1] + 5, y1 - 5)
                                    
                                    # Draw background rectangle for better visibility
                                    cv2.rectangle(frame, 
                                                (text_x, text_y - label_size[1] - 2), 
                                                (text_x + label_size[0] + 4, text_y + 2), 
                                                (0, 0, 0), -1)
                                    
                                    # Draw text
                                    cv2.putText(frame, label, (text_x + 2, text_y), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw detection lines
                for line in detection_lines:
                    start_x = int(line["start"]["x"])
                    start_y = int(line["start"]["y"])
                    end_x = int(line["end"]["x"])
                    end_y = int(line["end"]["y"])
                    
                    line_color = (0, 0, 255) if current_light == "red" else (255, 255, 255)
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), line_color, 3)
                
                # Draw info panel
                cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
                
                status_color = (0, 0, 255) if current_light == "red" else (0, 255, 0) if current_light == "green" else (0, 255, 255)
                cv2.putText(frame, f"Den giao thong: {current_light.upper()}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(frame, f"Vi pham: {violation_count}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "LIVE STREAM", (20, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add frame to queue (non-blocking)
                try:
                    frame_queue.put(frame, block=False)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
                    
            cap.release()
            
        except Exception as e:
            print(f"Error in processing thread: {e}")
        finally:
            processing_active = False
    
    # Start background processing thread
    processing_thread = threading.Thread(target=process_video_thread, daemon=True)
    processing_thread.start()
    
    # Stream frames from queue
    try:
        while processing_active or not frame_queue.empty():
            try:
                # Get frame from queue
                frame = frame_queue.get(timeout=1.0)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except queue.Empty:
                # No frame available, continue
                continue
                
    except GeneratorExit:
        # Client disconnected
        processing_active = False
    finally:
        processing_active = False


# Clear uploaded video
@app.route("/clear_upload", methods=['POST'])
def clear_upload():
    if 'uploaded_video' in session:
        # Delete file if exists
        filepath = session['uploaded_video']
        if os.path.exists(filepath):
            os.remove(filepath)
        session.pop('uploaded_video', None)
    return jsonify({'success': True})


# Clear uploaded video for helmet detection
@app.route("/clear_upload_helmet", methods=['POST'])
def clear_upload_helmet():
    if 'uploaded_video_helmet' in session:
        # Delete file if exists
        filepath = session['uploaded_video_helmet']
        if os.path.exists(filepath):
            os.remove(filepath)
        session.pop('uploaded_video_helmet', None)
    return jsonify({'success': True})


# API endpoint for helmet violation statistics
@app.route("/api/helmet_violations", methods=['GET'])
def get_helmet_violations():
    # In production, this would fetch from database or real-time detection
    # For now, return sample data
    violations = session.get('helmet_violations', [])
    return jsonify({
        'total_count': len(violations),
        'violations': violations[-10:]  # Last 10 violations
    })


# Check processing status
@app.route("/api/processing_status", methods=['GET'])
def get_processing_status():
    processing_info = session.get('helmet_processing', {})
    if not processing_info:
        return jsonify({'status': 'no_job'})
    
    response = {
        'status': processing_info.get('status', 'unknown'),
        'job_id': processing_info.get('job_id')
    }
    
    if processing_info.get('status') == 'completed':
        response['stats'] = processing_info.get('stats', {})
        response['output_path'] = processing_info.get('output_path')
    elif processing_info.get('status') == 'error':
        response['error'] = processing_info.get('error', 'Unknown error')
    
    return jsonify(response)


# New API endpoint for real-time progress tracking
@app.route("/api/helmet_progress", methods=['GET'])
def get_helmet_progress():
    """Get real-time progress of helmet detection processing"""
    # For original testHelmet.py, just return basic progress
    import glob
    violation_files = glob.glob('data_xe_vp_bh/*.jpg')
    progress = {
        'violations': len(violation_files),
        'total_frames': 0,
        'current_frame': 0,
        'status': 'completed',
        'estimated_time_left': 0,
        'formatted_time_left': "Hoàn thành"
    }
    
    return jsonify(progress)


# Reset progress tracking
@app.route("/api/helmet_progress/reset", methods=['POST'])
def reset_helmet_progress():
    """Reset progress tracking"""
    reset_processing_progress()
    return jsonify({'success': True, 'message': 'Progress reset successfully'})


def reset_processing_progress():
    """Reset processing progress for helmet detection"""
    # Clear session data
    if 'helmet_processing' in session:
        session.pop('helmet_processing', None)
    if 'pending_helmet_process' in session:
        session.pop('pending_helmet_process', None)
    if 'uploaded_video_helmet' in session:
        session.pop('uploaded_video_helmet', None)
    
    # You can add additional cleanup here if needed
    print("Processing progress reset")


# API endpoints for red light violations
@app.route("/api/red_light_violations", methods=['GET'])
def get_red_light_violations():
    """Get red light violation statistics"""
    import glob
    
    # Count violation images
    violation_files = glob.glob('data_vuot_den_do/*.jpg')
    fine_documents = glob.glob('BienBanNopPhatVuotDenDo/*.pdf')
    
    # Get recent violations
    recent_violations = []
    for file_path in sorted(violation_files, reverse=True)[:10]:
        filename = os.path.basename(file_path)
        # Extract timestamp from filename if possible
        parts = filename.split('_')
        if len(parts) >= 3:
            recent_violations.append({
                'filename': filename,
                'timestamp': parts[1] + '_' + parts[2].split('.')[0] if len(parts) > 2 else 'unknown',
                'file_path': file_path
            })
    
    return jsonify({
        'total_violations': len(violation_files),
        'total_fines': len(fine_documents),
        'recent_violations': recent_violations,
        'detection_method': 'Advanced' if session.get('red_light_advanced', False) else 'Basic'
    })


@app.route("/api/set_detection_method", methods=['POST'])
def set_detection_method():
    """Set red light detection method preference"""
    data = request.get_json()
    method = data.get('method', 'original')
    
    session['red_light_advanced'] = (method == 'advanced')
    
    return jsonify({
        'success': True,
        'method': method,
        'message': f'Detection method set to {method}'
    })


@app.route("/api/session_status", methods=['GET'])
def get_session_status():
    """Debug endpoint to check session status"""
    return jsonify({
        'session_keys': list(session.keys()),
        'uploaded_video': session.get('uploaded_video', 'NOT_FOUND'),
        'uploaded_video_exists': os.path.exists(session.get('uploaded_video', '')) if session.get('uploaded_video') else False,
        'red_light_advanced': session.get('red_light_advanced', False),
        'session_id': session.get('_session_id', 'NO_ID')
    })


if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:8000/')
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=True)
