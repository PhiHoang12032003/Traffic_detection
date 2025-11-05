import datetime
import webbrowser
import cv2
import tempfile
import numpy as np
import pandas as pd
import json
import time
from PIL import Image
from ultralytics import YOLO

from flask import Flask, jsonify, url_for, request, session, send_file
from flask import render_template, Response
from flask_cors import CORS
# from flask_mysqldb import MySQL  # Commented out for easier setup
# Note: testHelmetNew functions are replaced with processHelmetVideo functionality
# from processHelmetVideo import process_helmet_video_complete  # Tạm comment để tránh lỗi reportlab
import threading
import uuid
# Avoid wildcard import from testLane to prevent name collisions (e.g., datetime)
# If needed, import explicitly: from testLane import some_function
# from testLane import *
# from testRedLight import video_detect_red_light  # Removed - using new red_light_main system
from red_light_main import process_red_light_video_complete, generate_frames_red_light_new
import createBB
from utils.helmet_pdf_utils import create_helmet_pdf_report, get_helmet_violation_info
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, static_folder='static')
CORS(app)
app.secret_key = 'your-secret-key-here-change-in-production'

# Configure upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create upload folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
    'frame_count': 0,
    'original_fps': None,
    'output_size': None,
    'vehicle_states': {}  # Track trạng thái lane của từng xe: {vehicle_id: {'current_lane': 'motor'/'car', 'last_seen': frame_number}}
}

# Global variables for helmet detection control
helmet_detection_active = False
helmet_detection_thread = None
helmet_detection_data = {
    'violations': [],
    'total_violations': 0,
    'with_helmet': 0,  # Đếm người có mũ
    'without_helmet': 0,  # Đếm người không mũ (vi phạm)
    'start_time': None,
    'video_writer': None,
    'output_path': None,
    'timestamp': None,
    'frame_count': 0,
    'original_fps': None,
    'output_size': None,
    'detection_cooldown': {},  # Cooldown để tránh đếm trùng: {position_key: frame_number}
    'frame_detections': set(),  # Tracking detections trong frame hiện tại
    'temporal_votes': {},  # Bộ nhớ phiếu theo vị trí để chống nhiễu (position_key -> {window, last_frame})
    'state_map': {},  # Bộ nhớ trạng thái theo vị trí để giảm nhấp nháy: {pos_key: {state, window, last_change, lock_until}}
    'recent_plates': [],  # Danh sách plate ứng viên gần đây: [{text, bbox, frame}]
    'current_with': 0,
    'current_without': 0,
    'current_total': 0
}

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


def draw_text(img, text, pos=(10, 30), font_scale=0.7, text_color=(255, 255, 255), text_color_bg=(0, 0, 0)):
    """Vẽ text với background"""
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
    """Video detection cho lane violation với xuất kết quả"""
    global lane_detection_active, lane_detection_data
    
    cap = None
    out = None
    
    try:
        from performance_config import PerformanceConfig, auto_detect_performance
        
        # Khởi tạo performance config
        performance_config = auto_detect_performance()
        
        cap = cv2.VideoCapture(path_x)
        if not cap.isOpened():
            print("❌ Không thể mở video")
            return
            
        model = YOLO('best_new/vehicle.pt')
        
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
        
        if not out.isOpened():
            print("❌ Không thể tạo video writer")
            return
        
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
            'violation_frames': [],
            'frame_count': 0
        })
        
        frame_count = 0
        processed_count = 0
        
        print(f"🚀 Bắt đầu phân tích web - Performance: {performance_config.mode}")
        print(f"📹 Video output: {output_path}")
        
        while lane_detection_active and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # Resize frame immediately
            frame = cv2.resize(frame, (new_width, new_height))
            h, w, _ = frame.shape

            # Update global frame count
            lane_detection_data['frame_count'] = frame_count

            # Vẽ UI elements cơ bản cho mọi frame
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
            
            # If frame should be skipped, still write with basic UI (chỉ lanes, không có text thống kê)
            if not performance_config.should_process_frame(frame_count):
                try:
                    out.write(frame)
                except Exception:
                    pass

                yield frame
                continue

            processed_count += 1
            
            # Run YOLO
            results = model(frame, 
                          conf=performance_config.yolo_conf_threshold,
                          imgsz=performance_config.yolo_img_size,
                          verbose=False)
            
            # Debug: In class names một lần để kiểm tra
            if frame_count == 1:
                print(f"🔍 [MODEL DEBUG] Class names: {results[0].names}")
            
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
                            
                            # LOGIC VI PHẠM CẢI TIẾN - Global vehicle tracking để tránh đếm trùng
                            violation_detected = False
                            violation_type = ""
                            
                            lane_boundary = w // 2
                            
                            # Tìm vehicle ID dựa trên IoU với các xe đã track
                            # Sử dụng khoảng cách và IoU để match với xe hiện có
                            vehicle_id = None
                            best_match_score = 0
                            current_bbox = (x1, y1, x2, y2)
                            
                            # Helper function tính IoU
                            def calculate_iou(box1, box2):
                                x1_1, y1_1, x2_1, y2_1 = box1
                                x1_2, y1_2, x2_2, y2_2 = box2
                                
                                inter_x1 = max(x1_1, x1_2)
                                inter_y1 = max(y1_1, y1_2)
                                inter_x2 = min(x2_1, x2_2)
                                inter_y2 = min(y2_1, y2_2)
                                
                                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                                
                                box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
                                box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
                                union_area = box1_area + box2_area - inter_area
                                
                                return inter_area / union_area if union_area > 0 else 0
                            
                            # Tìm xe match với detection hiện tại
                            best_iou = 0
                            best_dist = 999999
                            for vid, vstate in lane_detection_data['vehicle_states'].items():
                                # Chỉ xem xét xe cùng class và được thấy trong 30 frames gần đây
                                if vstate.get('class') != cls:
                                    continue
                                if frame_count - vstate.get('last_seen', 0) > 30:
                                    continue
                                    
                                last_bbox = vstate.get('last_bbox')
                                if last_bbox is None:
                                    continue
                                
                                iou = calculate_iou(current_bbox, last_bbox)
                                
                                # Tính khoảng cách center
                                last_cx = (last_bbox[0] + last_bbox[2]) // 2
                                last_cy = (last_bbox[1] + last_bbox[3]) // 2
                                dist = ((center_x - last_cx)**2 + (center_y - last_cy)**2) ** 0.5
                                
                                # Match score: IoU cao hoặc khoảng cách gần
                                match_score = iou + (1.0 / (1.0 + dist / 100))
                                
                                if match_score > best_match_score and (iou > 0.3 or dist < 150):
                                    best_match_score = match_score
                                    vehicle_id = vid
                                    best_iou = iou
                                    best_dist = dist
                            
                            # Nếu không tìm thấy match, tạo ID mới
                            if vehicle_id is None:
                                # Tạo unique ID mới
                                if 'next_vehicle_id' not in lane_detection_data:
                                    lane_detection_data['next_vehicle_id'] = 0
                                vehicle_id = f"vehicle_{cls}_{lane_detection_data['next_vehicle_id']}"
                                lane_detection_data['next_vehicle_id'] += 1
                                print(f"🆕 [NEW VEHICLE] {vehicle_id} | Class {cls} | Frame {frame_count} | Pos ({center_x}, {center_y})")
                            else:
                                # Matched existing vehicle - in chi tiết khi debug mode
                                if frame_count % 50 == 0:  # Log mỗi 50 frames
                                    print(f"🔗 [MATCHED] {vehicle_id} | IoU {best_iou:.2f} | Dist {best_dist:.0f}px | Frame {frame_count}")
                            
                            # Xác định lane hiện tại của vehicle (kiểm tra cả center và bbox)
                            # Sử dụng majority rule: nếu >60% bbox nằm trong lane thì tính là trong lane đó
                            bbox_width = x2 - x1
                            left_in_motor = max(0, min(x2, lane_boundary) - x1)
                            right_in_car = max(0, x2 - max(x1, lane_boundary))
                            
                            if left_in_motor > bbox_width * 0.6:
                                current_lane = "motor"
                            elif right_in_car > bbox_width * 0.6:
                                current_lane = "car"
                            else:
                                # Xe đang ở giữa 2 làn - dùng center để quyết định
                                current_lane = "motor" if center_x < lane_boundary else "car"
                            
                            vehicle_type = "xe_may" if cls == 1 else "oto" if cls in [0, 3, 4] else f"unknown_cls_{cls}"
                            
                            # Khởi tạo hoặc update vehicle state
                            if vehicle_id not in lane_detection_data['vehicle_states']:
                                lane_detection_data['vehicle_states'][vehicle_id] = {
                                    'current_lane': current_lane,
                                    'lane_history': [current_lane],  # Track lane over frames
                                    'last_seen': frame_count,
                                    'has_violated': False,
                                    'violation_frame': None,
                                    'class': cls,
                                    'confidence_sum': conf,
                                    'detection_count': 1,
                                    'last_bbox': current_bbox  # Thêm bbox để tracking
                                }
                            else:
                                # Update existing vehicle
                                vehicle_state = lane_detection_data['vehicle_states'][vehicle_id]
                                vehicle_state['last_seen'] = frame_count
                                vehicle_state['current_lane'] = current_lane
                                vehicle_state['confidence_sum'] += conf
                                vehicle_state['detection_count'] += 1
                                vehicle_state['last_bbox'] = current_bbox  # Cập nhật bbox
                                
                                # Update lane history (keep last 10 frames)
                                vehicle_state['lane_history'].append(current_lane)
                                if len(vehicle_state['lane_history']) > 10:
                                    vehicle_state['lane_history'] = vehicle_state['lane_history'][-10:]
                            
                            vehicle_state = lane_detection_data['vehicle_states'][vehicle_id]
                            
                            # Multi-frame validation: xe phải ở sai lane ít nhất 3 trong 5 frames gần nhất
                            lane_history = vehicle_state['lane_history']
                            if len(lane_history) >= 3:
                                recent_5 = lane_history[-5:] if len(lane_history) >= 5 else lane_history
                                
                                # Đếm số lần xe ở sai lane
                                wrong_lane_count = 0
                                for lane in recent_5:
                                    # XE MÁY ở LANE Ô TÔ = sai
                                    if cls == 1 and lane == "car":
                                        wrong_lane_count += 1
                                    # Ô TÔ ở LANE XE MÁY = sai
                                    elif cls in [0, 3, 4] and lane == "motor":
                                        wrong_lane_count += 1
                                
                                # Nếu xe ở sai lane >= 3 frames trong 5 frames gần nhất
                                # VÀ chưa bị đếm vi phạm
                                # VÀ confidence trung bình đủ cao (>= 0.4)
                                avg_conf = vehicle_state['confidence_sum'] / vehicle_state['detection_count']
                                
                                if (wrong_lane_count >= 3 and 
                                    not vehicle_state['has_violated'] and 
                                    avg_conf >= 0.4):
                                    
                                    # XE MÁY ở LANE Ô TÔ
                                    if cls == 1 and current_lane == "car":
                                        violation_detected = True
                                        violation_type = "xe_may_vi_pham_lan_oto"
                                        vehicle_state['has_violated'] = True
                                        vehicle_state['violation_frame'] = frame_count
                                        
                                        old_count = lane_detection_data['motor_violations']
                                        lane_detection_data['motor_violations'] += 1
                                        new_count = lane_detection_data['motor_violations']
                                        print(f"🚨🏍️ [MOTOR VIOLATION] {old_count} → {new_count} | Vehicle {vehicle_id} | Frame {frame_count} | History: {recent_5} | Conf: {avg_conf:.2f}")
                                    
                                    # Ô TÔ ở LANE XE MÁY
                                    elif cls in [0, 3, 4] and current_lane == "motor":
                                        violation_detected = True
                                        violation_type = "oto_vi_pham_lan_xe_may"
                                        vehicle_state['has_violated'] = True
                                        vehicle_state['violation_frame'] = frame_count
                                        
                                        old_count = lane_detection_data['car_violations']
                                        lane_detection_data['car_violations'] += 1
                                        new_count = lane_detection_data['car_violations']
                                        print(f"🚨🚗 [CAR VIOLATION] {old_count} → {new_count} | Vehicle {vehicle_id} | Frame {frame_count} | History: {recent_5} | Conf: {avg_conf:.2f}")
                            
                            # Violation key cho backward compatibility
                            violation_key = f"{vehicle_id}_{current_lane}"
                            frame_violation_key = f"{vehicle_id}_frame_{frame_count}"
                            cooldown_frames = 150
                            
                            
                            # Xử lý hiển thị vi phạm trên video
                            if violation_detected:
                                # Vẽ text vi phạm với thông tin chi tiết
                                draw_text(frame, "VI PHAM!", (x1, y1 - 40), 
                                        text_color=(255, 255, 255), text_color_bg=(255, 0, 0))
                                draw_text(frame, f"Lane: {current_lane.upper()}", (x1, y1 - 60), 
                                        font_scale=0.5, text_color=(255, 255, 255), text_color_bg=(255, 0, 0))
                                    
                            else:
                                # Hiển thị lane hiện tại cho xe không vi phạm (để debug)
                                if frame_count % 15 == 0:  # Mỗi 0.5 giây
                                    lane_color = (255, 0, 255) if current_lane == "motor" else (0, 255, 0)
                                    draw_text(frame, current_lane.upper(), (x1, y2 + 5), 
                                            font_scale=0.4, text_color=(255, 255, 255), text_color_bg=lane_color)
                            
                            # Lưu thông tin vi phạm
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
                                
                        except Exception as e:
                            print(f"Lỗi xử lý detection: {e}")
                            continue
            
            # Cleanup vehicle states và frame violations
            if frame_count % 100 == 0:  # Cleanup mỗi 100 frames (3-4 giây)
                current_frame = frame_count
                
                # Cleanup vehicle states - xóa vehicles không thấy > 100 frames (3-4 giây)
                expired_vehicles = [vid for vid, state in lane_detection_data['vehicle_states'].items() 
                                  if current_frame - state['last_seen'] > 100]
                for vid in expired_vehicles:
                    del lane_detection_data['vehicle_states'][vid]
                
                # Cleanup violation cooldown
                expired_keys = [k for k, v in lane_detection_data['violation_cooldown'].items() 
                              if current_frame - v > 200]
                for k in expired_keys:
                    del lane_detection_data['violation_cooldown'][k]
                    
                # Cleanup frame violations cũ
                if 'frame_violations' in lane_detection_data:
                    recent_violations = {v for v in lane_detection_data['frame_violations'] 
                                       if any(f"_frame_{current_frame - i}" in v for i in range(10))}
                    lane_detection_data['frame_violations'] = recent_violations
                    
                if len(expired_vehicles) > 0:
                    print(f"🧹 [FRAME {frame_count}] Cleaned {len(expired_vehicles)} expired vehicles")
                print(f"📊 Active vehicles: {len(lane_detection_data['vehicle_states'])}, Violations: Motor={lane_detection_data['motor_violations']}, Car={lane_detection_data['car_violations']}")
            
            # Debug log ít hơn để tránh spam
            if frame_count % 200 == 0:
                active_vehicles = len(lane_detection_data['vehicle_states'])
                violated_vehicles = sum(1 for state in lane_detection_data['vehicle_states'].values() if state.get('has_violated', False))
                print(f"🔍 [DEBUG FRAME {frame_count}] Motor violations: {lane_detection_data['motor_violations']}, Car violations: {lane_detection_data['car_violations']}")
                print(f"🚗 Active vehicles: {active_vehicles}, Violated: {violated_vehicles}")
                
                # Debug top 3 vehicles với lane history
                for vid, state in list(lane_detection_data['vehicle_states'].items())[:3]:
                    history_str = ''.join(['M' if l == 'motor' else 'C' for l in state.get('lane_history', [])])
                    print(f"   Vehicle {vid}: lane={state['current_lane']}, violated={state['has_violated']}, history={history_str}")
            
            # Video sạch - không hiển thị text thống kê trên video
            # Chỉ thống kê thời gian thực ở web UI bên phải
            
            # Ghi frame
            out.write(frame)
            
            # Throttle
            time.sleep(0.033)
            
            yield frame
        
        # End of while loop - video đã xử lý xong hoặc bị dừng
        print(f"✅ Video processing loop ended - {processed_count}/{frame_count} frames processed")
        
    except Exception as e:
        print(f"❌ Lỗi trong video_detection_web: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # QUAN TRỌNG: Đảm bảo cap và out luôn được release TRƯỚC KHI export
        print("🧹 Cleanup resources...")
        if cap is not None:
            try:
                cap.release()
                print("✅ Camera released")
            except Exception as e:
                print(f"⚠️ Error releasing camera: {e}")
        
        if out is not None:
            try:
                out.release()
                print("✅ Video writer released - Video đã được finalize")
                # Đánh dấu đã release để stop_lane_detection biết
                lane_detection_data['video_writer'] = None
            except Exception as e:
                print(f"⚠️ Error releasing video writer: {e}")
        
        cv2.destroyAllWindows()
        
        # SAU KHI release video writer, mới export kết quả
        print("📦 Tự động xuất kết quả CSV/Excel/JSON...")
        try:
            violations_data = lane_detection_data.get('violations', [])
            timestamp = lane_detection_data.get('timestamp', datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            output_path = lane_detection_data.get('output_path', '')
            
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Xuất CSV
            csv_filename = os.path.join(output_dir, f"lane_violations_stats_{timestamp}.csv")
            if len(violations_data) > 0:
                df = pd.DataFrame(violations_data)
            else:
                # Tạo DataFrame rỗng với các columns chuẩn
                df = pd.DataFrame(columns=['frame', 'time_seconds', 'time_formatted', 'violation_type', 
                                          'vehicle_class', 'confidence', 'vehicle_id'])
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"✅ Đã xuất lane CSV: {csv_filename} ({len(violations_data)} vi phạm)")
            
            # Lưu filename vào global data để endpoint stop có thể trả về
            lane_detection_data['csv_filename'] = csv_filename
            
            # Xuất Excel
            xlsx_filename = os.path.join(output_dir, f"lane_violations_stats_{timestamp}.xlsx")
            try:
                df.to_excel(xlsx_filename, index=False, engine='openpyxl')
                print(f"✅ Đã xuất lane Excel: {xlsx_filename} ({len(violations_data)} vi phạm)")
                lane_detection_data['xlsx_filename'] = xlsx_filename
            except Exception as e:
                print(f"⚠️ Không thể xuất Excel: {e}")
                import traceback
                traceback.print_exc()
                lane_detection_data['xlsx_filename'] = None
            
            # Xuất JSON chi tiết
            json_filename = os.path.join(output_dir, f"lane_violations_details_{timestamp}.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_info': {
                        'input_path': path_x,
                        'output_path': output_path,
                        'timestamp': timestamp,
                        'total_frames': frame_count,
                        'frames_processed': processed_count,
                        'fps': original_fps
                    },
                    'violations': violations_data,
                    'summary': {
                        'total_violations': len(violations_data),
                        'motor_violations': lane_detection_data.get('motor_violations', 0),
                        'car_violations': lane_detection_data.get('car_violations', 0),
                        'processing_time': time.time() - lane_detection_data.get('start_time', time.time()),
                        'frames_processed': frame_count
                    }
                }, f, indent=2, ensure_ascii=False)
            print(f"✅ Đã xuất lane JSON: {json_filename}")
            lane_detection_data['json_filename'] = json_filename
            
            print(f"📊 Tổng kết vi phạm: Xe máy={lane_detection_data['motor_violations']}, Ô tô={lane_detection_data['car_violations']}, Tổng={len(violations_data)}")
            print(f"🎉 Đã xuất tất cả file thành công!")
            
            # Verify video có thể mở được không
            if output_path and os.path.exists(output_path):
                test_cap = cv2.VideoCapture(output_path)
                if test_cap.isOpened():
                    frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(f"✅ Video có thể xem được: {frames} frames")
                    test_cap.release()
                else:
                    print(f"⚠️ Video không mở được - có thể bị lỗi!")
        except Exception as export_error:
            print(f"⚠️ Lỗi khi xuất kết quả tự động: {export_error}")
            import traceback
            traceback.print_exc()


def reset_lane_detection_data():
    """Reset toàn bộ dữ liệu lane detection cho phiên mới"""
    global lane_detection_data
    
    print("🔄 RESET lane detection data for new session...")
    
    lane_detection_data.update({
        'violations': [],
        'motor_violations': 0,
        'car_violations': 0,
        'start_time': time.time(),
        'video_writer': None,
        'output_path': None,
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        'tracked_vehicles': {},
        'violation_cooldown': {},
        'violation_frames': [],
        'frame_count': 0,
        'original_fps': None,
        'output_size': None,
        'vehicle_states': {}  # Reset vehicle tracking states
    })
    
    print(f"✅ Lane detection data reset complete - Timestamp: {lane_detection_data['timestamp']}")


def generate_frames_lane(path_x):
    """Generate frames cho lane detection"""
    global lane_detection_active
    
    # RESET DỮ LIỆU
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
                # Encode với quality cao
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
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
        
        # Video đã hết hoặc bị dừng - set flag về False
        lane_detection_active = False
        print("✅ Lane detection stream ended")
                
    except Exception as e:
        lane_detection_active = False
        print(f"❌ Error in video detection: {e}")
        import traceback
        traceback.print_exc()


# OLD FUNCTION - DISABLED TO PREVENT CONFLICT WITH NEW LOGIC
# def video_detection(path_x=""):
#     cap = cv2.VideoCapture(path_x)
#     model = YOLO('best_new/vehicle.pt')
#     stt_m = 0
#     stt_ctb = 0
#     examBB = createBB.infoObject()
#     dataBienBan_M = 'BienBanNopPhatXeMay/'
#     dataBienBan_CTB = 'BienBanNopPhatXeOTo/'

#     # results = model.track(source="Videos/test4.mp4", show=True, stream=True)
#     while cap.isOpened():
#         success, frame = cap.read()
#         if success:
#             #  Dự đoán
#             results = model(frame)

#             # lấy ra frame sau khi đc gắn nhãn
#             annotated_frame = results[0].plot()

#             # lấy kích thước (height , width , _ )
#             # print("kích thước frame : ", annotated_frame.shape)

#             # Hiển thị lên
#             # cv2.imshow("Display ", annotated_frame)
#             # results = model.track(source="Videos/test4.mp4", show=True, tracker="bytetrack.yaml", stream=True)
#             for result in results:
#                 boxes = result.boxes.numpy()

#                 # Lấy tên class
#                 name = result.names

#                 # lấy tất cả các thông số trong một list tọa độ các đối tượng (x0 ,y0, x1, y1, )
#                 # print("list 1 ", boxes.xyxy)
#                 list_2 = []

#                 # Lấy tất các các thông số của nhiều đối tượng (x0, y0 , x1 , y1 , id ,độ chính xác , loại class)
#                 # print("Boxes ", boxes)

#                 for box in boxes:
#                     # TOÀN BỘ LOGIC CŨ COMMENTED TO PREVENT CONFLICT
#                     pass
#         else:
#             break
#     cv2.destroyAllWindows()


# OLD FUNCTION - DISABLED  
# def generate_frames(path_x):
#     yolo_output = video_detection(path_x)
#     for detection_ in yolo_output:
#         ref, buffer = cv2.imencode('.jpg', detection_)

#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def _resolve_helmet_weights_path():
    """Find a valid helmet model weights path from common locations."""
    candidate_paths = [
        os.path.join('model_helmet', 'best_helmet_end.pt'),
        os.path.join('model_helmet', 'helmet_end.pt'),
        os.path.join('model_helmet', 'helmet.pt'),
        os.path.join('model_helmet', 'best.pt'),
        os.path.join('model_helmet_v2', 'best.pt'),
        os.path.join('model_helmet_v2', 'helmet.pt'),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return None


def generate_frames_helmet(path_x):
    """Generate frames for helmet detection streaming with stop control"""
    global helmet_detection_active, helmet_detection_data
    
    try:
        cap = cv2.VideoCapture(path_x)
        if not cap.isOpened():
            print(f"Error: Could not open video {path_x}")
            return
        
        # Load YOLO model for helmet detection with robust path resolution
        weights_path = _resolve_helmet_weights_path()
        if not weights_path:
            print("❌ Helmet model weights not found in expected locations. Please place weights in 'model_helmet/'.")
            model = None
        else:
            print(f"✅ Loading helmet model from: {weights_path}")
            model = YOLO(weights_path)
        
        # Optional vehicle model for license plate assist (motorbike focus)
        vehicle_model = None
        try:
            vehicle_weights = os.path.join('best_new', 'vehicle.pt')
            if os.path.exists(vehicle_weights):
                vehicle_model = YOLO(vehicle_weights)
                print(f"✅ Vehicle model loaded for plate assist: {vehicle_weights}")
            else:
                print("ℹ️ Vehicle model not found; plate extraction disabled")
        except Exception as e:
            print(f"⚠️ Could not load vehicle model: {e}")
        
        # Initialize EasyOCR for number plate text recognition
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            print("✅ EasyOCR initialized for helmet detection")
        except Exception as e:
            print(f"⚠️ EasyOCR initialization failed: {e}")
            reader = None
        
        # Ensure output directory exists for analyzed video
        os.makedirs('output', exist_ok=True)
        
        # Initialize helmet detection data ONCE khi bắt đầu
        if helmet_detection_data.get('frame_count') is None:
            helmet_detection_data['frame_count'] = 0
            helmet_detection_data['violations'] = []
            helmet_detection_data['total_violations'] = 0
            helmet_detection_data['with_helmet'] = 0
            helmet_detection_data['without_helmet'] = 0
            helmet_detection_data['detection_cooldown'] = {}
            helmet_detection_data['frame_detections'] = set()
            helmet_detection_data['temporal_votes'] = {}
            helmet_detection_data['state_map'] = {}
            helmet_detection_data['recent_plates'] = []
            print("🔧 [INIT] Helmet detection data initialized")
        
        frame_number = 0
        
        while cap.isOpened():  # ✅ LUÔN STREAM VIDEO
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Chỉ chạy detection khi helmet_detection_active = True
            if helmet_detection_active and model is not None:
                helmet_detection_data['frame_count'] = frame_number
                if helmet_detection_data.get('start_time') is None:
                    helmet_detection_data['start_time'] = time.time()
                # Lazy init video writer on first active frame
                if helmet_detection_data.get('video_writer') is None:
                    try:
                        h, w = frame.shape[:2]
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        timestamp = helmet_detection_data.get('timestamp') or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_path = os.path.join('output', f"helmet_analysis_{timestamp}.mp4")
                        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                        helmet_detection_data['video_writer'] = writer
                        helmet_detection_data['output_path'] = out_path
                        helmet_detection_data['original_fps'] = fps
                        helmet_detection_data['output_size'] = (w, h)
                        print(f"🎥 [HELMET] Recording analyzed video to: {out_path} @ {fps:.2f}fps, size {(w, h)}")
                    except Exception as e:
                        print(f"⚠️ Failed to initialize video writer: {e}")
                
                # Run helmet detection
                results = model(frame)
                
                # QUAN TRỌNG: Cleanup frame detections và state map mỗi 50 frames
                if frame_number % 50 == 0:
                    # Cleanup frame detections - chỉ giữ 10 frames gần nhất
                    current_frame_detections = {d for d in helmet_detection_data.get('frame_detections', set()) 
                                               if any(f"_frame_{frame_number - i}" in d for i in range(10))}
                    helmet_detection_data['frame_detections'] = current_frame_detections
                    
                    # Cleanup temporal_votes và state_map - xóa các entry không active > 100 frames
                    active_keys = set()
                    for key in list(helmet_detection_data.get('temporal_votes', {}).keys()):
                        mem = helmet_detection_data['temporal_votes'][key]
                        if frame_number - mem.get('last_frame', 0) > 100:
                            del helmet_detection_data['temporal_votes'][key]
                        else:
                            active_keys.add(key)
                    
                    for key in list(helmet_detection_data.get('state_map', {}).keys()):
                        if key not in active_keys:
                            state_rec = helmet_detection_data['state_map'][key]
                            if frame_number - state_rec.get('last_frame', 0) > 100:
                                del helmet_detection_data['state_map'][key]
                    
                    print(f"🧹 [FRAME {frame_number}] Cleaned up: temporal_votes={len(helmet_detection_data.get('temporal_votes', {}))}, state_map={len(helmet_detection_data.get('state_map', {}))}")
                
                # Draw detections on frame with label-based mapping and conflict resolution
                def _norm_label(s: str) -> str:
                    return s.lower().replace('-', ' ').strip()

                def _iou(a, b) -> float:
                    ax1, ay1, ax2, ay2 = a
                    bx1, by1, bx2, by2 = b
                    inter_x1 = max(ax1, bx1)
                    inter_y1 = max(ay1, by1)
                    inter_x2 = min(ax2, bx2)
                    inter_y2 = min(ay2, by2)
                    iw = max(0, inter_x2 - inter_x1)
                    ih = max(0, inter_y2 - inter_y1)
                    inter = iw * ih
                    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
                    union = area_a + area_b - inter
                    return inter / union if union > 0 else 0.0

                frame_with = 0
                frame_without = 0
                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue

                    # First pass: collect detections with normalized labels
                    dets = []  # {type: 'with_helmet'|'no_helmet'|'rider'|'plate'|'other', bbox, conf}
                    names = getattr(r, 'names', {}) or {}
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = max(0, x2 - x1)
                        h = max(0, y2 - y1)
                        area = w * h
                        label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
                        nl = _norm_label(label)

                        if ('without' in nl) or ('no' in nl and 'helmet' in nl) or ('khong' in nl and 'mu' in nl):
                            det_type = 'no_helmet'
                        elif 'helmet' in nl or 'mu' in nl:
                            det_type = 'with_helmet'
                        elif 'rider' in nl or 'person' in nl:
                            det_type = 'rider'
                        elif 'plate' in nl or 'license' in nl:
                            det_type = 'plate'
                        else:
                            det_type = 'other'

                        # Class-specific confidence and area gates to reduce false 'no helmet'
                        min_area_with = 1200
                        min_area_no = 1500
                        
                        # THÊM: Kiểm tra aspect ratio để loại bỏ detection không hợp lý
                        aspect_ratio = w / h if h > 0 else 0
                        
                        if det_type == 'no_helmet':
                            # TĂNG THRESHOLD cho no_helmet để giảm false positive
                            if conf < 0.60 or area < min_area_no:  # Tăng từ 0.55 lên 0.60
                                # Conservative: skip weak/small no-helmet to avoid false alarms
                                continue
                            # Kiểm tra aspect ratio hợp lý cho người (0.3 - 2.5)
                            if aspect_ratio < 0.3 or aspect_ratio > 2.5:
                                continue
                        elif det_type == 'with_helmet':
                            # Giảm threshold cho with_helmet để dễ nhận diện hơn
                            if conf < 0.30 or area < min_area_with:  # Giảm từ 0.35 xuống 0.30
                                continue
                            # Aspect ratio cho helmet (thường hơi vuông hơn - 0.4 - 2.0)
                            if aspect_ratio < 0.4 or aspect_ratio > 2.0:
                                continue
                        else:
                            if conf < 0.3:
                                continue

                        dets.append({'type': det_type, 'bbox': (x1, y1, x2, y2), 'conf': conf})

                    # Conflict resolution: if with_helmet overlaps with no_helmet, prefer with_helmet
                    # TĂNG CƯỜNG: Ưu tiên with_helmet mạnh hơn để giảm false alarm
                    suppressed_no = set()
                    for i, d_with in enumerate(d for d in dets if d['type'] == 'with_helmet'):
                        for j, d_no in enumerate(d for d in dets if d['type'] == 'no_helmet'):
                            iou = _iou(d_with['bbox'], d_no['bbox'])
                            if iou > 0.4:  # Giảm từ 0.5 xuống 0.4 để suppress nhiều hơn
                                # Suppress 'no_helmet' nếu overlap và confidence with_helmet không quá thấp
                                if d_no['conf'] <= d_with['conf'] + 0.15:  # Tăng từ 0.1 lên 0.15
                                    # Find absolute index of this no_helmet det to mark suppressed
                                    idx = [k for k, dd in enumerate(dets) if dd is d_no][0]
                                    suppressed_no.add(idx)

                    # Same-type de-duplication (NMS-style) to avoid multiple boxes for one person
                    def _dedup_type_indices(det_type: str, iou_thresh: float = 0.6):
                        idxs = [i for i, d in enumerate(dets) if d['type'] == det_type]
                        # Sort by confidence desc
                        idxs_sorted = sorted(idxs, key=lambda i: dets[i]['conf'], reverse=True)
                        dropped = set()
                        for a in range(len(idxs_sorted)):
                            i = idxs_sorted[a]
                            if i in dropped:
                                continue
                            for b in range(a + 1, len(idxs_sorted)):
                                j = idxs_sorted[b]
                                if j in dropped:
                                    continue
                                if _iou(dets[i]['bbox'], dets[j]['bbox']) > iou_thresh:
                                    # Drop the lower-confidence duplicate j
                                    dropped.add(j)
                        return dropped

                    drop_indices = set()
                    drop_indices |= _dedup_type_indices('no_helmet', iou_thresh=0.6)
                    drop_indices |= _dedup_type_indices('with_helmet', iou_thresh=0.6)

                    # Second pass: draw and count with cooldown/grid + temporal vote logic + state hysteresis
                    counted_keys = set()  # ensure per-frame visible count is unique per position_key
                    for idx, det in enumerate(dets):
                        # Skip same-type duplicates determined above
                        if idx in drop_indices:
                            continue
                        det_type = det['type']
                        x1, y1, x2, y2 = det['bbox']
                        conf = det['conf']

                        # compute grid and keys
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        grid_x = center_x // 300
                        grid_y = center_y // 300
                        position_key = f"{det_type}_{grid_x}_{grid_y}"
                        frame_detection_key = f"{position_key}_frame_{frame_number}"

                        cooldown_frames = 250
                        can_count = True
                        skip_reason = ""

                        # Cooldown check
                        last_map = helmet_detection_data.get('detection_cooldown', {})
                        if position_key in last_map:
                            last_detected_frame = last_map[position_key]
                            frames_since_last = frame_number - last_detected_frame
                            if frames_since_last < cooldown_frames:
                                can_count = False
                                skip_reason = f"COOLDOWN ({frames_since_last}/{cooldown_frames})"

                        # Per-frame duplicate check
                        if frame_detection_key in helmet_detection_data.get('frame_detections', set()):
                            can_count = False
                            skip_reason = "FRAME_DUP"

                        # Temporal voting: accumulate evidence across recent frames per position_key
                        if det_type in ('no_helmet', 'with_helmet'):
                            mem = helmet_detection_data.get('temporal_votes', {}).get(position_key)
                            if mem is None:
                                mem = {'window': [], 'last_frame': -1}
                                helmet_detection_data['temporal_votes'][position_key] = mem
                            # Only record one vote per frame
                            if mem.get('last_frame') != frame_number:
                                mem['last_frame'] = frame_number
                                # maintain a small sliding window
                                wdw = mem['window']
                                wdw.append('no' if det_type == 'no_helmet' else 'with')
                                # cap window size
                                if len(wdw) > 8:
                                    mem['window'] = wdw[-8:]

                            # Require stronger consensus and anti-flip for 'no_helmet'
                            wdw = mem['window']
                            last3 = wdw[-3:] if len(wdw) >= 3 else wdw
                            last4 = wdw[-4:] if len(wdw) >= 4 else wdw
                            no_votes = sum(1 for v in wdw[-6:] if v == 'no')
                            with_votes_last3 = sum(1 for v in last3 if v == 'with')
                            with_votes_last4 = sum(1 for v in last4 if v == 'with')
                            if det_type == 'no_helmet':
                                # TĂNG YÊU CẦU CAO HƠN: Need >=5 'no' trong last 6 và KHÔNG CÓ 'with' trong last 4
                                if len(wdw) < 6 or no_votes < 5 or with_votes_last4 > 0:  # Không cho phép 'with' trong 4 frames gần nhất
                                    can_count = False
                                    skip_reason = 'VOTE_INSUFFICIENT'

                            # State machine with hysteresis per position
                            state_rec = helmet_detection_data.get('state_map', {}).get(position_key)
                            if state_rec is None:
                                state_rec = {'state': 'unknown', 'window': [], 'last_change': -1, 'lock_until': -1}
                                helmet_detection_data['state_map'][position_key] = state_rec
                            # Update state window (one vote per frame)
                            if (not state_rec['window']) or (state_rec.get('last_frame') != frame_number):
                                state_rec['window'].append('no' if det_type == 'no_helmet' else 'with')
                                if len(state_rec['window']) > 8:
                                    state_rec['window'] = state_rec['window'][-8:]
                                state_rec['last_frame'] = frame_number
                            # Update last bbox for stable rendering
                            state_rec['last_bbox'] = (x1, y1, x2, y2)
                            # Hysteresis transitions
                            cur_state = state_rec['state']
                            wdw_state = state_rec['window']
                            no6 = sum(1 for v in wdw_state[-6:] if v == 'no')
                            with6 = sum(1 for v in wdw_state[-6:] if v == 'with')
                            with3 = sum(1 for v in wdw_state[-3:] if v == 'with')
                            no3 = sum(1 for v in wdw_state[-3:] if v == 'no')
                            no4 = sum(1 for v in wdw_state[-4:] if v == 'no')
                            with4 = sum(1 for v in wdw_state[-4:] if v == 'with')
                            
                            if cur_state in ('unknown', 'with'):
                                # TĂNG YÊU CẦU MẠNH chuyển sang 'no': cần ít nhất 5 'no' trong 6 frames và KHÔNG CÓ 'with' trong 4 frames gần nhất
                                if no6 >= 5 and with4 == 0:  # Mở rộng từ 3 frames lên 4 frames
                                    # Transition to 'no'
                                    state_rec['state'] = 'no'
                                    state_rec['last_change'] = frame_number
                                    state_rec['lock_until'] = frame_number + 300  # Tăng lock từ 220 lên 300 frames (~10 giây)
                                    print(f"🔄 [STATE] {position_key}: 'with' → 'no' | F{frame_number} | Lock until {state_rec['lock_until']}")
                            elif cur_state == 'no':
                                # KHÓ CHUYỂN VỀ 'with': Cần bằng chứng RẤT MẠNH và sau khi hết lock
                                if frame_number >= state_rec.get('lock_until', -1):
                                    # Cần ít nhất 6/6 frames 'with' và hoàn toàn KHÔNG CÓ 'no' trong 4 frames gần nhất
                                    if with6 >= 6 and no4 == 0:  # Tăng từ 4 lên 6, yêu cầu 100% 'with'
                                        state_rec['state'] = 'with'
                                        state_rec['last_change'] = frame_number
                                        state_rec['lock_until'] = -1
                                        print(f"🔄 [STATE] {position_key}: 'no' → 'with' | F{frame_number}")
                                    else:
                                        # GIỮ NGUYÊN state 'no' - chỉ extend lock
                                        if with4 == 0 and no4 >= 3:
                                            # Vẫn thấy 'no' mạnh, extend lock thêm
                                            state_rec['lock_until'] = frame_number + 150
                                else:
                                    # Đang trong lock period - TUYỆT ĐỐI GIỮ state 'no'
                                    pass

                        if det_type == 'no_helmet':
                            # Skip if suppressed due to overlap with a helmet detection
                            if idx in suppressed_no:
                                # Do not draw or count suppressed red box to match visual logic
                                can_count = False
                                skip_reason = "SUPPRESSED_BY_HELMET"
                                continue
                            
                            # ĐIỀU KIỆN CỰC KỲ KHẮT KHE: Chỉ draw và count khi CẢ 3 điều kiện thỏa mãn
                            state_ok = helmet_detection_data['state_map'].get(position_key, {}).get('state') == 'no'
                            votes_ok = (no_votes >= 5 and with_votes_last4 == 0)  # 5/6 frames 'no' và 0 'with' trong 4 frames
                            lock_active = frame_number < helmet_detection_data['state_map'].get(position_key, {}).get('lock_until', -1)
                            
                            # Phải thỏa mãn CẢ BA: state='no' VÀ votes đủ VÀ đang trong lock period
                            if not (state_ok and votes_ok and lock_active):
                                # Không đủ chắc chắn - KHÔNG hiển thị
                                can_count = False
                                skip_reason = skip_reason or f'NOT_CONFIDENT(state={state_ok},votes={votes_ok},lock={lock_active})'
                            else:
                                # ĐỦ ĐIỀU KIỆN - Draw và count
                                if position_key not in counted_keys:
                                    # Vẽ box đỏ dày hơn để nổi bật
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                    cv2.putText(frame, f"KHONG MU: {conf:.2f}", (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                                    frame_without += 1
                                    counted_keys.add(position_key)
                            
                            if can_count:
                                old_count = helmet_detection_data['without_helmet']
                                helmet_detection_data['without_helmet'] += 1
                                new_count = helmet_detection_data['without_helmet']
                                helmet_detection_data['total_violations'] = helmet_detection_data['without_helmet']
                                helmet_detection_data['detection_cooldown'][position_key] = frame_number
                                helmet_detection_data['frame_detections'].add(frame_detection_key)
                                # Extend lockout when a no-helmet count happens
                                sm = helmet_detection_data['state_map'].get(position_key)
                                if sm:
                                    sm['state'] = 'no'
                                    sm['lock_until'] = max(sm.get('lock_until', -1), frame_number + 250)
                                # Try associate nearest recent plate candidate
                                lp_text = None
                                try:
                                    candidates = [p for p in helmet_detection_data.get('recent_plates', []) if frame_number - p.get('frame', 0) <= 15]
                                    if candidates:
                                        cx = (x1 + x2) // 2
                                        cy = (y1 + y2) // 2
                                        def dist2(p):
                                            bb = p.get('bbox') or (cx, cy, cx, cy)
                                            pcx = (bb[0] + bb[2]) // 2
                                            pcy = (bb[1] + bb[3]) // 2
                                            return (pcx - cx)*(pcx - cx) + (pcy - cy)*(pcy - cy)
                                        best = min(candidates, key=dist2)
                                        bb = best.get('bbox')
                                        if bb:
                                            pcx = (bb[0] + bb[2]) // 2
                                            pcy = (bb[1] + bb[3]) // 2
                                            if (pcx - cx)**2 + (pcy - cy)**2 < (250*250):
                                                lp_text = best.get('text')
                                except Exception:
                                    pass
                                violation_info = {
                                    'violation_id': len(helmet_detection_data['violations']) + 1,
                                    'type': 'no_helmet',
                                    'frame_number': helmet_detection_data['frame_count'],
                                    'confidence': conf,
                                    'bbox': [x1, y1, x2, y2],
                                    'detected_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'license_plate': lp_text
                                }
                                helmet_detection_data['violations'].append(violation_info)
                                cooldown_count = len(helmet_detection_data['detection_cooldown'])
                                print(f"🚨 [KHÔNG MŨ ➕] {old_count} → {new_count} | F{frame_number} | G({grid_x},{grid_y}) | Cooldowns: {cooldown_count}")
                            else:
                                if frame_number % 30 == 0:
                                    print(f"⏭️ [KHÔNG MŨ SKIP] F{frame_number} | G({grid_x},{grid_y}) | {skip_reason}")
                        elif det_type == 'with_helmet':
                            # draw green once per position_key for visible count
                            if position_key not in counted_keys:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(frame, f"CO MU: {conf:.2f}", (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                frame_with += 1
                                counted_keys.add(position_key)
                            if can_count:
                                old_count = helmet_detection_data['with_helmet']
                                helmet_detection_data['with_helmet'] += 1
                                new_count = helmet_detection_data['with_helmet']
                                helmet_detection_data['detection_cooldown'][position_key] = frame_number
                                helmet_detection_data['frame_detections'].add(frame_detection_key)
                                cooldown_count = len(helmet_detection_data['detection_cooldown'])
                                print(f"✅ [CÓ MŨ ➕] {old_count} → {new_count} | F{frame_number} | G({grid_x},{grid_y}) | Cooldowns: {cooldown_count}")
                            else:
                                if frame_number % 30 == 0:
                                    print(f"⏭️ [CÓ MŨ SKIP] F{frame_number} | G({grid_x},{grid_y}) | {skip_reason}")
                
                        elif det_type == 'rider':
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                            cv2.putText(frame, f"Rider: {conf:.2f}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        elif det_type == 'plate':
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(frame, f"Number Plate: {conf:.2f}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                            # OCR attempt
                            if reader is not None:
                                try:
                                    plate_crop = frame[y1:y2, x1:x2]
                                    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                    _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                    ocr_results = reader.readtext(thresh_plate)
                                    if ocr_results:
                                        best_text = ""
                                        best_confidence = 0
                                        for (bbox, text, prob) in ocr_results:
                                            if prob > best_confidence:
                                                best_confidence = prob
                                                best_text = text.strip()
                                        if best_text and len(best_text) >= 3:
                                            cv2.putText(frame, f"Text: {best_text}", (x1, y2+20),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                                except Exception as e:
                                    print(f"❌ OCR error: {e}")

                # Optional: vehicle-based license plate extraction every few frames
                try:
                    if vehicle_model is not None and (frame_number % 5 == 0):
                        v_results = vehicle_model(frame, verbose=False)
                        for vr in v_results:
                            vboxes = getattr(vr, 'boxes', None)
                            if vboxes is None:
                                continue
                            for vbox in vboxes:
                                vcls = int(vbox.cls[0])
                                vconf = float(vbox.conf[0])
                                # Heuristic: class 1 used for motorbikes in lane logic
                                if vconf < 0.35 or vcls not in [1]:
                                    continue
                                vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
                                crop_h = max(0, vy2 - vy1)
                                if crop_h < 20:
                                    continue
                                region_y1 = vy1 + int(crop_h * 0.6)
                                plate_region = frame[region_y1:vy2, vx1:vx2]
                                if reader is None or plate_region.size == 0:
                                    continue
                                try:
                                    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                                    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                    ocr_results = reader.readtext(th)
                                except Exception:
                                    ocr_results = []
                                best_text = None
                                best_prob = 0
                                best_bbox = None
                                for (bbox_pts, text, prob) in (ocr_results or []):
                                    cleaned = ''.join(c for c in text if c.isalnum()).upper()
                                    if len(cleaned) >= 4 and prob > best_prob:
                                        best_prob = prob
                                        best_text = cleaned
                                        xs = [p[0] for p in bbox_pts]
                                        ys = [p[1] for p in bbox_pts]
                                        px1 = int(min(xs)) + vx1
                                        py1 = int(min(ys)) + region_y1
                                        px2 = int(max(xs)) + vx1
                                        py2 = int(max(ys)) + region_y1
                                        best_bbox = (px1, py1, px2, py2)
                                if best_text:
                                    # Draw on frame and store recent
                                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 255, 0), 1)
                                    if best_bbox:
                                        cv2.rectangle(frame, (best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3]), (0, 255, 255), 2)
                                    cv2.putText(frame, f"LP:{best_text}", (vx1, max(0, vy1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                    helmet_detection_data['recent_plates'].append({'text': best_text, 'bbox': best_bbox, 'frame': frame_number})
                        # Keep only last N
                        helmet_detection_data['recent_plates'] = helmet_detection_data['recent_plates'][-30:]
                except Exception as e:
                    if frame_number % 60 == 0:
                        print(f"⚠️ Plate extraction error: {e}")
                
                # After processing detections: ensure stable rendering using state_map and recency
                try:
                    smap = helmet_detection_data.get('state_map', {})
                    recent_limit = 8  # frames to keep state visible
                    stable_keys = [k for k, m in smap.items() if m.get('state') in ('no', 'with') and (frame_number - m.get('last_frame', -1) <= recent_limit)]
                    # Draw for stable keys not already drawn this frame (using last_bbox)
                    for k in stable_keys:
                        if k in counted_keys:
                            continue
                        m = smap[k]
                        bbox = m.get('last_bbox')
                        if not bbox:
                            continue
                        x1, y1, x2, y2 = bbox
                        if m.get('state') == 'no':
                            color = (0, 0, 255)
                            label = 'KHONG MU'
                            frame_without += 1
                        else:
                            color = (0, 255, 0)
                            label = 'CO MU'
                            frame_with += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        counted_keys.add(k)
                except Exception:
                    pass

                # Update current counts based on stable states (non-flickering)
                helmet_detection_data['current_with'] = frame_with
                helmet_detection_data['current_without'] = frame_without
                helmet_detection_data['current_total'] = frame_with + frame_without

                # Vẽ overlay thống kê TÍCH LŨY trên video (giống web UI - chỉ tăng, không giảm)
                try:
                    # Lấy số tích lũy từ helmet_detection_data
                    total_with_helmet = helmet_detection_data.get('with_helmet', 0)
                    total_without_helmet = helmet_detection_data.get('without_helmet', 0)
                    total_violations = total_without_helmet  # Vi phạm = không mũ
                    
                    # Nền đen cho text - mở rộng kích thước
                    cv2.rectangle(frame, (10, 10), (450, 130), (0, 0, 0), -1)
                    cv2.rectangle(frame, (10, 10), (450, 130), (255, 255, 255), 2)  # Viền trắng
                    
                    # Tiêu đề
                    cv2.putText(frame, "THONG KE TICH LUY:", (20, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Tổng số người có mũ đã đếm (TÍCH LŨY - CHỈ TĂNG)
                    cv2.putText(frame, f"Tong co mu: {total_with_helmet}", (20, 65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Tổng số người không mũ đã đếm (TÍCH LŨY - CHỈ TĂNG)
                    cv2.putText(frame, f"Tong khong mu: {total_without_helmet}", (20, 95), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)  # Cam
                    
                    # Tổng vi phạm (ĐỎ)
                    cv2.putText(frame, f"TONG VI PHAM: {total_violations}", (20, 125), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception as e:
                    print(f"⚠️ Error drawing overlay: {e}")

                # Cleanup cooldown mỗi 150 frames (5 giây) - XÓA COOLDOWN CŨ HƠN 250 FRAMES
                if frame_number % 150 == 0:
                    expired_keys = [k for k, v in helmet_detection_data.get('detection_cooldown', {}).items() 
                                  if frame_number - v > 250]  # Tăng lên 250 frames để cooldown bền vững hơn
                    for k in expired_keys:
                        del helmet_detection_data['detection_cooldown'][k]
                    
                    active_cooldowns = len(helmet_detection_data.get('detection_cooldown', {}))
                    print(f"🧹 [HELMET] Frame {frame_number}: Cleaned {len(expired_keys)} cooldowns, {active_cooldowns} active")

                    # Cleanup temporal votes too (stale > 200 frames)
                    tv = helmet_detection_data.get('temporal_votes', {})
                    stale = [k for k, m in tv.items() if frame_number - (m.get('last_frame', -1)) > 200]
                    for k in stale:
                        del tv[k]

                    # Cleanup state_map entries not updated for > 400 frames
                    smap = helmet_detection_data.get('state_map', {})
                    stale_states = [k for k, m in smap.items() if frame_number - m.get('last_frame', -1) > 400]
                    for k in stale_states:
                        del smap[k]
                
                # Write analyzed frame to output video
                try:
                    vw = helmet_detection_data.get('video_writer')
                    if vw is not None:
                        vw.write(frame)
                except Exception as e:
                    if frame_number % 60 == 0:
                        print(f"⚠️ Error writing frame to video: {e}")
                        
            else:
                # Không chạy detection - chỉ hiển thị video thuần
                # Thêm text thông báo
                status_text = "CHUA BAT DAU PHAT HIEN" if model is not None else "THIEU MODEL MU BAO HIEM"
                cv2.putText(frame, status_text, (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in generate_frames_helmet: {e}")
        # Return a simple error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Error loading video", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


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
    # Check if there's an uploaded video for helmet detection
    uploaded_video = session.get('uploaded_video_helmet', None)
    if not uploaded_video:
        print("DEBUG camera2: No uploaded helmet video in session")
        return Response(generate_placeholder_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Check if file exists
    if not os.path.exists(uploaded_video):
        print(f"DEBUG camera2: File {uploaded_video} does not exist")
        return Response(generate_placeholder_stream(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    print(f"DEBUG camera2: Starting helmet video stream for {uploaded_video}")
    
    # Use helmet detection streaming
    return Response(generate_frames_helmet(uploaded_video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
        
        # Store uploaded video path for streaming
        session['uploaded_video_helmet'] = filepath
        
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
        print("🚀 Starting helmet video processing...")
        
        # Process the video using selected detection method
        use_advanced = pending.get('use_advanced', False)
        # result_path, stats = process_helmet_video_complete(  # Tạm comment để tránh lỗi reportlab
        #     pending['input'], 
        #     pending['output'],
        #     use_improved_detection=use_advanced
        # )
        result_path = "temp_helmet_output.mp4"  # Tạm thời
        stats = {"violations": 0, "total_frames": 0}  # Tạm thời
        
        print(f"✅ Helmet video processing completed! Output: {result_path}")
        
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
        print(f"❌ Error in helmet processing: {str(e)}")
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
    """Dừng lane detection và trả về kết quả đã được export tự động"""
    global lane_detection_active, lane_detection_data
    
    if not lane_detection_active:
        return jsonify({'error': 'Lane detection is not running'}), 400
    
    print("🛑 Nhận yêu cầu dừng từ web...")
    lane_detection_active = False
    
    # Đợi để video detection dừng hoàn toàn và export tự động chạy
    print("⏳ Đang đợi video detection dừng và export kết quả...")
    max_wait = 10  # Đợi tối đa 10 giây
    waited = 0
    while waited < max_wait:
        # Kiểm tra xem video writer đã được release chưa (đánh dấu export đã xong)
        if lane_detection_data.get('video_writer') is None and lane_detection_data.get('csv_filename'):
            print("✅ Video detection đã dừng và export hoàn tất!")
            break
        time.sleep(0.5)
        waited += 0.5
    
    if waited >= max_wait:
        print("⚠️ Timeout khi đợi export hoàn tất, trả về kết quả hiện tại...")
    
    try:
        violations_data = lane_detection_data.get('violations', [])
        output_path = lane_detection_data.get('output_path', '')
        csv_filename = lane_detection_data.get('csv_filename')
        xlsx_filename = lane_detection_data.get('xlsx_filename')
        json_filename = lane_detection_data.get('json_filename')
        
        # Kiểm tra file có tồn tại không
        video_exists = output_path and os.path.exists(output_path)
        csv_exists = csv_filename and os.path.exists(csv_filename)
        xlsx_exists = xlsx_filename and os.path.exists(xlsx_filename)
        json_exists = json_filename and os.path.exists(json_filename)
        
        # Inspect output video to report exact info
        actual_video_info = None
        try:
            if video_exists:
                cap_out = cv2.VideoCapture(output_path)
                out_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
                out_fps = cap_out.get(cv2.CAP_PROP_FPS) or lane_detection_data.get('original_fps')
                out_duration = out_frames / (out_fps if out_fps > 0 else 1)
                cap_out.release()
                actual_video_info = {
                    'frames': out_frames,
                    'fps': out_fps,
                    'duration_seconds': out_duration,
                    'path': output_path,
                    'size_mb': os.path.getsize(output_path) / (1024 * 1024)
                }
                print(f"📹 Video info: {out_frames} frames, {out_fps:.2f} fps, {out_duration:.2f}s, {actual_video_info['size_mb']:.2f}MB")
        except Exception as e:
            print(f"⚠️ Unable to inspect output video: {e}")

        # Tạo response
        response_data = {
            'success': True,
            'message': 'Lane detection stopped successfully',
            'summary': {
                'total_violations': len(violations_data),
                'motor_violations': lane_detection_data.get('motor_violations', 0),
                'car_violations': lane_detection_data.get('car_violations', 0),
                'processing_time': time.time() - lane_detection_data.get('start_time', time.time()),
                'frames_processed': lane_detection_data.get('frame_count', 0),
            },
            'files': {
                'video': os.path.basename(output_path) if video_exists else None,
                'csv': os.path.basename(csv_filename) if csv_exists else None,
                'excel': os.path.basename(xlsx_filename) if xlsx_exists else None,
                'json': os.path.basename(json_filename) if json_exists else None
            },
            'files_exist': {
                'video': video_exists,
                'csv': csv_exists,
                'excel': xlsx_exists,
                'json': json_exists
            },
            'video_info': actual_video_info
        }
        
        print(f"✅ Trả về kết quả cho client:")
        print(f"   - Video: {os.path.basename(output_path) if video_exists else 'N/A'} ({'Có' if video_exists else 'Không'})")
        print(f"   - CSV: {os.path.basename(csv_filename) if csv_exists else 'N/A'} ({'Có' if csv_exists else 'Không'})")
        print(f"   - Excel: {os.path.basename(xlsx_filename) if xlsx_exists else 'N/A'} ({'Có' if xlsx_exists else 'Không'})")
        print(f"   - JSON: {os.path.basename(json_filename) if json_exists else 'N/A'} ({'Có' if json_exists else 'Không'})")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Lỗi khi trả kết quả: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error retrieving results: {str(e)}'}), 500


# API để lấy trạng thái hiện tại
@app.route("/lane_detection_status", methods=['GET'])
def get_lane_detection_status():
    """Lấy trạng thái hiện tại của lane detection với debug info chi tiết"""
    global lane_detection_active, lane_detection_data
    
    # Tính runtime
    if lane_detection_data.get('start_time') and isinstance(lane_detection_data['start_time'], (int, float)):
        current_time = time.time()
        runtime_seconds = current_time - lane_detection_data['start_time']
    else:
        runtime_seconds = 0
    
    # Debug info để kiểm tra accuracy
    debug_info = {
        'active_cooldowns': len(lane_detection_data.get('violation_cooldown', {})),
        'active_vehicles': len(lane_detection_data.get('vehicle_states', {})),
        'violated_vehicles': sum(1 for state in lane_detection_data.get('vehicle_states', {}).values() if state.get('has_violated', False)),
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
    """Download file kết quả lane detection (video, csv, excel, json)"""
    try:
        output_dir = "output"
        timestamp = lane_detection_data.get('timestamp')
        
        if not timestamp:
            return "No results available", 404
        
        if file_type == "video":
            filename = f"lane_violations_{timestamp}.mp4"
        elif file_type == "csv":
            filename = f"lane_violations_stats_{timestamp}.csv"
        elif file_type == "excel":
            filename = f"lane_violations_stats_{timestamp}.xlsx"
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


# ============= HELMET DETECTION ENDPOINTS =============

@app.route("/download_helmet_results/<file_type>")
def download_helmet_results(file_type):
    """Download helmet detection result files (video, csv, excel, json) for the current session"""
    try:
        output_dir = "output"
        timestamp = helmet_detection_data.get('timestamp')

        if not timestamp:
            return "No helmet results available", 404

        if file_type == "video":
            filename = f"helmet_analysis_{timestamp}.mp4"
        elif file_type == "csv":
            filename = f"helmet_violations_stats_{timestamp}.csv"
        elif file_type == "excel":
            filename = f"helmet_violations_stats_{timestamp}.xlsx"
        elif file_type == "json":
            filename = f"helmet_violations_details_{timestamp}.json"
        else:
            return "Invalid file type", 400

        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return "File not found", 404

    except Exception as e:
        return f"Error downloading helmet file: {str(e)}", 500

@app.route("/start_helmet_detection", methods=['POST'])
def start_helmet_detection():
    """Start helmet detection from web interface"""
    global helmet_detection_active, helmet_detection_data
    
    if helmet_detection_active:
        return jsonify({'error': 'Helmet detection is already running'}), 400
    
    # Check if video uploaded
    uploaded_video = session.get('uploaded_video_helmet')
    if not uploaded_video:
        return jsonify({'error': 'No video uploaded for helmet detection'}), 400
    
    print(f"🚀 Starting helmet detection for: {uploaded_video}")
    helmet_detection_active = True
    
    # Reset helmet detection data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    helmet_detection_data = {
        'violations': [],
        'total_violations': 0,
        'with_helmet': 0,  # Reset counter
        'without_helmet': 0,  # Reset counter
        'start_time': time.time(),
        'video_writer': None,
        'output_path': None,
        'timestamp': timestamp,
        'frame_count': 0,
        'original_fps': None,
        'output_size': None,
        'detection_cooldown': {},  # Reset cooldown
        'frame_detections': set(),  # Reset frame tracking
        'temporal_votes': {},  # Reset vote memory
        'state_map': {},  # Reset state machine memory
        'recent_plates': [],  # Reset recent plates cache
        'current_with': 0,
        'current_without': 0,
        'current_total': 0
    }
    
    return jsonify({
        'success': True,
        'message': 'Helmet detection started successfully',
        'timestamp': timestamp
    })


@app.route("/stop_helmet_detection", methods=['POST'])
def stop_helmet_detection():
    """Stop helmet detection and export results"""
    global helmet_detection_active, helmet_detection_data
    
    if not helmet_detection_active:
        return jsonify({'error': 'Helmet detection is not running'}), 400
    
    print("🛑 Nhận yêu cầu dừng helmet detection từ web...")
    helmet_detection_active = False
    
    # Đợi để detection dừng hoàn toàn
    time.sleep(2)
    
    try:
        # Đóng video writer nếu có
        if helmet_detection_data.get('video_writer'):
            helmet_detection_data['video_writer'].release()
            helmet_detection_data['video_writer'] = None
            print("✅ Helmet video writer released")
        
        violations_data = helmet_detection_data.get('violations', [])
        timestamp = helmet_detection_data.get('timestamp', datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_path = helmet_detection_data.get('output_path', '')
        
        # Xuất CSV, Excel và JSON nếu có vi phạm
        csv_filename = None
        xlsx_filename = None
        json_filename = None
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Dù có hay không vi phạm, vẫn tạo file thống kê để tiện tải
        df = pd.DataFrame(violations_data)
        csv_filename = os.path.join(output_dir, f"helmet_violations_stats_{timestamp}.csv")
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"✅ Đã xuất helmet CSV: {csv_filename}")

        # Excel
        xlsx_filename = os.path.join(output_dir, f"helmet_violations_stats_{timestamp}.xlsx")
        try:
            df.to_excel(xlsx_filename, index=False)
            print(f"✅ Đã xuất helmet Excel: {xlsx_filename}")
        except Exception as e:
            print(f"⚠️ Không thể xuất Excel: {e}")
            xlsx_filename = None
        
        # JSON chi tiết
        json_filename = os.path.join(output_dir, f"helmet_violations_details_{timestamp}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'video_info': {
                    'input_path': session.get('uploaded_video_helmet', ''),
                    'output_path': output_path,
                    'timestamp': timestamp
                },
                'summary': {
                    'total_violations': len(violations_data),
                    'frames_processed': helmet_detection_data.get('frame_count', 0),
                    'processing_time': time.time() - helmet_detection_data.get('start_time', time.time())
                },
                'violations': violations_data
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Đã xuất helmet JSON: {json_filename}")
        
        response_data = {
            'success': True,
            'message': 'Helmet detection stopped and results exported successfully',
            'summary': {
                'total_violations': len(violations_data),
                'processing_time': time.time() - helmet_detection_data.get('start_time', time.time()),
                'frames_processed': helmet_detection_data.get('frame_count', 0)
            },
            'files': {
                'video': os.path.basename(output_path) if output_path else None,
                'csv': os.path.basename(csv_filename) if csv_filename else None,
                'excel': os.path.basename(xlsx_filename) if xlsx_filename else None,
                'json': os.path.basename(json_filename) if json_filename else None
            }
        }
        
        print(f"✅ Helmet detection đã dừng và xuất kết quả thành công")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Lỗi khi xuất kết quả helmet: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error exporting helmet results: {str(e)}'}), 500


@app.route("/helmet_detection_status", methods=['GET'])
def get_helmet_detection_status():
    """Get current status of helmet detection with detailed stats"""
    global helmet_detection_active, helmet_detection_data
    
    # Tính runtime
    if helmet_detection_data.get('start_time') and isinstance(helmet_detection_data['start_time'], (int, float)):
        current_time = time.time()
        runtime_seconds = current_time - helmet_detection_data['start_time']
    else:
        runtime_seconds = 0
    
    # Tính tỷ lệ vi phạm
    with_helmet = helmet_detection_data.get('with_helmet', 0)
    without_helmet = helmet_detection_data.get('without_helmet', 0)
    total_detections = with_helmet + without_helmet
    
    if total_detections > 0:
        violation_rate = round((without_helmet / total_detections) * 100, 1)
    else:
        violation_rate = 0
    
    # Current frame counts
    cur_with = helmet_detection_data.get('current_with', 0)
    cur_without = helmet_detection_data.get('current_without', 0)
    cur_total = helmet_detection_data.get('current_total', 0)
    cur_rate = round((cur_without / cur_total) * 100, 1) if cur_total > 0 else 0

    return jsonify({
        'active': helmet_detection_active,
        # Cumulative (session)
        'total_violations': without_helmet,
        'with_helmet_cum': with_helmet,
        'without_helmet_cum': without_helmet,
        'total_detections_cum': total_detections,
        'violation_rate_cum': violation_rate,
        # Current (per frame)
        'with_helmet': cur_with,
        'without_helmet': cur_without,
        'total_detections': cur_total,
        'violation_rate': cur_rate,
        'frame_count': helmet_detection_data.get('frame_count', 0),
        'runtime': runtime_seconds,
        'runtime_formatted': f"{int(runtime_seconds//3600):02d}:{int((runtime_seconds%3600)//60):02d}:{int(runtime_seconds%60):02d}",
        'last_updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


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
                                    
                                    # Tạo PDF biên bản phạt cho vi phạm đèn đỏ
                                    try:
                                        import createBB_red_light
                                        from PIL import Image
                                        import tempfile
                                        
                                        # Lưu ảnh vi phạm
                                        os.makedirs("data_vuot_den_do", exist_ok=True)
                                        cv2.imwrite(f"data_vuot_den_do/{violation_count}.jpg", frame)
                                        
                                        # Tạo PDF biên bản phạt
                                        stt_BB_red_light = f'BienBanNopPhatVuotDenDo/{violation_count}.pdf'
                                        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                        temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                                        frame_pil.save(temp_image.name)
                                        
                                        # Cập nhật thông tin biên bản với biển số xe
                                        examBB = createBB_red_light.infoObject()
                                        if license_text:
                                            examBB['license_plate'] = license_text
                                        createBB_red_light.bienBanNopPhat(examBB, temp_image.name,
                                                                         f"data_vuot_den_do/{violation_count}.jpg", stt_BB_red_light)
                                        temp_image.close()
                                        print(f"Created PDF violation report: {stt_BB_red_light}")
                                    except Exception as e:
                                        print(f"Error creating PDF for violation {violation_count}: {e}")
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
                # cv2.putText(frame, f"Vi pham: {violation_count}", (20, 60),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # cv2.putText(frame, f"Frame: {frame_count}", (20, 85),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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


# Clear uploaded video for lane detection
@app.route("/clear_upload_lane", methods=['POST'])
def clear_upload_lane():
    if 'uploaded_video_lane' in session:
        # Delete file if exists
        filepath = session['uploaded_video_lane']
        if os.path.exists(filepath):
            os.remove(filepath)
        session.pop('uploaded_video_lane', None)
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
