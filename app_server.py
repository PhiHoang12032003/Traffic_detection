import datetime
import webbrowser
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO

from flask import Flask, jsonify, url_for, request, session, send_file
from flask import render_template, Response
from flask_cors import CORS
# from flask_mysqldb import MySQL  # Commented out for easier setup
# from testHelmetNew import video_detect_helmet_with_plate, get_processing_progress, reset_processing_progress
from processHelmetVideo import process_helmet_video_complete
import threading
import uuid
from testLane import *
from testRedLight import video_detect_red_light
from processRedLightVideo import process_red_light_video_complete
import createBB
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


def video_detection(path_x=""):
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


def generate_frames(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_helmet(path_x):
    yolo_output = video_detect_helmet_with_plate(path_x)
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
    
    return Response(generate_frames(path_x=uploaded_video),
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
    # Check if there's an uploaded video
    uploaded_video = session.get('uploaded_video', None)
    if not uploaded_video:
        return jsonify({'error': 'No video uploaded'}), 400
    
    return Response(generate_frames_red_light(path_x=uploaded_video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Route to upload video for analysis
@app.route("/upload_video", methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store in session
        session['uploaded_video'] = filepath
        
        # Get detection method preference
        detection_method = request.form.get('detection_method', 'original')
        session['red_light_advanced'] = (detection_method == 'advanced')
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'detection_method': detection_method,
            'message': f'Video uploaded successfully! Using {detection_method} detection method.'
        })
    
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


# Generate frames for red light detection (basic streaming)
def generate_frames_red_light(path_x):
    red_light_output = video_detect_red_light(path_x)
    
    for detection_ in red_light_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


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


@app.route("/process_red_light_video", methods=['POST'])
def process_red_light_video():
    """Process red light video and generate output video"""
    uploaded_video = session.get('uploaded_video', None)
    if not uploaded_video:
        return jsonify({'error': 'No video uploaded'}), 400
    
    if not os.path.exists(uploaded_video):
        return jsonify({'error': 'Uploaded video file not found'}), 400
    
    try:
        # Generate unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        # Create output path
        output_path = os.path.join('processed_videos', f'red_light_{job_id}_processed.mp4')
        
        # Store processing info in session
        session['red_light_processing'] = {
            'job_id': job_id,
            'status': 'processing',
            'input_path': uploaded_video,
            'output_path': output_path,
            'message': 'Processing red light detection...'
        }
        
        # Get detection method before starting thread
        use_advanced = session.get('red_light_advanced', False)
        
        # Start processing in background thread
        def process_video_bg():
            try:
                result_path, stats = process_red_light_video_complete(
                    uploaded_video, 
                    output_path, 
                    use_improved_detection=use_advanced
                )
                
                # Use app context to update session
                with app.app_context():
                    from flask import session as thread_session
                    # Store results in a temporary file for status checking
                    import json
                    status_file = f"{output_path}_status.json"
                    status_data = {
                        'status': 'completed',
                        'stats': stats,
                        'output_path': result_path,
                        'message': 'Processing completed successfully!'
                    }
                    with open(status_file, 'w') as f:
                        json.dump(status_data, f)
                
            except Exception as e:
                # Store error status in file
                with app.app_context():
                    import json
                    status_file = f"{output_path}_status.json"
                    status_data = {
                        'status': 'error',
                        'error': str(e),
                        'message': f'Processing failed: {str(e)}'
                    }
                    with open(status_file, 'w') as f:
                        json.dump(status_data, f)
        
        # Start background processing
        thread = threading.Thread(target=process_video_bg)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Video processing started. Please wait...',
            'status_url': '/api/red_light_processing_status'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/red_light_processing_status", methods=['GET'])
def get_red_light_processing_status():
    """Get red light processing status"""
    processing_info = session.get('red_light_processing', {})
    
    if not processing_info:
        return jsonify({'status': 'no_job'})
    
    # Check if there's a status file from background thread
    output_path = processing_info.get('output_path')
    if output_path:
        status_file = f"{output_path}_status.json"
        if os.path.exists(status_file):
            try:
                import json
                with open(status_file, 'r') as f:
                    file_status = json.load(f)
                
                # Update session with file status
                session['red_light_processing'].update(file_status)
                processing_info = session['red_light_processing']
                
                # Clean up status file
                os.remove(status_file)
            except Exception as e:
                print(f"Error reading status file: {e}")
    
    response = {
        'status': processing_info.get('status', 'processing'),
        'job_id': processing_info.get('job_id'),
        'message': processing_info.get('message', 'Processing...')
    }
    
    if processing_info.get('status') == 'completed':
        response['stats'] = processing_info.get('stats', {})
        response['output_path'] = processing_info.get('output_path')
        response['download_url'] = f"/download_processed_video/{processing_info.get('job_id')}"
    elif processing_info.get('status') == 'error':
        response['error'] = processing_info.get('error', 'Unknown error')
    
    return jsonify(response)


@app.route("/download_processed_video/<job_id>", methods=['GET'])
def download_processed_video(job_id):
    """Download processed red light video"""
    processing_info = session.get('red_light_processing', {})
    
    if (processing_info.get('job_id') != job_id or 
        processing_info.get('status') != 'completed'):
        return jsonify({'error': 'Invalid job ID or processing not completed'}), 404
    
    output_path = processing_info.get('output_path')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'Processed video file not found'}), 404
    
    return send_file(output_path, 
                     mimetype='video/mp4',
                     as_attachment=True,
                     download_name=f'red_light_detection_{job_id}.mp4')


if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:8000/')
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=True)
