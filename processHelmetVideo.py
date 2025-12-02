# Process entire video for helmet violations and save result - OPTIMIZED VERSION
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
from collections import defaultdict
import tempfile
from PIL import Image
import json
import sys
import os
import easyocr
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helmet_pdf_utils import create_helmet_pdf_report, get_helmet_violation_info
# Removed accuracy_config import - using fixed values instead

# Removed violation validation function - no longer counting violations


def imageViolateHelmet(frame, r1, r2, c1, c2, stt):
    """Lưu ảnh vi phạm mũ bảo hiểm."""
    try:
        cropped_region = frame[r1:r2, c1:c2].copy()
        save_dir = "data_xe_vp_bh"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{stt}.jpg")
        cv2.imwrite(save_path, cropped_region)
        return save_path
    except Exception as e:
        print(f"⚠️ Lỗi khi lưu ảnh vi phạm: {e}")
        return None


def process_helmet_video_complete(input_path, output_path, use_improved_detection=False):
    """
    SIMPLE HELMET DETECTION - No violation counting
    Args:
        input_path: Path to input video
        output_path: Path to save processed video
        use_improved_detection: Ignored
    Returns: Path to processed video and basic stats
    """
    print(f"🚀 Starting helmet detection: {input_path}")
    print("⚡ Processing with simple detection - no violation counting...")
    
    # Load YOLOv8 Custom helmet model v2 với tham số tối ưu
    model_path = 'model_helmet_v2/best.pt'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return None, {}
    
    print(f"✅ Loading helmet model from: {model_path}")
    model = YOLO(model_path)
    
    # In thông tin model để debug
    print(f"📊 Model classes: {model.names if hasattr(model, 'names') else 'Unknown'}")
    
    # Initialize EasyOCR for number plate text recognition
    try:
        reader = easyocr.Reader(['en'])
        print("✅ EasyOCR initialized successfully")
    except Exception as e:
        print(f"⚠️ EasyOCR initialization failed: {e}")
        reader = None
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {input_path}")
        return None, {}
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Simple processing variables
    frame_count = 0
    # MODEL HELMET V2 CLASSES (THỰC TẾ): 0=head (không mũ), 1=helmet (có mũ), 2=person (người)
    name_class = model.names if hasattr(model, 'names') else {0: 'head', 1: 'helmet', 2: 'person'}
    pdf_creation_count = 0
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # YOLOv8 Custom helmet detection với tham số tối ưu (giống Colab)
        # conf: confidence threshold (0.25 theo Colab)
        # imgsz: image size 800 (theo Colab - model được train với imgsz=800)
        # classes: chỉ detect class 0 (head) và class 1 (helmet), bỏ class 2 (person)
        results = model(frame, 
                       conf=0.25,  # Confidence threshold theo Colab
                       imgsz=800,  # Image size 800 - GIỐNG COLAB để nhận diện chính xác
                       classes=[0, 1],  # Chỉ detect class 0 (head) và 1 (helmet), bỏ class 2 (person)
                       verbose=False)  # Không in log chi tiết
        
        # Variables to store detection results for this frame
        current_license_plate = None
        has_helmet_violation = False
        
        # SỬ DỤNG PLOT() MẶC ĐỊNH CỦA YOLO GIỐNG COLAB - KHÔNG VẼ THỦ CÔNG
        # YOLO plot() sẽ tự động vẽ boxes với màu sắc và labels phù hợp
        if results and len(results) > 0:
            # Dùng plot() mặc định của YOLO - giống Colab
            annotated_frame = results[0].plot(conf=True, labels=True, boxes=True)
            frame = annotated_frame
            
            # Kiểm tra vi phạm (class 0 = head = không mũ)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # head (không mũ) - VI PHẠM
                        has_helmet_violation = True
        
        # Create PDF for violation if helmet violation detected
        if has_helmet_violation:
            pdf_creation_count += 1
            if pdf_creation_count % 5 == 1:  # Create PDF for 1st, 6th, 11th, etc.
                try:
                    # Save violation image
                    imageViolateHelmet(frame, 0, height, 0, width, pdf_creation_count)
                    
                    # Sử dụng utils mới để tạo PDF với biển số xe
                    pdf_path = create_helmet_pdf_report(frame, pdf_creation_count, current_license_plate)
                    print(f"📄 Created PDF violation report: {pdf_path}")
                    if current_license_plate:
                        print(f"🚗 License plate included in PDF: {current_license_plate}")
                except Exception as e:
                    print(f"❌ Error saving evidence: {e}")
        
        # Add frame info
        # cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(frame, f"Violations: {violation_count}", (10, 70), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Progress update
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"📊 Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Simple statistics
    stats = {
        'total_frames': frame_count,
        'processing_mode': 'SIMPLE_HELMET_DETECTION',
        'message': 'No violation counting - simple detection only'
    }
    
    print(f"✅ Processing complete!")
    print(f"📊 Total frames processed: {frame_count}")
    print(f"💾 Output saved: {output_path}")
    print("🎯 SIMPLE MODE: No violation counting - detection only")
    
    return output_path, stats
