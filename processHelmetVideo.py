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
    
    # Load YOLOv8 Custom helmet model v2
    model = YOLO('model_helmet_v2/best.pt')
    
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
    name_class = ["with helmet", "without helmet", "rider", "number plate"]
    pdf_creation_count = 0
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # YOLOv8 Custom helmet detection (every frame for accuracy)
        results = model(frame)
        
        # Variables to store detection results for this frame
        current_license_plate = None
        has_helmet_violation = False
        
        # Simple detection - just draw bounding boxes
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > 0.5:  # Simple confidence threshold
                        # Lấy tọa độ và đảm bảo nằm trong frame
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Clamp coordinates to frame boundaries
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width - 1))
                        y2 = max(0, min(y2, height - 1))
                        
                        # Draw bounding box based on class
                        if cls == 1:  # without helmet - VIOLATION (class 1 in new model)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            label = f"KHONG MU: {conf:.2f}"
                            # Vẽ background cho text để dễ đọc
                            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)
                            cv2.putText(frame, label, (x1, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            has_helmet_violation = True
                            
                        elif cls == 0:  # with helmet detected
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            label = f"CO MU: {conf:.2f}"
                            # Vẽ background cho text để dễ đọc
                            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                            cv2.putText(frame, label, (x1, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        elif cls == 2:  # rider detected - BỎ VẼ BOUNDING BOX
                            pass  # Không vẽ gì cả
                        elif cls == 3:  # number plate detected
                            # Luôn vẽ bounding box cho biển số xe (màu tím)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            
                            # IMPROVED OCR - Theo notebook model_helmet_v2
                            plate_text = "BIEN SO"  # Text mặc định
                            if reader is not None:
                                try:
                                    # Crop the number plate region
                                    plate_crop = frame[y1:y2, x1:x2]
                                    
                                    # Kiểm tra crop có hợp lệ
                                    if plate_crop.size > 0 and plate_crop.shape[0] >= 10 and plate_crop.shape[1] >= 20:
                                        # Preprocess for better OCR - Cải tiến từ notebook
                                        # Convert to grayscale
                                        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                        
                                        # Apply denoising để giảm nhiễu
                                        gray_plate = cv2.fastNlMeansDenoising(gray_plate, None, 30, 7, 21)
                                        
                                        # Apply adaptive thresholding (tốt hơn binary threshold)
                                        thresh_plate = cv2.adaptiveThreshold(
                                            gray_plate, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2
                                        )
                                        
                                        # Apply Gaussian blur để làm mượt
                                        thresh_plate = cv2.GaussianBlur(thresh_plate, (5, 5), 0)
                                        
                                        # Use EasyOCR to extract text
                                        ocr_results = reader.readtext(thresh_plate)
                                        
                                        if ocr_results:
                                            # Collect all detected texts with confidence > 0.3
                                            detected_texts = []
                                            for (bbox, text, prob) in ocr_results:
                                                if prob > 0.3:
                                                    # Clean text: remove spaces, uppercase, alphanumeric only
                                                    cleaned_text = text.strip().replace(" ", "").upper()
                                                    cleaned_text = ''.join(filter(str.isalnum, cleaned_text))
                                                    # Lọc text có độ dài hợp lý cho biển số (5-10 ký tự)
                                                    if len(cleaned_text) >= 5 and len(cleaned_text) <= 10:
                                                        detected_texts.append((cleaned_text, prob))
                                            
                                            # Get the best result (longest text first, then highest confidence)
                                            if detected_texts:
                                                detected_texts.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
                                                best_text, best_confidence = detected_texts[0]
                                                
                                                # Update plate text
                                                plate_text = best_text
                                                print(f"🔍 Detected plate: {best_text} (conf: {best_confidence:.2f})")
                                                
                                                # Store the detected license plate for this frame
                                                current_license_plate = best_text
                                        
                                except Exception as e:
                                    # Silent fail - không in lỗi cho mỗi frame
                                    pass
                            
                            # Vẽ text trên bounding box (hoặc "BIEN SO" nếu chưa nhận diện được)
                            label = f"{plate_text}"
                            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (255, 0, 255), -1)
                            cv2.putText(frame, label, (x1, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
