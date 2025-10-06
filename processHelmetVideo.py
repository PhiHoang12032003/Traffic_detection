# Process entire video for helmet violations and save result - OPTIMIZED VERSION
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
from collections import defaultdict
import tempfile
from PIL import Image
import createBB_helmet
from testLane import *
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helmet_pdf_utils import create_helmet_pdf_report, get_helmet_violation_info
# Removed accuracy_config import - using fixed values instead

# Removed violation validation function - no longer counting violations


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
    
    # Load YOLOv8 Custom helmet model
    model = YOLO('model_helmet/best_helmet_end.pt')
    
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
    name_class = ["without helmet", "helmet"]
    pdf_creation_count = 0
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # YOLOv8 Custom helmet detection (every frame for accuracy)
        results = model(frame)
        
        # Simple detection - just draw bounding boxes
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > 0.5:  # Simple confidence threshold
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw bounding box based on class
                        if cls == 0:  # No helmet - VIOLATION
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"No Helmet: {conf:.2f}", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            # Create PDF for violation (every 5th detection)
                            pdf_creation_count += 1
                            if pdf_creation_count % 5 == 1:  # Create PDF for 1st, 6th, 11th, etc.
                                try:
                                    # Save violation image
                                    imageViolateHelmet(frame, 0, height, 0, width, pdf_creation_count)
                                    
                                    # Sử dụng utils mới để tạo PDF
                                    pdf_path = create_helmet_pdf_report(frame, pdf_creation_count)
                                    print(f"📄 Created PDF violation report: {pdf_path}")
                                except Exception as e:
                                    print(f"❌ Error saving evidence: {e}")
                        else:  # Helmet detected
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Helmet: {conf:.2f}", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
