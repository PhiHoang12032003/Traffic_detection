import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from datetime import datetime
import os
from collections import defaultdict
import time

# Initialize YOLO model for vehicle detection
vehicle_model = YOLO('best_new/vehicle.pt')

# Initialize EasyOCR for license plate recognition
reader = easyocr.Reader(['vi', 'en'])  # Vietnamese and English

# Create directories if not exist
os.makedirs('data_vuot_den_do', exist_ok=True)
os.makedirs('BienBanNopPhatVuotDenDo', exist_ok=True)

# Traffic light detection parameters
RED_LOWER = np.array([0, 120, 70])
RED_UPPER = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

GREEN_LOWER = np.array([40, 40, 40])
GREEN_UPPER = np.array([80, 255, 255])

YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([30, 255, 255])


def detect_traffic_light_color(frame, roi):
    """Detect traffic light color in a region of interest"""
    x, y, w, h = roi
    traffic_light_region = frame[y:y+h, x:x+w]
    
    # Convert to HSV
    hsv = cv2.cvtColor(traffic_light_region, cv2.COLOR_BGR2HSV)
    
    # Create masks for different colors
    red_mask1 = cv2.inRange(hsv, RED_LOWER, RED_UPPER)
    red_mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    
    # Count pixels
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    
    # Determine dominant color
    if red_pixels > green_pixels and red_pixels > yellow_pixels:
        return "RED"
    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
        return "GREEN"
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
        return "YELLOW"
    else:
        return "UNKNOWN"


def extract_license_plate(frame, bbox):
    """Extract license plate from vehicle bounding box"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop vehicle region
    vehicle_crop = frame[y1:y2, x1:x2]
    
    # Focus on lower part where license plate usually is
    h, w = vehicle_crop.shape[:2]
    license_region = vehicle_crop[int(h*0.6):, :]
    
    # Try to detect license plate text
    try:
        results = reader.readtext(license_region)
        for (bbox, text, prob) in results:
            if prob > 0.5 and len(text) > 5:  # Filter by confidence and length
                # Clean text - remove spaces and special characters
                cleaned_text = ''.join(c for c in text if c.isalnum())
                if len(cleaned_text) >= 6:  # Valid license plate length
                    return cleaned_text.upper()
    except:
        pass
    
    return None


def video_detect_red_light(video_path):
    """Main function to detect red light violations"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define regions
    # Traffic light region (top center of frame)
    traffic_light_roi = (int(width*0.4), int(height*0.1), int(width*0.2), int(height*0.2))
    
    # Stop line (horizontal line where vehicles should stop)
    stop_line_y = int(height * 0.6)
    
    # Track vehicles
    vehicle_tracker = defaultdict(lambda: {"positions": [], "violated": False, "license_plate": None})
    
    frame_count = 0
    current_light = "UNKNOWN"
    violation_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Detect traffic light color every 10 frames
        if frame_count % 10 == 0:
            current_light = detect_traffic_light_color(frame, traffic_light_roi)
        
        # Detect vehicles
        results = vehicle_model(frame)
        
        # Draw traffic light status
        light_color = (0, 0, 255) if current_light == "RED" else (0, 255, 0) if current_light == "GREEN" else (0, 255, 255)
        cv2.rectangle(frame, (10, 10), (200, 60), light_color, -1)
        cv2.putText(frame, f"Light: {current_light}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw stop line
        cv2.line(frame, (0, stop_line_y), (width, stop_line_y), (0, 255, 255), 3)
        cv2.putText(frame, "STOP LINE", (10, stop_line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Process detections
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Filter for vehicles only (car, motorcycle, bus, truck)
                    if cls in [2, 3, 5, 7] and conf > 0.5:
                        # Vehicle center
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Simple tracking by position
                        vehicle_id = f"{center_x}_{center_y}"
                        
                        # Check for red light violation
                        if current_light == "RED" and center_y > stop_line_y and not vehicle_tracker[vehicle_id]["violated"]:
                            vehicle_tracker[vehicle_id]["violated"] = True
                            violation_count += 1
                            
                            # Try to extract license plate
                            license_plate = extract_license_plate(frame, (x1, y1, x2, y2))
                            vehicle_tracker[vehicle_id]["license_plate"] = license_plate
                            
                            # Save violation image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"data_vuot_den_do/violation_{timestamp}_{violation_count}.jpg"
                            cv2.imwrite(filename, frame)
                            
                            # Draw violation alert
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, "VIOLATION!", (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
                            if license_plate:
                                cv2.putText(frame, f"LP: {license_plate}", (x1, y2 + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            # Normal vehicle
                            color = (0, 255, 0) if current_light == "GREEN" else (255, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Display violation count
        cv2.putText(frame, f"Violations: {violation_count}", (width - 200, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw traffic light ROI
        x, y, w, h = traffic_light_roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(frame, "Traffic Light ROI", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        yield frame
    
    cap.release()


# Test function
if __name__ == "__main__":
    # Test with a video
    video_path = "Videos/traffic1.mp4"
    if os.path.exists(video_path):
        for frame in video_detect_red_light(video_path):
            cv2.imshow("Red Light Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        print(f"Video file not found: {video_path}")