import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from datetime import datetime
import os
from collections import defaultdict
import time
import json
import tempfile
from PIL import Image
import createBB
import uuid


class RedLightDetectorWithOutput:
    """
    Advanced Red Light Detector with Video Output Support
    """
    
    def __init__(self):
        # Initialize YOLO models
        self.model = YOLO('YoloWeights/yolov8n.pt')  # General YOLO for traffic lights
        self.vehicle_model = YOLO('best_new/vehicle.pt')  # Vehicle specific model
        
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['vi', 'en'])
        
        # Create directories
        os.makedirs('data_vuot_den_do', exist_ok=True)
        os.makedirs('BienBanNopPhatVuotDenDo', exist_ok=True)
        os.makedirs('processed_videos', exist_ok=True)
        
        # Traffic light color detection parameters (HSV)
        self.color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            'yellow': [(np.array([15, 50, 50]), np.array([40, 255, 255]))],
            'green': [(np.array([40, 50, 50]), np.array([90, 255, 255]))]
        }
        
        # Vehicle tracking and statistics
        self.processed_vehicles = set()
        self.violation_count = 0
        self.total_frames = 0
        self.current_frame = 0
        
        # Initialize BB creator
        self.examBB = createBB.infoObject()
        
        # Statistics for final report
        self.statistics = {
            'total_violations': 0,
            'vehicle_types': defaultdict(int),
            'detected_plates': [],
            'processing_time': 0,
            'total_frames': 0
        }

    def detect_traffic_lights(self, frame):
        """Detect traffic lights using YOLO"""
        results = self.model(frame)
        traffic_lights = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Traffic light class ID in COCO is 9
                    if cls == 9 and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        traffic_lights.append((x1, y1, x2, y2, conf))
        
        return traffic_lights

    def determine_traffic_light_color(self, frame, bbox):
        """Determine traffic light color using HSV analysis"""
        x1, y1, x2, y2 = bbox
        
        # Extract traffic light region
        traffic_light = frame[y1:y2, x1:x2]
        
        if traffic_light.size == 0:
            return "unknown"
        
        # Apply Gaussian blur to reduce noise
        traffic_light = cv2.GaussianBlur(traffic_light, (5, 5), 0)
        
        # Convert to HSV
        hsv = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)
        
        # Count pixels for each color
        color_pixels = {}
        
        for color, ranges in self.color_ranges.items():
            total_pixels = 0
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                total_pixels += cv2.countNonZero(mask)
            color_pixels[color] = total_pixels
        
        # Calculate minimum threshold
        total_area = traffic_light.shape[0] * traffic_light.shape[1]
        min_pixel_threshold = max(30, total_area * 0.03)
        
        # Determine dominant color
        max_color = max(color_pixels, key=color_pixels.get)
        max_pixels = color_pixels[max_color]
        
        if max_pixels > min_pixel_threshold:
            return max_color
        else:
            return "unknown"

    def detect_vehicles_crossing_line(self, frame, results, stop_line_y, current_light):
        """Detect vehicles crossing the stop line during red light"""
        violating_vehicles = []
        vehicle_classes = [0, 1, 2, 3]  # Vehicle classes in custom model
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls in vehicle_classes and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Check if vehicle crossed the stop line
                        vehicle_bottom = y2
                        
                        # Only flag as violation if:
                        # 1. Light is red
                        # 2. Vehicle bottom is past the stop line
                        # 3. Vehicle hasn't been processed yet
                        if (current_light == "red" and 
                            vehicle_bottom > stop_line_y):
                            
                            vehicle_id = f"{x1}_{y1}_{x2}_{y2}"
                            if vehicle_id not in self.processed_vehicles:
                                vehicle_type = self.vehicle_model.names[cls] if hasattr(self.vehicle_model, 'names') else f"vehicle_{cls}"
                                violating_vehicles.append(((x1, y1, x2, y2), vehicle_type))
        
        return violating_vehicles

    def extract_license_plate(self, frame, bbox):
        """Extract and recognize license plate using EasyOCR"""
        x1, y1, x2, y2 = bbox
        
        # Crop vehicle region
        vehicle_img = frame[y1:y2, x1:x2]
        if vehicle_img.size == 0:
            return None
        
        # Focus on lower part where license plate usually is
        h, w = vehicle_img.shape[:2]
        license_region = vehicle_img[int(h*0.6):, :]
        
        try:
            results = self.reader.readtext(license_region)
            for (bbox_ocr, text, prob) in results:
                if prob > 0.5 and len(text) > 5:
                    # Clean text
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    if len(cleaned_text) >= 6:
                        return cleaned_text.upper()
        except Exception as e:
            print(f"OCR Error: {e}")
        
        return None

    def save_violation(self, frame, bbox, vehicle_type, plate_text, violation_id):
        """Save violation data and create fine document"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save violation image
        filename = f"data_vuot_den_do/violation_{timestamp}_{violation_id}.jpg"
        cv2.imwrite(filename, frame)
        
        # Update statistics
        self.statistics['detected_plates'].append({
            'plate': plate_text,
            'vehicle_type': vehicle_type,
            'timestamp': timestamp,
            'image_path': filename
        })
        
        # Create fine document if license plate is detected
        if plate_text:
            stt_BB = f'BienBanNopPhatVuotDenDo/{violation_id}.pdf'
            
            # Convert frame to PIL for document creation
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            frame_pil.save(temp_image.name)
            
            try:
                createBB.bienBanNopPhat(
                    self.examBB,
                    temp_image.name,
                    filename,
                    stt_BB
                )
                print(f"Created fine document: {stt_BB}")
            except Exception as e:
                print(f"Error creating fine document: {e}")
            finally:
                temp_image.close()
                os.unlink(temp_image.name)

    def draw_violation_info(self, frame, bbox, vehicle_type, plate_text=None):
        """Draw violation information on frame"""
        x1, y1, x2, y2 = bbox
        
        # Draw red bounding box for violation
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Draw violation text
        cv2.putText(frame, f"VI PHAM: {vehicle_type}", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if plate_text:
            cv2.putText(frame, f"Bien so: {plate_text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add violation alert
        cv2.putText(frame, "VUOT DEN DO!", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame

    def draw_traffic_info(self, frame, current_light, stop_line_y, violation_count, frame_count):
        """Draw traffic information on frame"""
        height, width = frame.shape[:2]
        
        # Draw stop line with color based on traffic light status
        if current_light == "red":
            line_color = (0, 0, 255)  # Red
            thickness = 4
        elif current_light == "yellow":
            line_color = (0, 255, 255)  # Yellow
            thickness = 3
        elif current_light == "green":
            line_color = (0, 255, 0)  # Green
            thickness = 3
        else:
            line_color = (255, 255, 255)  # White for unknown
            thickness = 2
        
        # Draw stop line
        cv2.line(frame, (0, stop_line_y), (width, stop_line_y), line_color, thickness)
        cv2.putText(frame, f"VACH DUNG - {current_light.upper()}", (10, stop_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
        
        # Draw information panel
        panel_height = 120
        cv2.rectangle(frame, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, panel_height), (255, 255, 255), 2)
        
        # Traffic light status
        status_color = (0, 0, 255) if current_light == "red" else (0, 255, 0) if current_light == "green" else (0, 255, 255)
        cv2.putText(frame, f"Den giao thong: {current_light.upper()}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Violation count
        cv2.putText(frame, f"So vi pham: {violation_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_count}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

    def process_video_with_output(self, input_path, output_path=None):
        """
        Process video and save output with annotations
        
        Args:
            input_path: Path to input video
            output_path: Path for output video (auto-generated if None)
            
        Returns:
            Tuple of (output_path, statistics)
        """
        start_time = time.time()
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.statistics['total_frames'] = total_frames
        
        # Generate output path if not provided
        if output_path is None:
            job_id = str(uuid.uuid4())
            output_path = f"processed_videos/red_light_detection_{job_id}.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Define stop line position
        stop_line_y = int(height * 0.65)
        
        frame_count = 0
        current_light = "unknown"
        
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Total frames: {total_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.current_frame = frame_count
            
            # Progress update
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Detect traffic lights every 10 frames for performance
            if frame_count % 10 == 0:
                traffic_lights = self.detect_traffic_lights(frame)
                
                if traffic_lights:
                    # Use the traffic light with highest confidence
                    best_light = max(traffic_lights, key=lambda x: x[4])
                    current_light = self.determine_traffic_light_color(frame, best_light[:4])
                    
                    # Draw traffic light detection
                    x1, y1, x2, y2, conf = best_light
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, f"TL: {current_light} ({conf:.2f})", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Detect vehicles
            vehicle_results = self.vehicle_model(frame)
            
            # Check for violations only when light is red
            if current_light == "red":
                violations = self.detect_vehicles_crossing_line(frame, vehicle_results, stop_line_y, current_light)
                
                for bbox, vehicle_type in violations:
                    x1, y1, x2, y2 = bbox
                    vehicle_id = f"{x1}_{y1}_{x2}_{y2}"
                    
                    if vehicle_id not in self.processed_vehicles:
                        self.processed_vehicles.add(vehicle_id)
                        self.violation_count += 1
                        self.statistics['total_violations'] += 1
                        self.statistics['vehicle_types'][vehicle_type] += 1
                        
                        # Extract license plate
                        plate_text = self.extract_license_plate(frame, bbox)
                        
                        if plate_text:
                            # Save violation with all details
                            self.save_violation(frame, bbox, vehicle_type, plate_text, self.violation_count)
                            print(f"Red light violation detected: {plate_text} ({vehicle_type})")
                        
                        # Draw violation info
                        frame = self.draw_violation_info(frame, bbox, vehicle_type, plate_text)
            
            # Draw normal vehicle detections
            for r in vehicle_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls in [0, 1, 2, 3] and conf > 0.5:  # Vehicle classes
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Check if this is a violation
                            vehicle_id = f"{x1}_{y1}_{x2}_{y2}"
                            if vehicle_id not in self.processed_vehicles:
                                # Normal vehicle - green box
                                color = (0, 255, 0) if current_light != "red" else (255, 255, 0)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Add vehicle type label
                                vehicle_name = self.vehicle_model.names[cls] if hasattr(self.vehicle_model, 'names') else f"vehicle_{cls}"
                                cv2.putText(frame, vehicle_name, 
                                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw traffic information overlay
            frame = self.draw_traffic_info(frame, current_light, stop_line_y, self.violation_count, frame_count)
            
            # Write frame to output video
            out.write(frame)
            
            # Clean up old tracked vehicles periodically
            if frame_count % 100 == 0 and len(self.processed_vehicles) > 50:
                recent_vehicles = set(list(self.processed_vehicles)[-25:])
                self.processed_vehicles = recent_vehicles
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.statistics['processing_time'] = processing_time
        
        print(f"\nProcessing completed!")
        print(f"Output saved to: {output_path}")
        print(f"Total violations detected: {self.violation_count}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Save statistics
        stats_path = output_path.replace('.mp4', '_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, ensure_ascii=False, indent=2)
        
        return output_path, self.statistics


# Integration function for Flask app
def process_red_light_video_complete(input_path, output_path=None):
    """
    Complete red light detection processing with video output
    
    Args:
        input_path: Path to input video
        output_path: Path for output video (auto-generated if None)
        
    Returns:
        Tuple of (output_path, statistics)
    """
    detector = RedLightDetectorWithOutput()
    return detector.process_video_with_output(input_path, output_path)


# Streaming function for real-time display
def video_detect_red_light_advanced_stream(video_path):
    """Generator function for streaming processed frames"""
    detector = RedLightDetectorWithOutput()
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    stop_line_y = int(height * 0.65)
    frame_count = 0
    current_light = "unknown"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect traffic lights every 10 frames
        if frame_count % 10 == 0:
            traffic_lights = detector.detect_traffic_lights(frame)
            if traffic_lights:
                best_light = max(traffic_lights, key=lambda x: x[4])
                current_light = detector.determine_traffic_light_color(frame, best_light[:4])
                
                # Draw traffic light detection
                x1, y1, x2, y2, conf = best_light
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"TL: {current_light} ({conf:.2f})", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Detect vehicles
        vehicle_results = detector.vehicle_model(frame)
        
        # Check for violations
        if current_light == "red":
            violations = detector.detect_vehicles_crossing_line(frame, vehicle_results, stop_line_y, current_light)
            
            for bbox, vehicle_type in violations:
                x1, y1, x2, y2 = bbox
                vehicle_id = f"{x1}_{y1}_{x2}_{y2}"
                
                if vehicle_id not in detector.processed_vehicles:
                    detector.processed_vehicles.add(vehicle_id)
                    detector.violation_count += 1
                    
                    plate_text = detector.extract_license_plate(frame, bbox)
                    if plate_text:
                        detector.save_violation(frame, bbox, vehicle_type, plate_text, detector.violation_count)
                    
                    frame = detector.draw_violation_info(frame, bbox, vehicle_type, plate_text)
        
        # Draw normal vehicles
        for r in vehicle_results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls in [0, 1, 2, 3] and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vehicle_id = f"{x1}_{y1}_{x2}_{y2}"
                        
                        if vehicle_id not in detector.processed_vehicles:
                            color = (0, 255, 0) if current_light != "red" else (255, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw traffic info
        frame = detector.draw_traffic_info(frame, current_light, stop_line_y, detector.violation_count, frame_count)
        
        yield frame
    
    cap.release()


# Test function
if __name__ == "__main__":
    detector = RedLightDetectorWithOutput()
    
    # Test with a video
    video_path = "Videos/traffic1.mp4"
    if os.path.exists(video_path):
        print("Starting red light detection with video output...")
        output_path, stats = detector.process_video_with_output(video_path)
        print(f"Video processing completed. Output: {output_path}")
        print("Statistics:", json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print(f"Video file not found: {video_path}")
        print("Available videos:")
        if os.path.exists("Videos"):
            for f in os.listdir("Videos"):
                if f.endswith(('.mp4', '.avi', '.mov')):
                    print(f"  - {f}")

