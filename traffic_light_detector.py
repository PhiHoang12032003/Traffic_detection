import cv2
import numpy as np
from ultralytics import YOLO


class TrafficLightDetector:
    """
    Advanced Traffic Light Detection System
    Integrates YOLO object detection with HSV color analysis
    """
    
    def __init__(self, model_path='YoloWeights/yolov8n.pt'):
        """Initialize the traffic light detector"""
        self.model = YOLO(model_path)
        
        # HSV color ranges for traffic light colors
        self.color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),      # Lower red range
                (np.array([160, 100, 100]), np.array([180, 255, 255]))   # Upper red range
            ],
            'yellow': [
                (np.array([15, 50, 50]), np.array([40, 255, 255]))        # Yellow range
            ],
            'green': [
                (np.array([40, 50, 50]), np.array([90, 255, 255]))        # Green range
            ]
        }
    
    def detect_traffic_lights(self, frame, confidence_threshold=0.33):
        """
        Detect traffic lights in frame using YOLO
        
        Args:
            frame: Input video frame
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of traffic light bounding boxes with confidence scores
        """
        results = self.model(frame)
        traffic_lights = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls.item())
                    cls_name = result.names[cls]
                    conf = box.conf.item()
                    
                    # Traffic light class (class 9 in COCO dataset)
                    if cls_name == "traffic light" and conf > confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        traffic_lights.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'class_name': cls_name
                        })
        
        return traffic_lights
    
    def determine_color(self, frame, bbox, conf):
        """
        Determine traffic light color using HSV analysis
        
        Args:
            frame: Input video frame
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            conf: Confidence score
            
        Returns:
            Tuple of (annotated_frame, color_name)
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        result_frame = frame.copy()
        
        # Extract traffic light region
        traffic_light = frame[y1:y2, x1:x2]
        
        if traffic_light.size == 0:
            color_name = "unknown"
        else:
            # Apply Gaussian blur to reduce noise
            traffic_light = cv2.GaussianBlur(traffic_light, (5, 5), 0)
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)
            
            # Count pixels for each color
            color_pixel_counts = {}
            
            for color, ranges in self.color_ranges.items():
                total_pixels = 0
                for lower_bound, upper_bound in ranges:
                    mask = cv2.inRange(hsv, lower_bound, upper_bound)
                    total_pixels += cv2.countNonZero(mask)
                color_pixel_counts[color] = total_pixels
            
            # Calculate minimum pixel threshold
            total_area = traffic_light.shape[0] * traffic_light.shape[1]
            min_pixel_threshold = max(30, total_area * 0.03)
            
            # Determine dominant color
            max_pixels = max(color_pixel_counts.values())
            color_name = "unknown"
            
            for color, pixel_count in color_pixel_counts.items():
                if pixel_count == max_pixels and pixel_count > min_pixel_threshold:
                    color_name = color
                    break
        
        # Draw bounding box with color-coded border
        if color_name == "red":
            border_color = (0, 0, 255)
        elif color_name == "yellow":
            border_color = (0, 255, 255)
        elif color_name == "green":
            border_color = (0, 255, 0)
        else:
            border_color = (255, 255, 255)
        
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), border_color, 2)
        
        # Add label
        label = f"light: {color_name} ({conf:.2f})"
        cv2.putText(result_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame, color_name
    
    def count_traffic_lights(self, frame):
        """
        Count number of traffic lights in frame
        
        Args:
            frame: Input video frame
            
        Returns:
            Number of traffic lights detected
        """
        traffic_lights = self.detect_traffic_lights(frame)
        return len(traffic_lights)
    
    def draw_colored_lines(self, frame, color_name, lines):
        """
        Draw detection lines with colors based on traffic light status
        
        Args:
            frame: Input video frame
            color_name: Current traffic light color
            lines: List of line coordinates
            
        Returns:
            Frame with colored lines drawn
        """
        result_frame = frame.copy()
        
        # Set line color and thickness based on traffic light color
        if color_name == "red":
            line_color = (0, 0, 255)  # Red
            thickness = 3
        elif color_name == "yellow":
            line_color = (0, 255, 255)  # Yellow
            thickness = 3
        elif color_name == "green":
            line_color = (0, 255, 0)  # Green
            thickness = 3
        else:
            line_color = (255, 255, 255)  # White for unknown
            thickness = 1
        
        # Draw lines
        for line in lines:
            if isinstance(line, dict) and "start" in line and "end" in line:
                start_x = int(line["start"]["x"])
                start_y = int(line["start"]["y"])
                end_x = int(line["end"]["x"])
                end_y = int(line["end"]["y"])
                
                cv2.line(result_frame, (start_x, start_y), (end_x, end_y), line_color, thickness)
        
        return result_frame
    
    def analyze_frame(self, frame, detection_lines=None):
        """
        Complete traffic light analysis for a frame
        
        Args:
            frame: Input video frame
            detection_lines: Optional detection lines for violation checking
            
        Returns:
            Tuple of (annotated_frame, traffic_light_color, traffic_lights_info)
        """
        # Detect traffic lights
        traffic_lights = self.detect_traffic_lights(frame)
        
        annotated_frame = frame.copy()
        dominant_color = "unknown"
        
        if traffic_lights:
            # Use the traffic light with highest confidence
            best_light = max(traffic_lights, key=lambda x: x['confidence'])
            annotated_frame, dominant_color = self.determine_color(
                annotated_frame, best_light['bbox'], best_light['confidence']
            )
        
        # Draw detection lines if provided
        if detection_lines:
            annotated_frame = self.draw_colored_lines(annotated_frame, dominant_color, detection_lines)
        
        return annotated_frame, dominant_color, traffic_lights


# Utility functions for backward compatibility
def count_traffic_lights(frame):
    """Backward compatible function for counting traffic lights"""
    detector = TrafficLightDetector()
    return detector.count_traffic_lights(frame)


def determine_color(frame, bbox, conf):
    """Backward compatible function for color determination"""
    detector = TrafficLightDetector()
    return detector.determine_color(frame, bbox, conf)


def draw_colored_lines(frame, color_name, lines):
    """Backward compatible function for drawing colored lines"""
    detector = TrafficLightDetector()
    return detector.draw_colored_lines(frame, color_name, lines)


# Test function
if __name__ == "__main__":
    import os
    
    # Test the traffic light detector
    detector = TrafficLightDetector()
    
    # Test with webcam or video file
    video_source = 0  # Use webcam
    # video_source = "Videos/traffic1.mp4"  # Or use video file
    
    cap = cv2.VideoCapture(video_source)
    
    print("Traffic Light Detection Test")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze frame
        annotated_frame, color, lights_info = detector.analyze_frame(frame)
        
        # Display results
        cv2.putText(annotated_frame, f"Traffic Light: {color.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Count: {len(lights_info)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Traffic Light Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

