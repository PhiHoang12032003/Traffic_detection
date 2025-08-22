import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from datetime import datetime
import os
import time
import json
import uuid
import tempfile
from PIL import Image
import createBB


def process_red_light_video_complete(input_path, output_path=None, use_improved_detection=True):
    """
    Process red light violation detection and save output video
    Similar to helmet detection processing
    
    Args:
        input_path: Path to input video
        output_path: Path for output video (auto-generated if None)  
        use_improved_detection: Use advanced detection with license plate recognition
        
    Returns:
        Tuple of (output_path, statistics)
    """
    start_time = time.time()
    
    # Initialize models
    model = YOLO('YoloWeights/yolov8n.pt')  # General YOLO for traffic lights
    vehicle_model = YOLO('best_new/vehicle.pt')  # Vehicle specific model
    
    # Initialize EasyOCR for license plates if using improved detection
    if use_improved_detection:
        try:
            reader = easyocr.Reader(['vi', 'en'])
        except Exception as e:
            print(f"Warning: Could not initialize EasyOCR: {e}")
            reader = None
    else:
        reader = None
    
    # Create directories
    os.makedirs('data_vuot_den_do', exist_ok=True)
    os.makedirs('BienBanNopPhatVuotDenDo', exist_ok=True)
    os.makedirs('processed_videos', exist_ok=True)
    
    # Traffic light color detection parameters (HSV)
    color_ranges = {
        'red': [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))
        ],
        'yellow': [(np.array([15, 50, 50]), np.array([40, 255, 255]))],
        'green': [(np.array([40, 50, 50]), np.array([90, 255, 255]))]
    }
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Generate output path if not provided
    if output_path is None:
        job_id = str(uuid.uuid4())
        output_path = f"processed_videos/red_light_detection_{job_id}.mp4"
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Define detection lines (similar to original system)
    stop_line_y = int(height * 0.65)
    
    # Create detection lines structure - low diagonal line like in the image
    detection_lines = [
        {
            "start": {"x": int(width * 0.0), "y": int(height * 0.85)},   # Low left
            "end": {"x": int(width * 0.6), "y": int(height * 0.90)}      # Low right (almost bottom)
        }
    ]
    
    # Statistics
    statistics = {
        'total_violations': 0,
        'vehicle_types': {},
        'detected_plates': [],
        'processing_time': 0,
        'total_frames': total_frames,
        'detection_method': 'Advanced' if use_improved_detection else 'Basic'
    }
    
    # Processing variables
    processed_vehicles = set()
    violation_count = 0
    frame_count = 0
    
    # Traffic light simulation (like original system)
    current_light = "red"  # Start with red light (automatic simulation)
    frames_since_light_change = 0
    light_cycle_duration = 90  # Change light every 90 frames (~3 seconds at 30fps)
    
    # Traffic light simulation patterns (more red lights for demo)
    light_patterns = ["red", "red", "yellow", "green", "red", "red"]
    current_light_index = 0
    auto_simulate_lights = True  # Enable automatic light simulation
    
    # Initialize BB creator for fines
    if use_improved_detection:
        try:
            examBB = createBB.infoObject()
        except:
            examBB = None
    else:
        examBB = None
    
    print(f"Processing red light detection: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Total frames: {total_frames}")
    print(f"Detection method: {'Advanced' if use_improved_detection else 'Basic'}")
    
    def determine_traffic_light_color(frame, bbox):
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
        
        for color, ranges in color_ranges.items():
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
    
    def check_vehicle_crossed_line(frame, bbox, lines):
        """Check if vehicle COMPLETELY crossed diagonal detection line"""
        if not lines or not isinstance(lines, list) or not isinstance(lines[0], dict) or \
           "start" not in lines[0] or "end" not in lines[0]:
            return False

        x1, y1, x2, y2 = bbox  
        vehicle_center_x = int(x1 + (x2 - x1) // 2)
        vehicle_top = int(y1)        # Vehicle top
        vehicle_bottom = int(y2)     # Vehicle bottom

        for line in lines:
            try:
                x_start, y_start = line["start"]["x"], line["start"]["y"]
                x_end, y_end = line["end"]["x"], line["end"]["y"]
            except KeyError:
                continue

            # Original system logic: check if line intersects with vehicle bbox
            # Line equation from original: y_line_at_x = y_start + (y_end - y_start) * (vehicle_bottom[0] - x_start) // (x_end - x_start)
            if x_end != x_start:
                try:
                    # Calculate line y position at vehicle center x (using integer division like original)
                    y_line_at_x = y_start + (y_end - y_start) * (vehicle_center_x - x_start) // (x_end - x_start)
                    
                    # Original logic: check if line cuts through vehicle (between vehicle top and bottom)
                    if vehicle_top <= y_line_at_x <= vehicle_bottom:
                        return True
                except ZeroDivisionError:
                    continue

        return False
    
    def detect_vehicles_crossing_line(frame, results, lines, current_light, processed_vehicles, threshold=0.33):
        """Detect vehicles crossing line when light is red (improved tracking)"""
        violating_vehicles = []
        vehicle_classes = [0, 1, 2, 3]  # Vehicle classes in custom model
        
        # Only process violations when light is red
        if current_light != "red":
            return violating_vehicles
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls in vehicle_classes and conf > threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Create unique vehicle ID based on CENTER position (better tracking for same vehicle)
                        center_x = int((x1 + x2) // 2)
                        center_y = int((y1 + y2) // 2)
                        vehicle_id = f"{center_x//40}_{center_y//40}_{cls}"
                        
                        # Check if vehicle crossed the detection line AND not already processed
                        if check_vehicle_crossed_line(frame, (x1, y1, x2, y2), lines):
                            
                            # Only count as NEW violation if this vehicle hasn't been processed
                            if vehicle_id not in processed_vehicles:
                                processed_vehicles.add(vehicle_id)
                                
                                try:
                                    vehicle_type = vehicle_model.names[cls] if hasattr(vehicle_model, 'names') else f"vehicle_{cls}"
                                except:
                                    vehicle_type = f"vehicle_{cls}"
                                violating_vehicles.append(((x1, y1, x2, y2), vehicle_type))
        
        return violating_vehicles
    
    def draw_simulated_traffic_light(frame, current_light):
        """Draw simulated traffic light in top-right corner like the original system"""
        height, width = frame.shape[:2]
        
        # Traffic light position (top-right corner)
        light_width = 60
        light_height = 150
        light_x = width - light_width - 20
        light_y = 20
        
        # Draw traffic light background (black rectangle with border)
        cv2.rectangle(frame, (light_x, light_y), (light_x + light_width, light_y + light_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (light_x, light_y), (light_x + light_width, light_y + light_height), (255, 255, 255), 2)
        
        # Light circle parameters
        circle_radius = 18
        circle_x = light_x + light_width // 2
        
        # Red light position (top)
        red_y = light_y + 30
        # Yellow light position (middle)  
        yellow_y = light_y + 75
        # Green light position (bottom)
        green_y = light_y + 120
        
        # Draw all lights (inactive - gray)
        cv2.circle(frame, (circle_x, red_y), circle_radius, (64, 64, 64), -1)
        cv2.circle(frame, (circle_x, yellow_y), circle_radius, (64, 64, 64), -1)
        cv2.circle(frame, (circle_x, green_y), circle_radius, (64, 64, 64), -1)
        
        # Draw active light
        if current_light == "red":
            cv2.circle(frame, (circle_x, red_y), circle_radius, (0, 0, 255), -1)
            cv2.circle(frame, (circle_x, red_y), circle_radius, (255, 255, 255), 2)
        elif current_light == "yellow":
            cv2.circle(frame, (circle_x, yellow_y), circle_radius, (0, 255, 255), -1)
            cv2.circle(frame, (circle_x, yellow_y), circle_radius, (255, 255, 255), 2)
        elif current_light == "green":
            cv2.circle(frame, (circle_x, green_y), circle_radius, (0, 255, 0), -1)
            cv2.circle(frame, (circle_x, green_y), circle_radius, (255, 255, 255), 2)
        
        # Add light status text
        cv2.putText(frame, f"light: {current_light} (0.95)", 
                   (light_x - 50, light_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def extract_license_plate(frame, bbox, reader):
        """Extract license plate using approach from helmet detection system"""
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Quick size check
            vehicle_width = x2 - x1
            vehicle_height = y2 - y1
            if vehicle_width < 60 or vehicle_height < 50:
                return None

            # Try multiple crop regions to find license plate (from helmet system)
            crop_h = y2 - y1
            crop_w = x2 - x1
            
            # Search regions like in helmet system
            search_regions = [
                # Bottom 30% of vehicle (most common for cars - highest priority)
                (y1 + int(crop_h * 0.7), y2, x1, x2, "bottom_30"),
                # Bottom 50% as backup for different vehicle types
                (y1 + int(crop_h * 0.5), y2, x1, x2, "bottom_50")
            ]
            
            best_result = None
            best_score = 0
            
            for region_y1, region_y2, region_x1, region_x2, region_name in search_regions:
                # Ensure valid crop
                if region_y1 >= region_y2 - 5 or region_x1 >= region_x2 - 5:
                    continue
                    
                # Extract region
                license_region = frame[region_y1:region_y2, region_x1:region_x2]
                
                if license_region.size == 0 or license_region.shape[0] < 10 or license_region.shape[1] < 30:
                    continue
                
                # Fast preprocessing - like helmet system
                try:
                    # Simple resize + CLAHE (fastest effective method)
                    enhanced = cv2.resize(license_region, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_LINEAR)
                    
                    if len(enhanced.shape) == 3:
                        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                    
                    # Quick contrast enhancement
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
                    enhanced = clahe.apply(enhanced)
                    
                except:
                    # Fallback to simple resize
                    enhanced = cv2.resize(license_region, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_LINEAR)
                
                # Single OCR call
                if reader:
                    results = reader.readtext(enhanced, paragraph=False, width_ths=0.8, height_ths=0.8)
                    
                    # Find best result in this region
                    for (bbox_ocr, text, prob) in results:
                        if prob > 0.1:  # Even lower threshold to catch more candidates
                            cleaned_text = ''.join(c for c in text if c.isalnum())
                            
                            # More flexible validation for license plates
                            if len(cleaned_text) >= 4 and len(cleaned_text) <= 15:  # More flexible length
                                has_letters = any(c.isalpha() for c in cleaned_text)
                                has_numbers = any(c.isdigit() for c in cleaned_text)
                                
                                # Accept if has both letters and numbers OR just numbers (for some plate types)
                                if (has_letters and has_numbers) or (has_numbers and len(cleaned_text) >= 5):
                                    # Calculate position in enhanced image first
                                    if len(bbox_ocr) == 4:
                                        x_coords = [p[0] for p in bbox_ocr]
                                        y_coords = [p[1] for p in bbox_ocr]
                                        ocr_x1, ocr_x2 = min(x_coords), max(x_coords)
                                        ocr_y1, ocr_y2 = min(y_coords), max(y_coords)
                                        
                                        # Convert OCR coordinates back to original frame coordinates
                                        # OCR coordinates are in enhanced image (1.8x), need to scale down
                                        # Then add offset from region position in original frame
                                        ocr_scaled_x1 = ocr_x1 / 1.8
                                        ocr_scaled_y1 = ocr_y1 / 1.8
                                        ocr_scaled_x2 = ocr_x2 / 1.8
                                        ocr_scaled_y2 = ocr_y2 / 1.8
                                        
                                        # Add region offset to get final coordinates in original frame
                                        lp_x1 = int(region_x1 + ocr_scaled_x1)
                                        lp_y1 = int(region_y1 + ocr_scaled_y1)
                                        lp_x2 = int(region_x1 + ocr_scaled_x2)
                                        lp_y2 = int(region_y1 + ocr_scaled_y2)
                                        
                                        # Ensure coordinates are within frame bounds
                                        frame_h, frame_w = frame.shape[:2]
                                        lp_x1 = max(0, min(lp_x1, frame_w))
                                        lp_x2 = max(0, min(lp_x2, frame_w))
                                        lp_y1 = max(0, min(lp_y1, frame_h))
                                        lp_y2 = max(0, min(lp_y2, frame_h))
                                        
                                        # Validate license plate size and aspect ratio
                                        lp_width = lp_x2 - lp_x1
                                        lp_height = lp_y2 - lp_y1
                                        aspect_ratio = lp_width / max(lp_height, 1)
                                        
                                        # License plates typically have aspect ratio 2:1 to 4:1
                                        if (lp_width > 20 and lp_height > 8 and 
                                            lp_width < 200 and lp_height < 60 and
                                            1.5 <= aspect_ratio <= 5.0):
                                            
                                            # Score based on confidence, region type, and size
                                            region_bonus = {"bottom_30": 0.4, "bottom_50": 0.2}
                                            size_score = min(lp_width / 100, 1.0)  # Prefer reasonable sizes
                                            total_score = prob * 0.5 + region_bonus.get(region_name, 0) + size_score * 0.3
                                            
                                            if total_score > best_score:
                                                best_score = total_score
                                                best_result = (cleaned_text.upper(), [lp_x1, lp_y1, lp_x2, lp_y2], region_name)
                                                
                                                # Don't draw here - will draw later in main loop
                                                
                                                # Early stopping if we found a good result
                                                if total_score > 0.6:
                                                    break
                
                # Early stopping at region level too
                if best_result and best_score > 0.6:
                    break
            
            # Return best result found across all regions (like helmet system)
            if best_result:
                license_text, license_bbox, region_used = best_result
                return license_text, license_bbox  # Return both text and bbox
            else:
                return None, None
            
        except Exception as e:
            print(f"License plate extraction error: {e}")
            return None, None
    
    def save_violation(frame, bbox, vehicle_type, plate_text, violation_id):
        """Save violation data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save violation image
        filename = f"data_vuot_den_do/violation_{timestamp}_{violation_id}.jpg"
        cv2.imwrite(filename, frame)
        
        # Update statistics
        statistics['detected_plates'].append({
            'plate': plate_text,
            'vehicle_type': vehicle_type,
            'timestamp': timestamp,
            'image_path': filename
        })
        
        # Create fine document if license plate is detected and BB creator is available
        if plate_text and examBB:
            stt_BB = f'BienBanNopPhatVuotDenDo/{violation_id}.pdf'
            
            # Convert frame to PIL for document creation
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            frame_pil.save(temp_image.name)
            
            try:
                createBB.bienBanNopPhat(
                    examBB,
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
    
    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0 or frame_count == 1:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Auto simulate traffic lights (like original system)
        frames_since_light_change += 1
        
        # Change light color automatically every cycle
        if frames_since_light_change >= light_cycle_duration:
            frames_since_light_change = 0
            current_light_index = (current_light_index + 1) % len(light_patterns)
            current_light = light_patterns[current_light_index]
            print(f"Traffic light changed to: {current_light}")
            
            # Only clear processed vehicles when light changes FROM red to other colors
            # This prevents re-counting same vehicles during red light period
            if current_light != "red":
                # Clear old vehicles to allow fresh detection in next red light cycle
                if len(processed_vehicles) > 50:  # Limit memory usage
                    processed_vehicles.clear()
        
        # Try to detect real traffic lights first (every 10 frames for performance)
        detected_real_light = False
        if frame_count % 10 == 0:
            results = model(frame)
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
            
            if traffic_lights:
                # Use the traffic light with highest confidence
                best_light = max(traffic_lights, key=lambda x: x[4])
                detected_color = determine_traffic_light_color(frame, best_light[:4])
                if detected_color != "unknown":
                    current_light = detected_color
                    detected_real_light = True
                
                # Draw real traffic light detection
                x1, y1, x2, y2, conf = best_light
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"Den: {current_light} ({conf:.2f})", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw simulated traffic light in top-right corner (like in your image)
        if not detected_real_light or auto_simulate_lights:
            draw_simulated_traffic_light(frame, current_light)
        
        # Detect vehicles
        vehicle_results = vehicle_model(frame)
        
        # Check for violations using the improved method from original system
        violations = detect_vehicles_crossing_line(frame, vehicle_results, detection_lines, current_light, processed_vehicles)
        
        for bbox, vehicle_type in violations:
            x1, y1, x2, y2 = bbox
            
            # Extract license plate if using improved detection (like original system)
            plate_text = None
            if use_improved_detection:
                plate_text, _ = extract_license_plate(frame, (x1, y1, x2, y2), reader)  # Only need text for violations
            
            # Count ALL violations immediately (like original basic testRedLight.py)
            violation_count += 1
            statistics['total_violations'] += 1
            
            # Update vehicle type statistics
            if vehicle_type in statistics['vehicle_types']:
                statistics['vehicle_types'][vehicle_type] += 1
            else:
                statistics['vehicle_types'][vehicle_type] = 1
            
            # Save detailed info only if license plate detected
            if plate_text:
                save_violation(frame, (x1, y1, x2, y2), vehicle_type, plate_text, violation_count)
                print(f"Red light violation detected: {plate_text} ({vehicle_type})")
                
            # Draw ALL violations with red color
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"VI PHAM: {vehicle_type}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if plate_text:
                cv2.putText(frame, f"Bien so: {plate_text}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Khong nhan dien bien so", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            cv2.putText(frame, "VUOT DEN DO!", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Extract license plates for ALL vehicles (like helmet system)
        vehicles_with_plates = []
        for r in vehicle_results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls in [0, 1, 2, 3] and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Extract license plate for each vehicle (like helmet system approach)
                        license_text, license_bbox = extract_license_plate(frame, [x1, y1, x2, y2], reader)
                        
                        vehicle_data = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls
                        }
                        
                        # Store license plate info if found (debug log)
                        if license_text and license_bbox:
                            vehicle_data['license_plate'] = license_text
                            vehicle_data['license_bbox'] = license_bbox
                            if frame_count % 100 == 0:  # Log every 100 frames
                                print(f"DEBUG: License plate detected: {license_text} at frame {frame_count}")
                        elif frame_count % 100 == 0:  # Debug why no license plate
                            print(f"DEBUG: No license plate found for vehicle at frame {frame_count} - text: {license_text}, bbox: {license_bbox}")
                        
                        vehicles_with_plates.append(vehicle_data)
                        
                        # Check if this is a violation
                        vehicle_id = f"{x1}_{y1}_{x2}_{y2}"
                        if vehicle_id not in processed_vehicles:
                            # Normal vehicle - green box
                            color = (0, 255, 0) if current_light != "red" else (255, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Add vehicle type label
                            try:
                                vehicle_name = vehicle_model.names[cls] if hasattr(vehicle_model, 'names') else f"vehicle_{cls}"
                            except:
                                vehicle_name = f"vehicle_{cls}"
                            cv2.putText(frame, vehicle_name, 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw detection lines with color based on traffic light status (from original system)
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
        
        # Draw detection lines
        for line in detection_lines:
            start_x = int(line["start"]["x"])
            start_y = int(line["start"]["y"])
            end_x = int(line["end"]["x"])
            end_y = int(line["end"]["y"])
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), line_color, thickness)
        
        cv2.putText(frame, f"VACH DUNG - {current_light.upper()}", (10, detection_lines[0]["start"]["y"] - 10),
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
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw license plate detection boxes for ALL vehicles (like helmet system)
        plates_detected = 0
        for vehicle in vehicles_with_plates:
            if vehicle.get('license_plate') and vehicle.get('license_bbox'):
                license_text = vehicle['license_plate']
                license_bbox = vehicle['license_bbox']
                plates_detected += 1
                
                # Use actual license plate bbox from OCR detection
                lp_x1, lp_y1, lp_x2, lp_y2 = license_bbox
                
                # Validate bbox coordinates
                if lp_x2 <= lp_x1 or lp_y2 <= lp_y1:
                    continue  # Skip invalid boxes
                
                # Additional validation: license plate should be in reasonable position
                lp_center_y = (lp_y1 + lp_y2) // 2
                if lp_center_y < height * 0.3:  # Skip if in top 30% of frame
                    continue
                
                # Draw license plate bounding box in bright green (like helmet system)
                box_color = (0, 255, 0)  # Bright green for real-time detection
                cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), box_color, 4)
                
                # Add license plate text with background for better visibility
                text_size = cv2.getTextSize(license_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = lp_x1
                text_y = lp_y1 - 10
                
                # Ensure text is within frame bounds
                if text_y < 25:
                    text_y = lp_y2 + 25
                if text_x + text_size[0] > width:
                    text_x = width - text_size[0] - 5
                
                # Draw background rectangle for text
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                             (text_x + text_size[0] + 5, text_y + 5), box_color, -1)
                
                # Draw license plate text in black for contrast
                cv2.putText(frame, license_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Add real-time indicator (like helmet system)
                cv2.putText(frame, "LIVE", (text_x, text_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        # Add license plate count to information panel
        cv2.putText(frame, f"Bien so: {plates_detected}", (220, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Clean up old tracked vehicles periodically
        if frame_count % 100 == 0 and len(processed_vehicles) > 50:
            recent_vehicles = set(list(processed_vehicles)[-25:])
            processed_vehicles = recent_vehicles
    
    # Cleanup
    cap.release()
    out.release()
    
    # Calculate processing time
    processing_time = time.time() - start_time
    statistics['processing_time'] = processing_time
    
    print(f"\nProcessing completed!")
    print(f"Output saved to: {output_path}")
    print(f"Total violations detected: {violation_count}")
    print(f"Processing time: {processing_time:.2f} seconds")
    
    # Save statistics
    stats_path = output_path.replace('.mp4', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    return output_path, statistics


# Test function
if __name__ == "__main__":
    # Test with a video
    video_path = "Videos/traffic1.mp4"
    if os.path.exists(video_path):
        print("Starting red light detection processing...")
        output_path, stats = process_red_light_video_complete(video_path, use_improved_detection=True)
        print(f"Video processing completed. Output: {output_path}")
        print("Statistics:", json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print(f"Video file not found: {video_path}")
        print("Available videos:")
        if os.path.exists("Videos"):
            for f in os.listdir("Videos"):
                if f.endswith(('.mp4', '.avi', '.mov')):
                    print(f"  - {f}")
