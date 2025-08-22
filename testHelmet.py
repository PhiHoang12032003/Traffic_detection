# Object Detecion

import math
import tempfile

import numpy as np
from PIL import Image
import easyocr

import createBB_helmet
# from sort import Sort  # Disabled due to filterpy dependency issue
from testLane import *

# Initialize EasyOCR for license plate recognition
reader = easyocr.Reader(['vi', 'en'])


# basics
# Display image and videos


# plots

# %matplotlib inline
# Video  path for experiment

def risize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized

    # --------------------------------------------------------------


def extract_license_plate(frame, bbox):
    """Fast license plate extraction - simplified for speed"""
    x1, y1, x2, y2 = map(int, bbox)

    # Check if vehicle is too small for reliable OCR (relaxed thresholds for distant vehicles)
    vehicle_width = x2 - x1
    vehicle_height = y2 - y1
    if vehicle_width < 50 or vehicle_height < 40:
        print(f"Vehicle too small for OCR: {vehicle_width}x{vehicle_height}")
        return None, None

    # Expanded crop - bottom 40% of vehicle for distant plates
    crop_h = y2 - y1
    bottom_start = y1 + int(crop_h * 0.6)  # Bottom 40%
    
    # Ensure valid crop
    bottom_start = max(y1, bottom_start)
    bottom_start = min(y2 - 10, bottom_start)  # At least 10 pixels
    
    if bottom_start >= y2:
        return None, None
    
    # Extract bottom region only
    license_region = frame[bottom_start:y2, x1:x2]
    
    if license_region.size == 0:
        return None, None
    
    try:
        # Enhance image for better OCR on small/distant plates (3x zoom for far vehicles)
        enhanced_region = cv2.resize(license_region, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        
        # Apply additional preprocessing for distant plates
        # 1. Convert to grayscale for better OCR
        if len(enhanced_region.shape) == 3:
            gray_enhanced = cv2.cvtColor(enhanced_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_enhanced = enhanced_region
        
        # 2. Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray_enhanced)
        
        # 3. Apply sharpening filter
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1], 
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        
        # Use the preprocessed image for OCR
        final_region = sharpened
        
        # Single, fast OCR call on preprocessed image
        results = reader.readtext(final_region, paragraph=False)
        print(f"OCR found {len(results)} text regions in license area (enhanced 3x + preprocessing)")
        
        for (bbox_ocr, text, prob) in results:
            print(f"OCR result: '{text}' confidence: {prob:.3f}")
            if prob > 0.02:  # Ultra low threshold to catch more text
                # Clean text
                cleaned_text = ''.join(c for c in text if c.isalnum())
                
                # Very flexible validation - at least 3 chars with letters OR numbers
                if len(cleaned_text) >= 3:
                    has_letters = any(c.isalpha() for c in cleaned_text)
                    has_numbers = any(c.isdigit() for c in cleaned_text)
                    
                    # Accept any text with letters or numbers (more flexible)
                    if has_letters or has_numbers:
                        print(f"Found text candidate: {cleaned_text} (conf: {prob:.3f})")
                        
                        # Calculate license plate bounding box in original frame coordinates
                        # bbox_ocr is a list of 4 points, we need to get the bounding rectangle
                        ocr_points = bbox_ocr
                        if len(ocr_points) == 4:
                            # Get min/max coordinates from the 4 corner points
                            x_coords = [p[0] for p in ocr_points]
                            y_coords = [p[1] for p in ocr_points]
                            ocr_x1, ocr_x2 = min(x_coords), max(x_coords)
                            ocr_y1, ocr_y2 = min(y_coords), max(y_coords)
                            
                            # Convert OCR bbox from enhanced image back to original frame coordinates
                            # OCR was done on 3x enhanced image, so divide by 3
                            lp_x1 = int(x1 + ocr_x1 / 3.0)
                            lp_y1 = int(bottom_start + ocr_y1 / 3.0)
                            lp_x2 = int(x1 + ocr_x2 / 3.0)
                            lp_y2 = int(bottom_start + ocr_y2 / 3.0)
                            
                            # Ensure coordinates are within frame bounds
                            frame_h, frame_w = frame.shape[:2]
                            lp_x1 = max(0, min(lp_x1, frame_w))
                            lp_x2 = max(0, min(lp_x2, frame_w))
                            lp_y1 = max(0, min(lp_y1, frame_h))
                            lp_y2 = max(0, min(lp_y2, frame_h))
                            
                            # Validate license plate size (should be reasonable)
                            lp_width = lp_x2 - lp_x1
                            lp_height = lp_y2 - lp_y1
                            if lp_width > 10 and lp_height > 5 and lp_width < 300 and lp_height < 100:
                                license_bbox = [lp_x1, lp_y1, lp_x2, lp_y2]
                                print(f"Valid license bbox: {license_bbox} (size: {lp_width}x{lp_height})")
                                return cleaned_text.upper(), license_bbox
                            else:
                                print(f"Invalid license size: {lp_width}x{lp_height}, skipping")
                        
    except Exception:
        pass
    
    return None, None


def video_detect_helmet(path_x):
    cap = cv2.VideoCapture(path_x)  # For video
    examBB = createBB_helmet.infoObject()
    model = YOLO('model_helmet/helmet.pt')  # large model works better with the GPU
    vehicle_model = YOLO('best_new/vehicle.pt')  # Model for vehicle detection

    # tracking - disabled due to dependency issue
    # tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    dataBienBan_XEMAYVIPHAMBAOHIEM = 'BienBanNopPhatXeMayViPhamMuBaoHiem/'
    name_class = ["without helmet", "helmet"]
    array_helmet_filter = []
    count = 0
    frame_counter = 0
    # Removed violation cooldown for faster processing
    while True:

        success, frame = cap.read()
        if not success or frame is None:
            break
        frame_counter += 1
        
        # Run helmet detection every frame, vehicle detection every 3rd frame for speed
        results = model(frame, stream=True)
        
        detections = np.empty((0, 6))
        vehicles = []
        
        # Initialize license plate cache if not exists
        if not hasattr(video_detect_helmet, 'license_cache'):
            video_detect_helmet.license_cache = {}
        if not hasattr(video_detect_helmet, 'license_display_cache'):
            video_detect_helmet.license_display_cache = {}
        
        # Clean expired license plates from cache (keep for 60 frames = ~2 seconds)
        expired_frames = [k for k, v in video_detect_helmet.license_cache.items() 
                         if frame_counter - v['detected_frame'] > 60]
        for expired in expired_frames:
            del video_detect_helmet.license_cache[expired]
        
        # Clean expired license display cache (keep for 90 frames = ~3 seconds)
        expired_display = [k for k, v in video_detect_helmet.license_display_cache.items() 
                          if frame_counter - v['last_seen'] > 90]
        for expired in expired_display:
            del video_detect_helmet.license_display_cache[expired]
        
        # Only run vehicle detection every 5th frame to save processing time
        if frame_counter % 5 == 0:
            vehicle_results = vehicle_model(frame, stream=True)
            
            # Store vehicle detections and immediately detect license plates
            for v_result in vehicle_results:
                v_boxes = v_result.boxes
                if v_boxes is not None:
                    for v_box in v_boxes:
                        v_x1, v_y1, v_x2, v_y2 = v_box.xyxy[0]
                        v_x1, v_y1, v_x2, v_y2 = int(v_x1), int(v_y1), int(v_x2), int(v_y2)
                        v_conf = float(v_box.conf[0])
                        v_cls = int(v_box.cls[0])
                        
                        # Store vehicle info if confidence is good enough
                        # Lower confidence for distant vehicles, check size
                        vehicle_width = v_x2 - v_x1
                        vehicle_height = v_y2 - v_y1
                        vehicle_area = vehicle_width * vehicle_height
                        
                        # Accept vehicles based on size and confidence (relaxed for distant detection)
                        if v_conf > 0.4 and vehicle_area > 3000:  # Lower thresholds for distant vehicles
                            # Generate unique vehicle key based on position
                            vehicle_key = f"{v_x1}_{v_y1}_{v_x2}_{v_y2}"
                            
                            # Only try OCR on some vehicles to save time
                            license_plate = None
                            license_bbox = None
                            if len(vehicles) < 2:  # Limit to first 2 vehicles per frame
                                print(f"Attempting OCR on vehicle {len(vehicles)+1} at coords [{v_x1}, {v_y1}, {v_x2}, {v_y2}]")
                                license_plate, license_bbox = extract_license_plate(frame, [v_x1, v_y1, v_x2, v_y2])
                                
                                # Save to cache if found
                                if license_plate and license_bbox:
                                    video_detect_helmet.license_cache[vehicle_key] = {
                                        'license_plate': license_plate,
                                        'license_bbox': license_bbox,
                                        'detected_frame': frame_counter
                                    }
                                    # Also add to display cache for persistent display
                                    video_detect_helmet.license_display_cache[license_plate] = {
                                        'license_bbox': license_bbox,
                                        'frame_start': frame_counter,
                                        'last_seen': frame_counter
                                    }
                            
                            # Check cache for this vehicle
                            cached_info = video_detect_helmet.license_cache.get(vehicle_key)
                            if cached_info:
                                license_plate = cached_info['license_plate']
                                license_bbox = cached_info['license_bbox']
                                # Update display cache last_seen time
                                if license_plate in video_detect_helmet.license_display_cache:
                                    video_detect_helmet.license_display_cache[license_plate]['last_seen'] = frame_counter
                            
                            vehicles.append({
                                'bbox': [v_x1, v_y1, v_x2, v_y2],
                                'confidence': v_conf,
                                'class': v_cls,
                                'license_plate': license_plate,
                                'license_bbox': license_bbox
                            })
                            
                            # Print detection result only if OCR was attempted
                            if len(vehicles) <= 2:
                                if license_plate:
                                    print(f"Frame {frame_counter}: Vehicle detected with license plate: {license_plate}")
                                else:
                                    print(f"Frame {frame_counter}: Vehicle detected but no license plate found")
        # print("shape frame : ", frame.shape)
        for r in results:
            boxes = r.boxes
            name = r.names
            for box in boxes:
                # BBOX
                print(box)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                w, h = x2 - x1, y2 - y1
                bbox = (x1, y1, w, h)
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                currentClass = model.names[cls]
                print(currentClass)
                
                # Draw helmet detection (class 1)
                if cls == 1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (134, 255, 162), 2)
                    draw_text(frame, " Helmet", font_scale=0.5,
                              pos=(int(x1), int(y1)), text_color=(26, 93, 26),
                              text_color_bg=(208, 192, 79))
                
                # Add to detections if no helmet (class 0)
                if cls == 0:
                    currentArray = np.array([x1, y1, x2, y2, conf, cls])
                    detections = np.vstack((detections, currentArray))
        classes_array = detections[:, -1:]
        print("class_array", classes_array)
        # Tracking disabled due to dependency issue - process detections directly
        # Original tracking code commented out
        # resultsTracker = tracker.update(detections)
        
        # Process detections directly without tracking
        for detection in detections:
            x, y, w, h, conf, cls = detection
            x, y, w, h, cls = int(x), int(y), int(w), int(h), int(cls)

            text = name_class[cls] + ": " + str(round(conf, 2))
            
            # For helmet violations, try to find nearest vehicle's license plate
            license_plate = None
            if vehicles:
                # Find the closest vehicle to this helmet detection
                center_helmet_x = (x + w) // 2
                center_helmet_y = (y + h) // 2
                
                min_distance = float('inf')
                closest_vehicle = None
                
                for vehicle in vehicles:
                    vx1, vy1, vx2, vy2 = vehicle['bbox']
                    center_vehicle_x = (vx1 + vx2) // 2
                    center_vehicle_y = (vy1 + vy2) // 2
                    
                    # Quick distance calculation
                    distance = abs(center_helmet_x - center_vehicle_x) + abs(center_helmet_y - center_vehicle_y)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_vehicle = vehicle
                
                # Use license plate from closest vehicle if available
                if closest_vehicle and min_distance < 200:
                    license_plate = closest_vehicle.get('license_plate')
                    if license_plate:
                        text += f" | BS: {license_plate}"

            #####################################################################
            #####################################################################
            center_x = (x + w) // 2
            center_y = (y + h) // 2

            filterData = 0 <= center_x <= (int(frame.shape[1])) and int(
                3 * frame.shape[0] / 10) <= center_y <= int(
                4 * frame.shape[0] / 10)

            # xét vùng roi theo trục Y
            if 0 < center_x < int(frame.shape[1]) and int((2 * frame.shape[0]) / 10) < center_y < int(
                    (8 * frame.shape[0]) / 10):
                cv2.rectangle(frame, (x, y), (w, h), (36, 255, 12), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                if filterData:
                    draw_text(frame, text + " warning", font_scale=0.5,
                              pos=(int(x), int(y)),
                              text_color_bg=(0, 0, 0))
                    
                    # Fast violation processing - no duplicate check
                    count += 1
                    
                    # Simple image saving for violation
                    cropped_frame = frame[
                                    int((3 * frame.shape[0]) / 10):int((8 * frame.shape[0]) / 10),
                                    6 * int(frame.shape[1] / 10):int(frame.shape[1])]
                    imageViolateHelmet(frame, int((0 * frame.shape[0]) / 10),
                                       int((8 * frame.shape[0]) / 10), 0 * int(frame.shape[1] / 10),
                                       8 *
                                       int(frame.shape[1] / 10), count)

                    # Only create PDF every 10th violation for speed
                    if count % 10 == 1:
                        stt_BB_CTB = dataBienBan_XEMAYVIPHAMBAOHIEM + str(count) + '.pdf'
                        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                        frame_pil.save(temp_image.name)
                        
                        # Update with license plate info
                        if license_plate:
                            examBB['license_plate'] = license_plate
                        
                        createBB_helmet.bienBanNopPhat(examBB,
                                                       temp_image.name,
                                                       "data_xe_vp_bh/" + str(
                                                           count) + '.jpg',
                                                       stt_BB_CTB)
                        temp_image.close()
                    
                    license_info = f" - BS: {license_plate}" if license_plate else ""
                    print(f"Violation #{count} at frame {frame_counter}{license_info}")
                else:
                    draw_text(frame, text, font_scale=0.5,
                              pos=(int(x), int(y)),
                              text_color_bg=(78, 235, 133))
                print("count : ", count)
        start_point = (0, int((2 * frame.shape[0]) / 10))
        # vẽ hết chiều rộng và chiểu cao lấy 9/10
        end_point = (int(frame.shape[1]), int((8 * frame.shape[0]) / 10))
        color = (255, 0, 0)
        image = draw_text(frame, "So luong vi pham : " + str(count), font_scale=1.5,
                          pos=(int(0), int(0)),
                          text_color_bg=(255, 255, 255))

        # vẽ ra cái ROI
        image = cv2.rectangle(frame, start_point, end_point, color, 2)
        
        # Draw license plate detection boxes from display cache (persistent display)
        for license_text, info in video_detect_helmet.license_display_cache.items():
            bbox = info['license_bbox']
            frames_since_detection = frame_counter - info['frame_start']
            
            # Draw license plate bounding box with thick, bright green border
            lp_x1, lp_y1, lp_x2, lp_y2 = bbox
            cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 255, 0), 5)  # Very thick green box
            
            # Add license plate text with background for better visibility
            text_size = cv2.getTextSize(license_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = lp_x1
            text_y = lp_y1 - 10
            
            # Ensure text is within frame bounds
            if text_y < 25:
                text_y = lp_y2 + 30
            if text_x + text_size[0] > frame.shape[1]:
                text_x = frame.shape[1] - text_size[0] - 5
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), (0, 255, 0), -1)
            
            # Draw license plate text in black for contrast
            cv2.putText(frame, license_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            
            # Add detection duration indicator
            duration_text = f"({frames_since_detection}f)"
            cv2.putText(frame, duration_text, (text_x, text_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw vehicle detections with license plates (smaller, secondary display)
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 255, 0), 2)  # Yellow for vehicles
            
            # Display vehicle info with license plate
            vehicle_text = f"Vehicle: {vehicle['confidence']:.2f}"
            if vehicle.get('license_plate'):
                vehicle_text += f" | {vehicle['license_plate']}"
            
            draw_text(frame, vehicle_text, font_scale=0.4,
                      pos=(vx1, vy1-15), text_color=(255, 255, 0), text_color_bg=(0, 0, 0))
        
        # cv2.imshow("Roi ", image)
        # cv2.waitKey(1)
        yield image


if __name__ == '__main__':
    video_detect_helmet("Videos/test9.mp4")
