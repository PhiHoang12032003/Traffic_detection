# Process entire video for helmet violations and save result
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from datetime import datetime
import os
from collections import defaultdict
import tempfile
from PIL import Image
import createBB_helmet
from testLane import *
import json

# Initialize EasyOCR for license plate recognition
reader = easyocr.Reader(['vi', 'en'])

def extract_license_plate_fast(frame, bbox):
    """Super fast license plate extraction - minimal processing to avoid hang"""
    try:
        x1, y1, x2, y2 = map(int, bbox)

        # Quick size check
        vehicle_width = x2 - x1
        vehicle_height = y2 - y1
        if vehicle_width < 60 or vehicle_height < 50:
            return None, None

        # Try multiple crop regions to find license plate
        crop_h = y2 - y1
        crop_w = x2 - x1
        
        # Fast approach: Only 2 most effective regions
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
            
            # Fast preprocessing - single best method only
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
            results = reader.readtext(enhanced, paragraph=False, width_ths=0.8, height_ths=0.8)
            
            # Debug disabled for speed
            # print(f"    Region {region_name}: Found {len(results)} text candidates")
            
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
                                        # print(f"    New best: {cleaned_text} in {region_name} (score: {total_score:.3f})")
                                        
                                        # Early stopping if we found a good result (lower threshold for speed)
                                        if total_score > 0.6:
                                            # print(f"    🚀 Early stop - good score!")
                                            break
            
            # Early stopping at region level too
            if best_result and best_score > 0.6:
                break
        
        # Return best result found across all regions
        if best_result:
            license_text, license_bbox, region_used = best_result
            # print(f"✅ Final choice: {license_text} from {region_used}")
            return license_text, license_bbox
                        
    except Exception as e:
        # Don't print errors to avoid spam
        pass
    
    return None, None

def extract_license_plate(frame, bbox, expand_ratio=0.2):
    """Extract license plate from vehicle bounding box"""
    x1, y1, x2, y2 = map(int, bbox)

    # Check if vehicle is too small for reliable OCR
    vehicle_width = x2 - x1
    vehicle_height = y2 - y1
    if vehicle_width < 80 or vehicle_height < 60:
        print(f"Vehicle too small for OCR: {vehicle_width}x{vehicle_height}")
        return None, None

    # Expand the bounding box slightly
    height = y2 - y1
    width = x2 - x1
    
    x1 = max(0, x1 - int(width * expand_ratio))
    x2 = min(frame.shape[1], x2 + int(width * expand_ratio))
    y1 = max(0, y1 - int(height * expand_ratio))
    y2 = min(frame.shape[0], y2 + int(height * expand_ratio))
    
    vehicle_crop = frame[y1:y2, x1:x2]
    
    # Focus on different regions for motorcycles
    h, w = vehicle_crop.shape[:2]
    license_regions = [
        vehicle_crop[int(h*0.5):, :],
        vehicle_crop[int(h*0.3):int(h*0.7), :],
        vehicle_crop
    ]
    
    best_text = None
    best_confidence = 0
    
    for region_idx, region in enumerate(license_regions):
        try:
            results = reader.readtext(region, paragraph=False)
            for (bbox_ocr, text, prob) in results:
                if prob > 0.1:  # Very low threshold to catch any text
                    # Clean text
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    
                    # Very flexible validation - at least 3 chars with letters OR numbers
                    if len(cleaned_text) >= 3:
                        has_letters = any(c.isalpha() for c in cleaned_text)
                        has_numbers = any(c.isdigit() for c in cleaned_text)
                        
                        # Accept any text with letters or numbers (more flexible)
                        if has_letters or has_numbers and prob > best_confidence:
                            print(f"Found license plate candidate: {cleaned_text} (conf: {prob:.3f}) in region {region_idx}")
                            
                            # Calculate license plate bounding box in original frame coordinates
                            ocr_points = bbox_ocr
                            if len(ocr_points) == 4:
                                # Get min/max coordinates from the 4 corner points
                                x_coords = [p[0] for p in ocr_points]
                                y_coords = [p[1] for p in ocr_points]
                                ocr_x1, ocr_x2 = min(x_coords), max(x_coords)
                                ocr_y1, ocr_y2 = min(y_coords), max(y_coords)
                                
                                # Adjust coordinates based on region
                                if region_idx == 0:  # Bottom half region
                                    ocr_y1 += int(vehicle_crop.shape[0] * 0.5)
                                    ocr_y2 += int(vehicle_crop.shape[0] * 0.5)
                                elif region_idx == 1:  # Middle region
                                    ocr_y1 += int(vehicle_crop.shape[0] * 0.3)
                                    ocr_y2 += int(vehicle_crop.shape[0] * 0.3)
                                
                                # Convert to original frame coordinates
                                lp_x1 = int(x1 + ocr_x1)
                                lp_y1 = int(y1 + ocr_y1)
                                lp_x2 = int(x1 + ocr_x2)
                                lp_y2 = int(y1 + ocr_y2)
                                
                                license_bbox = [lp_x1, lp_y1, lp_x2, lp_y2]
                                print(f"License bbox: {license_bbox}")
                                
                                return cleaned_text.upper(), license_bbox
        except:
            continue
    
    return None, None


def process_helmet_video_complete(input_path, output_path, use_improved_detection=False):
    """
    FAST HELMET DETECTION - Only YOLOv8 Custom (No YOLO12, Minimal OCR)
    Args:
        input_path: Path to input video
        output_path: Path to save processed video
        use_improved_detection: Ignored - always use fast detection
    Returns: Path to processed video and violation statistics
    """
    print(f"🚀 Starting FAST helmet detection: {input_path}")
    print("✅ Using YOLOv8 Custom only (No YOLO12, Minimal OCR)")
    
    # Load YOLOv8 Custom helmet model
    helmet_model = YOLO('model_helmet/helmet.pt')
    # Load vehicle model for license plate detection
    vehicle_model = YOLO('best_new/vehicle.pt')
    examBB = createBB_helmet.infoObject()
    
    # Create directories
    os.makedirs('data_xe_vp_bh', exist_ok=True)
    os.makedirs('BienBanNopPhatXeMayViPhamMuBaoHiem', exist_ok=True)
    os.makedirs('processed_videos', exist_ok=True)
    
    # Get video properties
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height}, {fps}fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize counters
    violation_count = 0
    violations_data = []
    frame_count = 0
    name_class = ["without helmet", "helmet"]
    
    # License plate detection toggle:
    # True = Helmet + License detection (slower but complete)
    # False = Helmet only (much faster - recommended for speed)
    ENABLE_LICENSE_DETECTION = True  # Set to False for speed, True for license plates
    
    if ENABLE_LICENSE_DETECTION:
        print("🔥 Processing with helmet detection + License plate recognition...")
    else:
        print("⚡ Processing with SPEED MODE - Helmet detection only (no license plates)...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Vehicle detection every 8th frame for speed (still good tracking)
            vehicles = []
            if frame_count % 8 == 0 and ENABLE_LICENSE_DETECTION:
                # Fast mode - reduced logging
                if frame_count % 100 == 0:  # Only log every 100th frame
                    print(f"🚗 Vehicle detection at frame {frame_count}")
                try:
                    vehicle_results = vehicle_model(frame, verbose=False)
                    
                    vehicle_count = 0
                    for v_result in vehicle_results:
                        v_boxes = v_result.boxes
                        if v_boxes is not None:
                            for v_box in v_boxes:
                                v_x1, v_y1, v_x2, v_y2 = v_box.xyxy[0]
                                v_x1, v_y1, v_x2, v_y2 = int(v_x1), int(v_y1), int(v_x2), int(v_y2)
                                v_conf = float(v_box.conf[0])
                                
                                if v_conf > 0.3:
                                    vehicle_width = v_x2 - v_x1
                                    vehicle_height = v_y2 - v_y1
                                    vehicle_area = vehicle_width * vehicle_height
                                    
                                    if vehicle_area > 2000:
                                        vehicle_count += 1
                                        # Reduced logging for speed
                                        # print(f"  Vehicle {vehicle_count}: {vehicle_width}x{vehicle_height} (conf: {v_conf:.2f})")
                                        
                                        # Try license plate detection with timeout protection
                                        license_plate = None
                                        license_bbox = None
                                        
                                        try:
                                            # Quick license plate detection
                                            license_plate, license_bbox = extract_license_plate_fast(frame, [v_x1, v_y1, v_x2, v_y2])
                                            if license_plate:
                                                print(f"    ✅ License found: {license_plate} at bbox: {license_bbox}")
                                                
                                                # Debug: Save crop with detected license for verification (less frequent)
                                                if frame_count % 200 == 0:  # Save every 200th frame for debugging
                                                    try:
                                                        debug_crop = frame[v_y1:v_y2, v_x1:v_x2]
                                                        os.makedirs('debug_license', exist_ok=True)
                                                        cv2.imwrite(f'debug_license/frame_{frame_count}_vehicle.jpg', debug_crop)
                                                        
                                                        if license_bbox:
                                                            lx1, ly1, lx2, ly2 = license_bbox
                                                            # Draw box on debug image
                                                            debug_with_box = debug_crop.copy()
                                                            # Adjust coordinates to crop coordinates
                                                            local_lx1 = max(0, lx1 - v_x1)
                                                            local_ly1 = max(0, ly1 - v_y1) 
                                                            local_lx2 = min(debug_crop.shape[1], lx2 - v_x1)
                                                            local_ly2 = min(debug_crop.shape[0], ly2 - v_y1)
                                                            cv2.rectangle(debug_with_box, (local_lx1, local_ly1), (local_lx2, local_ly2), (0, 255, 0), 2)
                                                            cv2.putText(debug_with_box, license_plate, (local_lx1, local_ly1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                                            cv2.imwrite(f'debug_license/frame_{frame_count}_license_{license_plate}.jpg', debug_with_box)
                                                            print(f"    📷 Debug image saved: frame_{frame_count}_license_{license_plate}.jpg")
                                                    except:
                                                        pass
                                            else:
                                                print(f"    ❌ No license found for vehicle {vehicle_count}")
                                        except Exception as e:
                                            print(f"    ❌ License detection error: {e}")
                                            license_plate = None
                                            license_bbox = None
                                        
                                        # No caching - just store in current vehicles list for this frame
                                        
                                        vehicles.append({
                                            'bbox': [v_x1, v_y1, v_x2, v_y2],
                                            'confidence': v_conf,
                                            'license_plate': license_plate,
                                            'license_bbox': license_bbox
                                        })
                except Exception as e:
                    print(f"❌ Vehicle detection error at frame {frame_count}: {e}")
                    vehicles = []
            
            # No cache cleaning needed - using real-time detection only
            
            # YOLOv8 Custom helmet detection (every frame for accuracy)
            results = helmet_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])  # 0=no helmet, 1=helmet
                        
                        if conf > 0.6:  # High confidence only
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Check detection zone
                            in_zone = (0 < center_x < width and 
                                      int((2 * height) / 10) < center_y < int((8 * height) / 10))
                            
                            if in_zone:
                                if cls == 0:  # No helmet - VIOLATION
                                    violation_count += 1
                                    
                                    # Draw violation
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                                    
                                    # Simple text (no license plate to save time)
                                    violation_text = f"VIOLATION: {name_class[cls]} ({conf:.2f})"
                                    cv2.putText(frame, violation_text, (x1, y1-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    
                                    # Find nearest vehicle's license plate BEFORE saving evidence
                                    license_text = None
                                    license_bbox = None
                                    min_distance = float('inf')
                                    closest_vehicle_bbox = None
                                    
                                    # Check current frame vehicles first
                                    for vehicle in vehicles:
                                        vx1, vy1, vx2, vy2 = vehicle['bbox']
                                        v_center_x = (vx1 + vx2) // 2
                                        v_center_y = (vy1 + vy2) // 2
                                        distance = abs(center_x - v_center_x) + abs(center_y - v_center_y)
                                        
                                        if distance < min_distance and distance < 200:
                                            min_distance = distance
                                            closest_vehicle_bbox = [vx1, vy1, vx2, vy2]
                                            if vehicle.get('license_plate'):
                                                license_text = vehicle['license_plate']
                                                license_bbox = vehicle['license_bbox']
                                    
                                    # No cache search - only use current frame detection
                                    
                                    # Save evidence with license plate info
                                    if violation_count % 5 == 1:  # Every 5th violation for balance
                                        try:
                                            # Save violation image
                                            imageViolateHelmet(frame, 0, height, 0, width, violation_count)
                                            
                                            # Create PDF with license info
                                            stt_BB_CTB = f'BienBanNopPhatXeMayViPhamMuBaoHiem/{violation_count}.pdf'
                                            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                            temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                                            frame_pil.save(temp_image.name)
                                            
                                            createBB_helmet.bienBanNopPhat(examBB, temp_image.name,
                                                                           f"data_xe_vp_bh/{violation_count}.jpg", stt_BB_CTB)
                                            temp_image.close()
                                        except Exception as e:
                                            print(f"Error saving evidence: {e}")
                                    
                                    # Add to violations data with license plate info
                                    violation_data = {
                                        'frame': frame_count,
                                        'time': frame_count / fps,
                                        'confidence': conf,
                                        'bbox': [x1, y1, x2, y2]
                                    }
                                    
                                    # Add license plate info if found
                                    if license_text:
                                        violation_data['license_plate'] = license_text
                                        violation_data['license_bbox'] = license_bbox
                                    
                                    violations_data.append(violation_data)
                                    
                                    license_info = f" - BS: {license_text}" if license_text else ""
                                    print(f"🚨 Violation #{violation_count} at frame {frame_count} (conf: {conf:.2f}){license_info}")
                                    if license_text:
                                        print(f"🚗 License plate detected: {license_text}")
                                
                                else:  # Helmet detected - OK
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"OK: {name_class[cls]} ({conf:.2f})", (x1, y1-10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw license plate detection boxes for current frame vehicles only
            for vehicle in vehicles:
                if vehicle.get('license_plate') and vehicle.get('license_bbox'):
                    license_text = vehicle['license_plate']
                    bbox = vehicle['license_bbox']
                    
                    # Validate bbox coordinates
                    lp_x1, lp_y1, lp_x2, lp_y2 = bbox
                    if lp_x2 <= lp_x1 or lp_y2 <= lp_y1:
                        continue  # Skip invalid boxes
                    
                    # Additional validation: license plate should be in reasonable position
                    lp_center_y = (lp_y1 + lp_y2) // 2
                    if lp_center_y < height * 0.3:  # Skip if in top 30% of frame
                        continue
                    
                    # Draw license plate bounding box in bright green
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
                    
                    # Add real-time indicator
                    cv2.putText(frame, "LIVE", (text_x, text_y + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            
            # Draw detection zone
            cv2.rectangle(frame, (0, int((2 * height) / 10)), (width, int((8 * height) / 10)), (255, 0, 0), 2)
            
            # Display fast statistics
            if ENABLE_LICENSE_DETECTION:
                current_plates = len([v for v in vehicles if v.get('license_plate')])
                cv2.putText(frame, f"HELMET+LICENSE Mode - Violations: {violation_count} | Plates: {current_plates} | Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"SPEED Mode - Helmet Violations: {violation_count} | Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"Progress: {(frame_count/total_frames)*100:.1f}%", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Show progress less frequently for speed
            if frame_count % 300 == 0:  # Every 300 frames instead of 100
                progress = (frame_count / total_frames) * 100
                if ENABLE_LICENSE_DETECTION:
                    current_plates = len([v for v in vehicles if v.get('license_plate')])
                    print(f"⚡ HELMET+LICENSE Processing: {frame_count}/{total_frames} ({progress:.1f}%) - {violation_count} violations - {current_plates} plates")
                else:
                    print(f"⚡ SPEED Processing: {frame_count}/{total_frames} ({progress:.1f}%) - {violation_count} helmet violations")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        
    finally:
        cap.release()
        out.release()
    
    print(f"✅ FAST processing complete! {frame_count} frames, {violation_count} violations")
    
    # Create statistics
    stats = {
        'total_violations': violation_count,
        'violations': violations_data,
        'processed_frames': frame_count,
        'fps': fps,
        'duration': frame_count / fps if fps > 0 else 0,
        'processing_mode': 'SPEED_HELMET_ONLY' if not ENABLE_LICENSE_DETECTION else 'HELMET_PLUS_LICENSE'
    }
    
    # Save stats
    stats_path = output_path.replace('.mp4', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return output_path, stats


if __name__ == "__main__":
    # Test the function
    input_video = "Videos/test9.mp4"
    output_video = "processed_videos/test9_processed.mp4"
    
    result_path, stats = process_helmet_video_complete(input_video, output_video)
    print(f"Processed video saved to: {result_path}")
    print(f"Statistics: {stats}")

