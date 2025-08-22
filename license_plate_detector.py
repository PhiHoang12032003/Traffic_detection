import cv2
import numpy as np
import easyocr
import os
from PIL import Image


class LicensePlateDetector:
    """
    License Plate Detection and Recognition System
    Uses Haar Cascade for detection and EasyOCR for text recognition
    """
    
    def __init__(self):
        """Initialize the license plate detector"""
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['vi', 'en'])  # Vietnamese and English
        
        # Try to load Haar Cascade for license plate detection
        cascade_path = 'haarcascade_plate_number.xml'
        if os.path.exists(cascade_path):
            self.license_plate_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            print(f"Warning: {cascade_path} not found. Using alternative detection method.")
            self.license_plate_cascade = None
        
        # Processed plates cache to avoid duplicates
        self.processed_plates = []
    
    def extract_license_plate_haar(self, frame, vehicle_region):
        """
        Extract license plate using Haar Cascade
        
        Args:
            frame: Original frame
            vehicle_region: Cropped vehicle image
            
        Returns:
            Tuple of (annotated_frame, license_plate_images)
        """
        result_frame = frame.copy()
        license_plate_images = []
        
        if self.license_plate_cascade is None:
            return result_frame, license_plate_images
        
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Morphological operations to improve detection
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=1)
        
        # Find contours to crop the region
        non_black_points = cv2.findNonZero(gray)
        if non_black_points is not None:
            x, y, w, h = cv2.boundingRect(non_black_points)
            w = int(w * 0.7)  # Adjust width
            
            cropped_gray = gray[y:y+h, x:x+w]
            
            # Detect license plates
            license_plates = self.license_plate_cascade.detectMultiScale(
                cropped_gray, 
                scaleFactor=1.07, 
                minNeighbors=15, 
                minSize=(20, 20)
            )
            
            # Process detected plates
            for (x_plate, y_plate, w_plate, h_plate) in license_plates:
                # Draw rectangle on result frame
                cv2.rectangle(result_frame, 
                            (x_plate + x, y_plate + y), 
                            (x_plate + x + w_plate, y_plate + y + h_plate), 
                            (0, 255, 0), 3)
                
                # Extract license plate image
                license_plate_image = cropped_gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
                license_plate_images.append(license_plate_image)
        
        return result_frame, license_plate_images
    
    def extract_license_plate_roi(self, frame, bbox):
        """
        Extract license plate using ROI-based approach
        
        Args:
            frame: Input frame
            bbox: Vehicle bounding box (x1, y1, x2, y2)
            
        Returns:
            License plate region or None
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop vehicle region
        vehicle_crop = frame[y1:y2, x1:x2]
        
        if vehicle_crop.size == 0:
            return None
        
        # Focus on lower part where license plate usually is
        h, w = vehicle_crop.shape[:2]
        
        # License plate is typically in the lower 40% of the vehicle
        license_region = vehicle_crop[int(h*0.6):, :]
        
        return license_region
    
    def apply_ocr_to_image(self, image_array):
        """
        Apply OCR to extract text from license plate image
        
        Args:
            image_array: License plate image as numpy array
            
        Returns:
            Recognized text or empty string
        """
        if image_array is None or image_array.size == 0:
            return ""
        
        try:
            # Convert to RGB if it's a color image
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image = image_array.copy()
            
            # Resize for better OCR accuracy
            height, width = image.shape[:2]
            if height < 50 or width < 100:
                # Scale up small images
                scale_factor = max(2, 100 // width, 50 // height)
                image = cv2.resize(image, (width * scale_factor, height * scale_factor))
            
            # Apply OCR
            results = self.reader.readtext(image)
            
            if not results:
                return ""
            
            recognized_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter by confidence
                    # Clean the text
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    if len(cleaned_text) >= 6:  # Valid license plate length
                        recognized_texts.append(cleaned_text)
            
            # Return the longest recognized text
            if recognized_texts:
                final_text = max(recognized_texts, key=len)
                return final_text.upper()
            
            return ""
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def process_license_plate(self, frame, bbox, vehicle_type):
        """
        Complete license plate processing pipeline
        
        Args:
            frame: Input frame
            bbox: Vehicle bounding box
            vehicle_type: Type of vehicle
            
        Returns:
            Recognized license plate text or None
        """
        try:
            x1, y1, x2, y2 = bbox
        except ValueError as e:
            print(f"Invalid bbox format: {e}")
            return None
        
        # Extract vehicle image
        vehicle_img = frame[int(y1):int(y2), int(x1):int(x2)]
        if vehicle_img.size == 0:
            return None
        
        # Method 1: Try Haar Cascade if available
        if self.license_plate_cascade is not None:
            result_frame, plate_images = self.extract_license_plate_haar(frame, vehicle_img)
            
            if plate_images:
                for plate_img in plate_images:
                    plate_text = self.apply_ocr_to_image(plate_img)
                    if plate_text and len(plate_text) >= 6:
                        return plate_text
        
        # Method 2: ROI-based approach
        license_region = self.extract_license_plate_roi(frame, bbox)
        if license_region is not None:
            plate_text = self.apply_ocr_to_image(license_region)
            if plate_text and len(plate_text) >= 6:
                return plate_text
        
        return None
    
    def validate_license_plate(self, plate_text):
        """
        Validate Vietnamese license plate format
        
        Args:
            plate_text: Detected license plate text
            
        Returns:
            True if valid format, False otherwise
        """
        if not plate_text or len(plate_text) < 6:
            return False
        
        # Remove spaces and convert to uppercase
        plate_text = plate_text.replace(" ", "").upper()
        
        # Basic Vietnamese license plate patterns
        # Format examples: 30A12345, 51G12345, etc.
        patterns = [
            r'^[0-9]{2}[A-Z]{1}[0-9]{4,5}$',  # 2 digits + 1 letter + 4-5 digits
            r'^[0-9]{2}[A-Z]{2}[0-9]{4,5}$', # 2 digits + 2 letters + 4-5 digits
        ]
        
        import re
        for pattern in patterns:
            if re.match(pattern, plate_text):
                return True
        
        return False
    
    def enhance_license_plate_image(self, image):
        """
        Enhance license plate image for better OCR
        
        Args:
            image: Input license plate image
            
        Returns:
            Enhanced image
        """
        if image is None or image.size == 0:
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned


# Utility functions for backward compatibility
def extract_license_plate(frame, mask_line):
    """Backward compatible function for license plate extraction"""
    detector = LicensePlateDetector()
    return detector.extract_license_plate_haar(frame, mask_line)


def apply_ocr_to_image(image_array):
    """Backward compatible function for OCR"""
    detector = LicensePlateDetector()
    return detector.apply_ocr_to_image(image_array)


# Test function
if __name__ == "__main__":
    # Test the license plate detector
    detector = LicensePlateDetector()
    
    # Test with a sample image
    test_image_path = "data_xe_may_vi_pham/1.jpg"
    
    if os.path.exists(test_image_path):
        print("Testing license plate detection...")
        
        frame = cv2.imread(test_image_path)
        if frame is not None:
            h, w = frame.shape[:2]
            
            # Use the whole image as vehicle region for testing
            bbox = (0, 0, w, h)
            
            plate_text = detector.process_license_plate(frame, bbox, "motorcycle")
            
            if plate_text:
                print(f"Detected license plate: {plate_text}")
                print(f"Valid format: {detector.validate_license_plate(plate_text)}")
            else:
                print("No license plate detected")
            
            cv2.imshow("Test Image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Could not load image: {test_image_path}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Available test images:")
        for folder in ["data_xe_may_vi_pham", "data_oto_vi_pham", "data_xe_vp_bh"]:
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
                if files:
                    print(f"  {folder}: {files[:3]}...")  # Show first 3 files

