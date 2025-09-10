import re
import threading
from time import sleep

import cv2
import numpy as np
from config import OCR_CONFIG


class OCR_thread(threading.Thread):

    def __init__(self, violating_boxes_pipeline,violating_plates_text_pipeline, reader):
        threading.Thread.__init__(self, daemon=True)

        self.violating_boxes_pipeline = violating_boxes_pipeline
        self.violating_plates_text_pipeline = violating_plates_text_pipeline
        self.reader = reader

        self.stopped = False

    def preprocess_plate_image(self, plate_cropped):
        """Enhanced preprocessing for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_cropped, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Try different preprocessing methods
        methods = [
            adaptive_thresh,
            cleaned,
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
            # Add Gaussian blur method
            cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        ]
        
        return methods

    def clean_plate_text(self, text):
        """Clean and normalize license plate text"""
        # Remove extra spaces and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-alphanumeric characters except spaces
        text = re.sub(r'[^A-Z0-9\s]', '', text.upper())
        
        return text

    def apply_ocr_corrections(self, text):
        """Apply OCR corrections based on known patterns"""
        # Check if text matches any known OCR errors
        if text in OCR_CONFIG['ocr_corrections']:
            corrected = OCR_CONFIG['ocr_corrections'][text]
            print(f"OCR correction applied: '{text}' -> '{corrected}'")
            return corrected
        return text

    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two strings using Levenshtein distance"""
        if len(text1) < len(text2):
            return self.calculate_similarity(text2, text1)
        
        if len(text2) == 0:
            return len(text1)
        
        previous_row = list(range(len(text2) + 1))
        for i, c1 in enumerate(text1):
            current_row = [i + 1]
            for j, c2 in enumerate(text2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        distance = previous_row[-1]
        max_len = max(len(text1), len(text2))
        return 1 - (distance / max_len) if max_len > 0 else 1

    def is_similar_to_target(self, text):
        """Check if text is similar to any target plate"""
        for target in OCR_CONFIG['target_plates']:
            similarity = self.calculate_similarity(text, target)
            if similarity >= OCR_CONFIG['similarity_threshold']:
                print(f"Fuzzy match found: '{text}' similar to '{target}' (similarity: {similarity:.2f})")
                return True, target
        return False, None

    def is_valid_plate_format(self, text):
        """Check if text matches Vietnamese license plate format"""
        # Use patterns from configuration
        for pattern in OCR_CONFIG['patterns']:
            if re.match(pattern, text):
                return True
        
        # Check fuzzy patterns
        for pattern in OCR_CONFIG['fuzzy_patterns']:
            if re.match(pattern, text):
                return True
        
        # Check if it's similar to target plates
        is_similar, target = self.is_similar_to_target(text)
        if is_similar:
            return True
            
        return False

    def run(self):

        while not self.stopped:

            plate_cropped, box_info = self.violating_boxes_pipeline.get_message(block=True)

            # Try multiple preprocessing methods
            preprocessed_images = self.preprocess_plate_image(plate_cropped)
            
            best_result = None
            best_confidence = 0
            best_target = None
            
            for preprocessed_img in preprocessed_images:
                try:
                    res = self.reader.readtext(preprocessed_img)
                    
                    for (bbox, text, prob) in res:
                        # Clean the text
                        cleaned_text = self.clean_plate_text(text)
                        
                        # Apply OCR corrections first
                        corrected_text = self.apply_ocr_corrections(cleaned_text)
                        
                        # Check if it's a valid plate format
                        if self.is_valid_plate_format(corrected_text):
                            # Check if it's similar to target plates
                            is_similar, target = self.is_similar_to_target(corrected_text)
                            
                            if prob > best_confidence:
                                best_result = corrected_text
                                best_confidence = prob
                                best_target = target if is_similar else None
                            
                            # If it's a target plate, use the corrected version
                            if is_similar and target:
                                best_result = target
                            
                except Exception as e:
                    print(f"OCR error: {e}")
                    continue

            # Use confidence threshold from configuration
            if best_result and best_confidence > OCR_CONFIG['confidence_threshold']:
                final_result = best_target if best_target else best_result
                print(f"Detected plate: {best_result} -> {final_result} with confidence: {best_confidence:.2f}")
                self.violating_plates_text_pipeline.set_message((final_result, box_info))

            sleep(0.03)

    def stop(self):
        self.stopped = True
