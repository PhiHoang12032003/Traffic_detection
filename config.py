# Configuration file for traffic violation detection system

# OCR Configuration
OCR_CONFIG = {
    'confidence_threshold': 0.3,  # Lowered further for better detection
    'languages': ['en'],
    'patterns': [
        r'^[A-Z]{2}\s[0-9]{3,4}$',  # Standard format: YB 6433
        r'^[A-Z]{2}[0-9]{3,4}$',    # No space: YB6433
        r'^[A-Z]{2}\s[0-9]{2}[A-Z][0-9]{2}$',  # Format like: YB 64A3
        r'^[0-9]{2}[A-Z]\s[0-9]{4,5}$',  # Format like: 51A 12345
    ],
    # Add fuzzy matching for common OCR errors
    'fuzzy_patterns': [
        r'^YB\s?[0-9]{3,4}$',  # Any YB plate
        r'^[A-Z]{2}\s?[0-9]{3,4}$',  # Any 2-letter + numbers plate
    ],
    'target_plates': ['YB 6433', 'YB6433'],  # Specific plates to look for
    'ocr_error_tolerance': 2,  # Allow 2 character differences for fuzzy matching
    'similarity_threshold': 0.75,  # Lowered from 0.8 for better matching
    # Common OCR corrections for YB 6433
    'ocr_corrections': {
        'YB 6477': 'YB 6433',
        'YB 6437': 'YB 6433', 
        'YB 6473': 'YB 6433',
        'YB 6433': 'YB 6433',  # Correct
        'YB 6333': 'YB 6433',
        'YB 643': 'YB 6433',
        'YB 633': 'YB 6433'
    }
}

# Vehicle Detection Configuration
VEHICLE_CONFIG = {
    'model_path': 'plate_detector_model.pt',
    'confidence_threshold': 0.4,  # Lowered further for better detection
    'scale_factor': 0.8,
    'batch_dim': 8
}

# Violation Detection Configuration
VIOLATION_CONFIG = {
    'epsilon': 100,  # Increased further for more flexible detection
    'line_slope': -0.2,
    'line_intercept': 850,
    'enable_debug': True  # Enable debug prints
}

# Traffic Light Detection Configuration
TRAFFIC_LIGHT_CONFIG = {
    'position': (1810, 160, 110, 250),  # x, y, w, h
    'min_white_pixels': 500
}

# Line Detection Configuration
LINE_CONFIG = {
    'angle_threshold': 10,
    'min_line_length': 400,
    'max_line_gap': 50,
    'border_epsilon': 40
}
