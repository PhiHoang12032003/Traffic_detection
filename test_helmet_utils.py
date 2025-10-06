#!/usr/bin/env python3
"""
Test script for helmet PDF utils
"""
import cv2
import numpy as np
import os
from utils.helmet_pdf_utils import create_helmet_pdf_report, get_helmet_violation_info, create_helmet_violation_pdf

def test_helmet_utils():
    """Test helmet PDF utilities"""
    print("🧪 Testing helmet PDF utilities...")
    
    # Test 1: Get violation info
    print("\n1. Testing get_helmet_violation_info()...")
    info = get_helmet_violation_info("51A-12345")
    print(f"✅ Info created: {info['date']}")
    print(f"   License plate: {info.get('license_plate', 'None')}")
    
    # Test 2: Create sample frame
    print("\n2. Creating sample violation frame...")
    # Create a simple test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Gray background
    
    # Add some text to the frame
    cv2.putText(frame, "HELMET VIOLATION TEST", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Test Frame", (50, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    print("✅ Sample frame created")
    
    # Test 3: Create PDF report
    print("\n3. Testing create_helmet_pdf_report()...")
    try:
        pdf_path = create_helmet_pdf_report(frame, 1, "51A-12345")
        if os.path.exists(pdf_path):
            print(f"✅ PDF created successfully: {pdf_path}")
        else:
            print(f"❌ PDF file not found: {pdf_path}")
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
    
    # Test 4: Check directories
    print("\n4. Checking created directories...")
    dirs_to_check = [
        "data_xe_vp_bh",
        "BienBanNopPhatXeMayViPhamMuBaoHiem"
    ]
    
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            files = os.listdir(dir_name)
            print(f"✅ {dir_name}: {len(files)} files")
        else:
            print(f"❌ {dir_name}: Directory not found")
    
    print("\n🎉 Helmet utils test completed!")

if __name__ == "__main__":
    test_helmet_utils()

