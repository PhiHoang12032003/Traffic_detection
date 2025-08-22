#!/usr/bin/env python3
"""
Test Red Light Detection System with resource video
"""

from processRedLightVideo import process_red_light_video_complete
import os
import time

def test_red_light_detection():
    print('=== Testing Red Light Detection System ===')
    print()

    # Test with the original traffic video
    input_video = '../resource/traffic_video_original.mp4'
    output_video = 'processed_videos/test_red_light_output.mp4'

    print(f'Input video: {input_video}')
    print(f'Output video: {output_video}')
    print(f'Video exists: {os.path.exists(input_video)}')
    print()

    if not os.path.exists(input_video):
        print('❌ Input video not found!')
        return False

    # Create output directory
    os.makedirs('processed_videos', exist_ok=True)

    print('🚀 Starting red light detection processing...')
    print('This may take a few minutes...')
    print()

    try:
        start_time = time.time()
        result_path, stats = process_red_light_video_complete(
            input_video, 
            output_video, 
            use_improved_detection=True  # Use advanced detection with license plates
        )
        
        processing_time = time.time() - start_time
        print()
        print('✅ === PROCESSING COMPLETED ===')
        print(f'📁 Output saved to: {result_path}')
        print(f'⏱️  Processing time: {processing_time:.2f} seconds')
        print()
        print('📊 === STATISTICS ===')
        print(f'🚨 Total violations: {stats.get("total_violations", 0)}')
        print(f'🔍 Detection method: {stats.get("detection_method", "Unknown")}')
        print(f'🎬 Total frames: {stats.get("total_frames", 0)}')
        
        if stats.get("vehicle_types"):
            print()
            print('🚗 Vehicle types detected:')
            for vtype, count in stats["vehicle_types"].items():
                print(f'   - {vtype}: {count} violations')
        
        if stats.get("detected_plates"):
            print()
            print(f'🔢 License plates detected: {len(stats["detected_plates"])}')
            for i, plate_info in enumerate(stats["detected_plates"][:5]):  # Show first 5
                plate = plate_info.get("plate", "Unknown")
                vtype = plate_info.get("vehicle_type", "Unknown")
                timestamp = plate_info.get("timestamp", "Unknown")
                print(f'   {i+1}. {plate} ({vtype}) - {timestamp}')
        
        print()
        print('📁 === FILE INFO ===')
        print(f'✅ Output video exists: {os.path.exists(result_path)}')
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024*1024)
            print(f'📏 Output video size: {file_size:.2f} MB')
            
        # Check violation images
        violation_images = [f for f in os.listdir('data_vuot_den_do') if f.endswith('.jpg')] if os.path.exists('data_vuot_den_do') else []
        print(f'📸 Violation images saved: {len(violation_images)}')
        
        # Check fine documents  
        fine_docs = [f for f in os.listdir('BienBanNopPhatVuotDenDo') if f.endswith('.pdf')] if os.path.exists('BienBanNopPhatVuotDenDo') else []
        print(f'📄 Fine documents created: {len(fine_docs)}')
        
        print()
        print('✅ === TEST COMPLETED SUCCESSFULLY ===')
        print(f'🎥 You can now view the processed video: {result_path}')
        return True
        
    except Exception as e:
        print()
        print('❌ === ERROR DURING PROCESSING ===')
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_red_light_detection()
    if success:
        print()
        print("🎉 Test passed! Red light detection system is working.")
    else:
        print()
        print("💥 Test failed! Please check the error messages above.")

