import cv2
import easyocr
import os
import tempfile
from datetime import datetime
from PIL import Image

from threads.OCR_thread import OCR_thread
from threads.frame_producer import FrameProducer
from threads.processor_thread import FrameProcessor

from threads.pipeline import Pipeline
import pandas as pd
import createBB_red_light


def process_red_light_video_complete(video_path, output_dir="output", video_id=None, violation_db_instance=None):
    """
    Process red light violation video with the new system
    Args:
        video_path: Path to video file
        output_dir: Output directory for results
        video_id: Database video ID (optional, for database integration)
        violation_db_instance: ViolationDatabase instance (passed from app_server)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Use violation_db passed as parameter instead of importing from app_server
    violation_db = violation_db_instance
    print(f"🔍 [DEBUG] Starting process with video_id={video_id}")
    print(f"🔍 [DEBUG] violation_db_instance provided: {violation_db_instance is not None}")
    
    if violation_db is not None:
        print(f"✅ [DATABASE] ViolationDatabase instance available for video_id={video_id}")
    else:
        if video_id:
            print(f"⚠️ [DATABASE] video_id={video_id} but violation_db_instance is None")
        else:
            print(f"⚠️ [DATABASE] No video_id and no violation_db_instance")

    #create the buffers
    frame_pipeline = Pipeline()
    processed_pipeline = Pipeline()
    violating_boxes_pipeline = Pipeline()
    violating_plates_text_pipeline = Pipeline()

    #create the threads
    t1 = FrameProducer(frame_pipeline, video_path)
    t2 = FrameProcessor(frame_pipeline, processed_pipeline,violating_boxes_pipeline, batch_dim=8)

    reader = easyocr.Reader(['en'])
    t3 = OCR_thread(violating_boxes_pipeline, violating_plates_text_pipeline, reader)

    # Initialize video writer
    video_writer = None
    frame_width = None
    frame_height = None
    fps = 30  # Default FPS
    output_filename = None

    t1.start()
    t2.start()
    t3.start()

    plates_text = []
    violation_info = []
    frame_count = 0
    violation_count = 0
    examBB = createBB_red_light.infoObject()
    
    while True:
        ret, frame = processed_pipeline.get_message()

        if not ret:  # Video finito
            break

        # Initialize video writer on first frame
        if video_writer is None:
            frame_height, frame_width = frame.shape[:2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(output_dir, f"processed_video_{timestamp}.mp4")
            
            # Get original video properties
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
            print(f"Video writer initialized: {output_filename}")

        violating_plate_text_box_info = violating_plates_text_pipeline.get_message(block=False)

        if violating_plate_text_box_info is not None:
            violating_plate_text, box_info = violating_plate_text_box_info
            #add text to list
            plates_text.append(violating_plate_text)
            violation_info.append(box_info)
            
            # Tạo PDF biên bản phạt cho vi phạm đèn đỏ
            violation_count += 1
            try:
                # Lưu ảnh vi phạm
                height, width = frame.shape[:2]
                # Lưu ảnh vi phạm đèn đỏ
                os.makedirs("data_vuot_den_do", exist_ok=True)
                image_path = f"data_vuot_den_do/{violation_count}.jpg"
                cv2.imwrite(image_path, frame)
                
                # Tạo PDF biên bản phạt
                stt_BB_red_light = f'BienBanNopPhatVuotDenDo/{violation_count}.pdf'
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                frame_pil.save(temp_image.name)
                
                # Cập nhật thông tin biên bản với biển số xe
                examBB['license_plate'] = violating_plate_text
                createBB_red_light.bienBanNopPhat(examBB, temp_image.name,
                                                 f"data_vuot_den_do/{violation_count}.jpg", stt_BB_red_light)
                temp_image.close()
                print(f"Created PDF violation report: {stt_BB_red_light}")
                
                # Save to database if available
                print(f"🔍 [DEBUG] Attempting database save:")
                print(f"  - violation_db is None: {violation_db is None}")
                print(f"  - violation_db type: {type(violation_db)}")
                print(f"  - video_id: {video_id}")
                
                if violation_db is not None and video_id is not None:
                    try:
                        time_in_video = frame_count / fps if fps > 0 else 0
                        bbox = box_info.get('bbox', [0, 0, 0, 0]) if isinstance(box_info, dict) else [0, 0, 0, 0]
                        confidence = box_info.get('confidence', 0.5) if isinstance(box_info, dict) else 0.5
                        
                        print(f"🔍 [DEBUG] Calling insert_red_light_violation with:")
                        print(f"  - video_id={video_id}")
                        print(f"  - frame_number={frame_count}")
                        print(f"  - time_in_video={time_in_video}")
                        print(f"  - license_plate={violating_plate_text}")
                        print(f"  - confidence={confidence}")
                        print(f"  - bbox={bbox}")
                        
                        v_id = violation_db.insert_red_light_violation(
                            video_id=video_id,
                            frame_number=frame_count,
                            time_in_video=time_in_video,
                            license_plate=violating_plate_text,
                            confidence=confidence,
                            bbox=bbox,
                            image_path=image_path,
                            pdf_report_path=stt_BB_red_light
                        )
                        
                        if v_id:
                            print(f"✅✅✅ [DATABASE SUCCESS] Violation saved to database!")
                            print(f"✅ [CAMERA 3] violation_id={v_id}, plate={violating_plate_text}")
                        else:
                            print(f"❌ [DATABASE ERROR] insert_red_light_violation returned None")
                            
                    except Exception as db_err:
                        print(f"❌❌❌ [DATABASE ERROR] Failed to save: {db_err}")
                        import traceback
                        traceback.print_exc()
                else:
                    if not violation_db:
                        print(f"⚠️ [DATABASE SKIP] violation_db is None")
                    if not video_id:
                        print(f"⚠️ [DATABASE SKIP] video_id is None")
                
            except Exception as e:
                print(f"Error creating PDF for violation {violation_count}: {e}")
        
        # Hiển thị thông tin vi phạm trên video (không có nền đen)
        if len(plates_text) > 0:
            for i, text in enumerate(plates_text):
                # Hiển thị text vi phạm trực tiếp, không vẽ nền đen
                cv2.putText(frame, f"Vi pham {i+1}: {text}", 
                          (int(710*t2.scale_factor), int(65*t2.scale_factor+(i*50))), 
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                          fontScale=0.8, 
                          color=(0,255,255),  # Màu vàng cyan
                          thickness=2)

        # Write frame to video
        if video_writer is not None:
            video_writer.write(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

        # For web streaming, we don't show cv2.imshow
        # cv2.imshow("c", frame)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #     t1.stop()
        #     t2.stop()
        #     break

    # Clean up
    # cv2.destroyAllWindows()
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved successfully: {output_filename}")
        print(f"Total frames processed: {frame_count}")

    # Stop all threads
    t1.stop()
    t2.stop()
    t3.stop()

    #write the results on csv
    if violation_info:
        df = pd.DataFrame(violation_info)
        df["plate"] = plates_text
        csv_filename = os.path.join(output_dir, f"violations_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Violations data saved: {csv_filename}")
    
    print("Processing complete!")
    return output_filename if video_writer else None, len(plates_text)


def generate_frames_red_light_new(video_path):
    """
    Generator function for streaming red light detection frames
    """
    # Create the buffers
    frame_pipeline = Pipeline()
    processed_pipeline = Pipeline()
    violating_boxes_pipeline = Pipeline()
    violating_plates_text_pipeline = Pipeline()

    # Create the threads
    t1 = FrameProducer(frame_pipeline, video_path)
    t2 = FrameProcessor(frame_pipeline, processed_pipeline, violating_boxes_pipeline, batch_dim=8)

    reader = easyocr.Reader(['en'])
    t3 = OCR_thread(violating_boxes_pipeline, violating_plates_text_pipeline, reader)

    t1.start()
    t2.start()
    t3.start()

    plates_text = []
    violation_info = []
    
    try:
        while True:
            ret, frame = processed_pipeline.get_message()

            if not ret:  # Video finished
                break

            violating_plate_text_box_info = violating_plates_text_pipeline.get_message(block=False)

            if violating_plate_text_box_info is not None:
                violating_plate_text, box_info = violating_plate_text_box_info
                plates_text.append(violating_plate_text)
                violation_info.append(box_info)
            
            # Hiển thị thông tin vi phạm trên video streaming (không có nền đen)
            if len(plates_text) > 0:
                for i, text in enumerate(plates_text):
                    # Hiển thị text vi phạm trực tiếp, không vẽ nền đen
                    cv2.putText(frame, f"Vi pham {i+1}: {text}", 
                              (int(710*t2.scale_factor), int(65*t2.scale_factor+(i*50))), 
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale=0.8, 
                              color=(0,255,255),  # Màu vàng cyan
                              thickness=2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        # Stop all threads
        t1.stop()
        t2.stop()
        t3.stop()
