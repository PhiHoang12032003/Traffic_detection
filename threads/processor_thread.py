import threading
from time import sleep
import cv2
import numpy as np

from ultralytics import YOLO

from detectors import detect_line, detect_trafficlight_color

def is_infraction(box: list, line: list, scale_factor) -> bool:
    lx1, ly1, lx2, ly2 = line.copy()
    bx1, by1, bx2, by2 = box

    epsilon = int(45*scale_factor)  # little higher due to prospective

    slope2, intercept2 = -0.2, int(850*scale_factor)  # linea inclinata destra

    def line2(y):
        return (y - intercept2) / slope2

    return by1 > ly1 - epsilon and bx1 > line2(by1)


class FrameProcessor(threading.Thread):
    def __init__(self, frame_pipeline, processed_pipeline, violating_boxes_pipeline, batch_dim, model_name='plate_detector_model.pt', scale_factor=0.8):
        threading.Thread.__init__(self, daemon=True)

        self.model = YOLO(model_name)

        self.batch_dim = batch_dim

        self.frame_pipeline = frame_pipeline
        self.processed_pipeline = processed_pipeline
        self.violating_boxes_pipeline = violating_boxes_pipeline

        self.line = [0,0,0,0]
        self.seen_ids = set()

        self.scale_factor = scale_factor


        self.stopped = False

    def run(self):
        while not self.stopped:

            #GET ELEMENT FROM FRAME BUFFER, PUT IT INTO A BATCH
            batch = []
            for _ in range(self.batch_dim):
                ret, frame = self.frame_pipeline.get_message()

                if not ret:
                    self.stop()
                    self.processed_pipeline.set_message((ret, frame))
                    break

                #reduce resolution

                frame_resized = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
                batch.append((ret,frame_resized))

            #EXTRACT FRAME AND RETS
            rets = [el[0] for el in batch]
            frames = [el[1] for el in batch]

            #if at least one frame to process
            if len(rets) > 0:


                #detect traffic light colors and lines
                tl_colors_batch = []
                batch_lines = []

                for i, frame in enumerate(frames):

                    tl_color_str, tl_color = detect_trafficlight_color(frame, self.scale_factor)
                    tl_colors_batch.append(tl_color_str)

                    self.line = detect_line(frame, self.line, self.scale_factor)
                    batch_lines.append(self.line)
                    #draw rectangle only if the traffic light is red
                    if tl_color_str == "red":
                        x1, y1, x2, y2 = self.line
                        # Calculate rectangle thickness (height of rectangle)
                        rectangle_thickness = int(40 * self.scale_factor)  # Độ dày hình chữ nhật
                        
                        # Get frame dimensions
                        height, width = frame.shape[:2]
                        
                        # Calculate slope and intercept of the line
                        if x2 != x1:
                            slope = (y2 - y1) / (x2 - x1)
                            intercept = y1 - slope * x1
                        else:
                            # Vertical line case
                            slope = 0
                            intercept = y1
                        
                        # Calculate y coordinates at left (x=0) and right (x=width) edges
                        y_left = int(slope * 0 + intercept)
                        y_right = int(slope * width + intercept)
                        
                        # Create 4 points for the diagonal rectangle
                        # Top-left, top-right, bottom-right, bottom-left
                        pts = np.array([
                            [0, y_left],                    # Top-left
                            [width, y_right],              # Top-right
                            [width, y_right + rectangle_thickness],  # Bottom-right
                            [0, y_left + rectangle_thickness]        # Bottom-left
                        ], np.int32)
                        
                        # Draw filled rectangle (semi-transparent for better visibility)
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [pts], tl_color)
                        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                        
                        # Draw rectangle border for better visibility (thicker border)
                        cv2.polylines(frame, [pts], True, tl_color, 3)

                    cv2.rectangle(frame, (int(80*self.scale_factor), int(40*self.scale_factor)), (int(687*self.scale_factor), int(120*self.scale_factor)), (0, 0, 0), -1)
                    cv2.putText(frame, f" Traffic Light Status:", (int(100*self.scale_factor), int(100*self.scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
                    cv2.putText(frame, f"  {tl_color_str}", (int(500*self.scale_factor), int(100*self.scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 1, tl_color, 2)

                #MAKE PREDICTION
                results = self.model.track(frames, persist=True, conf=0.7, verbose=False)

                if results is not None:
                    annotated_frames = []
                    violating_boxes = []

                    for r, tl_color, line, frame in zip(results, tl_colors_batch, batch_lines, frames): #FOR EACH FRAME
                        boxes = r.boxes.xyxy.cpu()
                        if r.boxes.id is not None:
                            track_ids = r.boxes.id.int().cpu().tolist()
                        else:
                            track_ids = []

                        annotated_frame = r.plot() #if no boxes found, return original frame
                        annotated_frames.append(annotated_frame)


                        for box, track_id in zip(boxes, track_ids):

                            box_info = {"FrameID": id(r), "ObjectID": track_id, "BBox": box}

                            if is_infraction(box_info['BBox'], line, self.scale_factor) and box_info["ObjectID"] not in self.seen_ids and tl_color == "red":

                                #crop the box
                                x1, y1, x2, y2 = box_info['BBox']
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                box_info['BBox'] = [int(x1), int(y1), int(x2), int(y2)]

                                violating_boxes.append((frame[y1:y2, x1:x2], box_info))
                                self.seen_ids.add(box_info["ObjectID"])



                    #write annotated images on buffer
                    frames = annotated_frames
                    for ret, annotated_frame in zip(rets, frames):
                        self.processed_pipeline.set_message((ret, annotated_frame))


                    #write violating boxes on buffer
                    for box in violating_boxes:
                        self.violating_boxes_pipeline.set_message(box)

            sleep(0.03)

    def stop(self):
        self.stopped = True
