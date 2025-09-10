import threading
from time import sleep
import cv2

class FrameProducer(threading.Thread):
    def __init__(self, frame_pipeline, video_url):
        threading.Thread.__init__(self, daemon=True)
        self.frame_pipeline = frame_pipeline
        self.stopped = False
        self.video_url = video_url

    def run(self):
        cap = cv2.VideoCapture(self.video_url)

        if cap.isOpened():

            while not self.stopped:

                ret, frame = cap.read()

                self.frame_pipeline.set_message((ret, frame))

                if not ret:
                    break

                sleep(0.03)

            cap.release()


    def stop(self):
        self.stopped = True
