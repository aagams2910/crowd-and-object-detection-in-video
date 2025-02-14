import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
import csv
from pathlib import Path

class PeopleDetector:
    def __init__(self, model_path: str, video_source: str = "path/to/your/video.mp4", confidence_threshold=0.05):
        self.count = 0
        self.video_source = video_source
        self.confidence_threshold = confidence_threshold
        self.cap = cv2.VideoCapture(self.video_source)
        self.model = self.initialize_model(model_path)
        self.log_file = Path('detections.csv')
        self.setup_logging()
        self.initialize_log_file()

    def initialize_model(self, model_path: str):
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {e}")

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def initialize_log_file(self):
        with self.log_file.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Frame', 'Count'])

    def log_detections(self, frame_number, count):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.log_file.open('a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, frame_number, count])
        logging.info(f"Frame {frame_number}: Detected {count} people")

    def process_video(self, output_path="output.avi"):
        frame_number = 0
        frame_skip = 5
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_writer = None

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if frame_number % frame_skip != 0:
                frame_number += 1
                continue
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            results = self.model(frame)
            human_boxes = [box for box in results[0].boxes if box.cls == 0]
            self.count = sum(1 for box in human_boxes if box.conf >= self.confidence_threshold)

            for box in human_boxes:
                if box.conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf.cpu().numpy().item()
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person: {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            self.log_detections(frame_number, self.count)
            cv2.putText(frame, f"People: {self.count}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if self.count > 18:
                cv2.putText(frame, "ALERT: CROWD DETECTED!", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if output_writer is None:
                output_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            output_writer.write(frame)
            cv2.imshow('output', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            frame_number += 1

        self.cap.release()
        output_writer.release()
        cv2.destroyAllWindows()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
