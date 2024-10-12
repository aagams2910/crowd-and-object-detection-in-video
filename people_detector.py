import cv2
import numpy as np
from ultralytics import YOLO

class PeopleDetector:
    CONFIDENCE_THRESHOLD = 0.3
    FONT_SIZE = 0.5
    TEXT_COLOR = (0, 255, 255)
    INFO_BOX_COLOR = (138, 72, 48)
    TEXT_THICKNESS = 1

    COUNT_LABEL_POSITION = (15, 22)
    COUNT_TEXT_POSITION = (100, 22)

    def __init__(self, model_version: str, video_source: str) -> None:
        self.count = 0
        self.model_version = model_version
        self.cap = cv2.VideoCapture(video_source)
        self.model = self.initialize_model()

    def initialize_model(self):
        if self.model_version == "face":
            model = YOLO("yolov5x.pt")  
        else:
            raise ValueError("Unsupported model version. Please use 'face'.")
        return model

    def display_count_box(self, image: np.ndarray) -> None:
        cv2.rectangle(image, (0, 0), (130, 35), self.INFO_BOX_COLOR, -1)
        cv2.putText(image, 'People: ', self.COUNT_LABEL_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)
        cv2.putText(image, str(self.count), self.COUNT_TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)

    def process_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break
            
            frame = cv2.resize(frame, (1280, 720))  

            results = self.model(frame)

            human_boxes = results[0].boxes if results else []

            self.count = 0
            for box in human_boxes:
                if box.cls == 0 and box.conf >= self.CONFIDENCE_THRESHOLD:  
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf.cpu().numpy().item()  
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    frame = cv2.putText(frame, f'Person: {conf:.2f}', (int(x1), int(y1) - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.TEXT_COLOR, self.TEXT_THICKNESS)
                    self.count += 1

            self.display_count_box(frame)
            cv2.imshow('output', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
