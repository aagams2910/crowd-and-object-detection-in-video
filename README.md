Make sure you install ultralytics library:

pip install ultralytics

Once you have ultralytics installed, you can load any of the YOLO models 
for this following project the following models are needed:

from ultralytics import YOLO

model_m = YOLO('yolov5m.pt')         # YOLOv5 Medium model

model_s_face = YOLO('yolov5s-face.pt') # YOLOv5 Small model - face

model_s = YOLO('yolov5s.pt')         # YOLOv5 Small model

model_xu = YOLO('yolov5xu.pt')       # YOLOv5 XU model

model_v8n = YOLO('yolov8n.pt')       # YOLOv8 Nano model

OR

you can use the wget command on cmd as follows:

wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt

wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s-face.pt

wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt

wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5xu.pt

wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov8n.pt
