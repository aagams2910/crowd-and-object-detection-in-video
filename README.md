# Crowd and Object Detection in Video

This project implements real-time crowd and object detection in video streams using YOLO (You Only Look Once) models. It's specifically designed to detect and count people in video footage, with crowd alert functionality when the number of people exceeds a threshold.

## Features

- Real-time people detection and counting
- Crowd alert system (triggers when more than 18 people are detected)
- Detection logging with timestamps
- Support for multiple YOLO model variants
- Video processing with detection visualization
- CSV output for detection analytics

## Prerequisites

- Python 3.6 or higher
- OpenCV
- Ultralytics YOLO
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/aagams2910/crowd-and-object-detection-in-video.git
cd crowd-and-object-detection-in-video
```

2. Install the required packages:
```bash
pip install ultralytics opencv-python numpy
```

3. Download the required YOLO models:
```bash
# You can use wget or manually download the following models:
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s-face.pt
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5xu.pt
wget https://github.com/ultralytics/yolov8/releases/download/v8.0.0/yolov8n.pt
```

## Usage

Run the detection system using the following command:
```bash
python main.py --video [path_to_video] --model [path_to_model] --output [output_path]
```

### Arguments:
- `--video`: Path to the input video file (default: 'railwayvid.mp4')
- `--model`: Path to the YOLO model file (default: 'yolov5s.pt')
- `--output`: Path for the processed output video (default: 'output.avi')

## Output

The system generates:
1. A processed video file with detection boxes and counts
2. A CSV file (`detections.csv`) containing:
   - Timestamps
   - Frame numbers
   - People count per frame

## Features in Detail

### People Detection
- Uses YOLO models for accurate person detection
- Configurable confidence threshold (default: 0.05)
- Real-time bounding box visualization
- Person count display

### Crowd Detection
- Monitors the number of people in each frame
- Visual alert when crowd threshold is exceeded (>18 people)
- Logging of detection events with timestamps

### Performance
- Frame skipping for improved performance (processes every 5th frame)
- Resizes frames to 1280x720 for consistent processing
- Support for early termination (press 'q' to quit)

## Project Structure

- `main.py`: Entry point and argument parsing
- `people_detector.py`: Core detection and processing logic
- `detections.csv`: Detection log output
- Various YOLO model files (*.pt)
