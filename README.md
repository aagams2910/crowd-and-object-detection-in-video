# Crowd and Object Detection in Video

This project implements real-time crowd and object detection in video streams using YOLO (You Only Look Once) models. It's specifically designed to detect and count people in video footage, with crowd alert functionality when the number of people exceeds a threshold.

## Features

- Real-time people detection and counting
- Crowd alert system (triggers when more than 18 people are detected)
- AI-powered frame analysis using Google's Gemini 1.5 Flash for context-aware crowd management solutions
- Visual scene analysis that provides customized solutions based on crowd patterns
- Detection logging with timestamps
- Support for multiple YOLO model variants
- Video processing with detection visualization
- CSV output for detection analytics

## Prerequisites

- Python 3.6 or higher
- OpenCV
- Ultralytics YOLO
- NumPy
- Google Generative AI Python SDK
- Flask web framework

## Installation

1. Clone this repository:
```bash
git clone https://github.com/aagams2910/crowd-and-object-detection-in-video.git
cd crowd-and-object-detection-in-video
```

2. Install the required packages:
```bash
pip install -r requirements.txt
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

You can use the system in two ways:

### 1. Command Line Interface
Run the detection system using the following command:
```bash
python main.py --video [path_to_video] --model [path_to_model] --output [output_path]
```

### Arguments:
- `--video`: Path to the input video file (default: 'railwayvid.mp4')
- `--model`: Path to the YOLO model file (default: 'yolov5s.pt')
- `--output`: Path for the processed output video (default: 'output.avi')

### 2. Web Interface (NEW)
The system now includes a web-based frontend for better visualization and control:

```bash
# Install all required dependencies
pip install -r requirements.txt

# Start the web application
python app.py
```

Then, open your browser and navigate to `http://127.0.0.1:49152` to access the interface.

### Web Interface Features:
- Live video feed with person detection
- Floor selection (1st, 2nd, 3rd floor) with different camera feeds
- Adjustable crowd threshold settings
- Visual alerts when crowd exceeds the threshold
- Visual AI-powered crowd analysis with Gemini 1.5 Flash
- Tailored crowd management solutions based on actual visual scene analysis
- User-friendly dashboard layout

## Output

The system generates:
1. A processed video file with detection boxes and counts
2. A CSV file (`detections.csv`) containing:
   - Timestamps
   - Frame numbers
   - People count per frame
3. AI-generated solutions when crowd exceeds threshold

## Features in Detail

### People Detection
- Uses YOLO models for accurate person detection
- Configurable confidence threshold (default: 0.05)
- Real-time bounding box visualization
- Person count display

### Crowd Detection
- Monitors the number of people in each frame
- Visual alert when crowd threshold is exceeded (>25 people by default)
- Logging of detection events with timestamps

### AI-Powered Visual Analysis (NEW)
- Captures frames where crowd exceeds threshold
- Sends those frames to Google's Gemini 1.5 Flash multimodal model
- AI analyzes the actual crowd pattern, distribution and movement
- Generates tailored and specific crowd management solutions based on visual context
- Updates solutions based on changing crowd conditions
- Includes fallback to text-only analysis when image processing is unavailable

### Performance
- Frame skipping for improved performance (processes every 5th frame)
- Resizes frames to 1280x720 for consistent processing
- Support for early termination (press 'q' to quit)
- Cooldown timer to prevent excessive API calls

### Web Interface
- Modern Bootstrap-based UI
- Real-time video streaming
- Dynamic floor switching
- Adjustable crowd threshold
- Animated alerts for crowd detection
- AI-powered crowd management solutions panel

## Project Structure

- `main.py`: Entry point and argument parsing for CLI mode
- `app.py`: Flask web application for the frontend interface
- `people_detector.py`: Core detection and processing logic
- `templates/`: Directory containing HTML templates
  - `index.html`: Main frontend interface
- `detections.csv`: Detection log output
- `requirements.txt`: List of Python dependencies
- Various YOLO model files (*.pt)
