from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import threading
import time
import google.generativeai as genai
import base64
import numpy as np
from dotenv import load_dotenv
from people_detector import PeopleDetector

app = Flask(__name__)

# Configure Google Gemini API
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Global variables
output_frame = None
lock = threading.Lock()
detector = None
current_video = 'Floor1.mp4'
current_floor = '1st Floor'
threshold = 25  # Updated threshold to 25
alert_status = False
current_solutions = []
last_frame_analysis_time = 0
frame_analysis_cooldown = 30  # Seconds between frame analyses
view_mode = 'single'  # 'single', 'grid', 'webcam'
grid_frames = {}  # Store frames for each floor in grid view
webcam = None  # Webcam capture object

# Available floor options with corresponding videos
floor_options = {
    '1st Floor': 'Floor1.mp4',
    '2nd Floor': 'Floor2.mp4',
    '3rd Floor': 'Floor3.mp4'
}

# View mode options
view_modes = {
    'Single Floor': 'single',
    'Grid View': 'grid',
    'Real-time Monitoring': 'webcam'
}

def initialize_detector(video_source='Floor1.mp4', model_path='yolov5su.pt'):
    global detector
    
    # Check if the video source exists directly or needs a path prefix
    if not os.path.isfile(video_source):
        # Try in the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(current_dir, video_source)
        
        # If still not found, try a videos subfolder
        if not os.path.isfile(video_path):
            video_path = os.path.join(current_dir, 'videos', video_source)
        
        print(f"Looking for video at: {video_path}")
        
        if os.path.isfile(video_path):
            video_source = video_path
        else:
            print(f"Warning: Video file not found: {video_source}")
    
    try:
        detector = PeopleDetector(model_path=model_path, video_source=video_source, confidence_threshold=0.05)
        print(f"Detector initialized with video: {video_source}")
        return detector
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")
        return None

def initialize_webcam():
    global webcam
    try:
        # Try different webcam indices
        for idx in range(2):  # Try index 0 and 1
            print(f"Attempting to open webcam at index {idx}")
            webcam = cv2.VideoCapture(idx)
            if webcam.isOpened():
                width = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Webcam opened successfully at index {idx} with resolution {width}x{height}")
                return True
            else:
                print(f"Failed to open webcam at index {idx}")
        
        # If we reached here, no webcam was found
        print("No webcam could be opened. Trying alternate method...")
        
        # Try with DirectShow on Windows
        webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if webcam.isOpened():
            print("Webcam opened successfully with DirectShow")
            return True
            
        print("All webcam initialization methods failed")
        return False
    except Exception as e:
        print(f"Error initializing webcam: {str(e)}")
        return False

def initialize_grid_view():
    global grid_frames
    # Initialize detectors for each floor
    for floor, video in floor_options.items():
        # Get the proper video path
        video_path = video
        if not os.path.isfile(video):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(current_dir, video)
            if os.path.isfile(potential_path):
                video_path = potential_path
            else:
                potential_path = os.path.join(current_dir, 'videos', video)
                if os.path.isfile(potential_path):
                    video_path = potential_path
        
        print(f"Grid view: Initializing {floor} with video: {video_path}")
        
        if floor not in grid_frames:
            try:
                grid_frames[floor] = {
                    'frame': np.zeros((720, 1280, 3), dtype=np.uint8),
                    'detector': PeopleDetector(model_path='yolov5su.pt', video_source=video_path, confidence_threshold=0.05)
                }
                cv2.putText(grid_frames[floor]['frame'], f"Loading {floor}...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error initializing grid detector for {floor}: {str(e)}")
                grid_frames[floor] = {
                    'frame': np.zeros((720, 1280, 3), dtype=np.uint8),
                    'detector': None
                }
                cv2.putText(grid_frames[floor]['frame'], f"Error: {str(e)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return True

def encode_image_for_gemini(frame):
    """Encode OpenCV frame to base64 for Gemini API"""
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return None
    
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

def analyze_frame_with_gemini(frame, count, floor):
    """Send the frame to Gemini 1.5 Flash for analysis"""
    try:
        encoded_image = encode_image_for_gemini(frame)
        if not encoded_image:
            return get_solutions_from_gemini(count, floor)  # Fallback to text-only if image encoding fails
        
        prompt = f"""
        TASK: Analyze this image of a mall surveillance showing {count} people on the {floor}.
        The crowd threshold is {threshold} people and has been exceeded.
        
        INSTRUCTIONS:
        1. Analyze the crowd distribution, density and movement patterns
        2. Consider potential safety risks or bottlenecks
        3. Generate 4 specific, actionable solutions for managing this exact crowd situation
        
        FORMAT RESPONSE AS A JSON LIST:
        [
          {{"title": "Short solution title", "description": "Brief actionable description (exactly 20-30 words)"}}
        ]
        
        Keep titles under 5 words and descriptions exactly 20-30 words.
        """
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": encoded_image}
        ])
        
        response_text = response.text
        
        # Extract JSON from response (handles potential markdown formatting)
        if "```json" in response_text:
            import json
            import re
            json_str = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_str:
                solutions = json.loads(json_str.group(1))
                # Print solutions in backend console
                print("\n===== VISUAL ANALYSIS SOLUTIONS =====")
                for idx, solution in enumerate(solutions):
                    print(f"{idx+1}. {solution['title']}: {solution['description']}")
                print("====================================\n")
                return solutions
        
        # Try direct JSON parsing
        import json
        try:
            solutions = json.loads(response_text)
            # Print solutions in backend console
            print("\n===== VISUAL ANALYSIS SOLUTIONS =====")
            for idx, solution in enumerate(solutions):
                print(f"{idx+1}. {solution['title']}: {solution['description']}")
            print("====================================\n")
            return solutions
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error: {str(json_err)}")
            print(f"Response text: {response_text}")
            raise Exception(f"Invalid JSON response: {str(json_err)}")
            
    except Exception as e:
        print(f"Error analyzing frame with Gemini: {str(e)}")
        return get_solutions_from_gemini(count, floor)  # Fallback to text-only if image analysis fails

def get_solutions_from_gemini(count, floor):
    try:
        prompt = f"""
        Generate 4 specific crowd management solutions for a shopping mall where {count} people have been detected on the {floor}.
        The crowd threshold is {threshold} people. Format as a JSON list with 'title' and 'description' fields.
        Keep titles under 5 words and descriptions exactly 20-30 words in length.
        Example format: [{{"title": "Solution Title", "description": "Brief description (exactly 20-30 words)"}}]
        """
        
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Extract JSON from response (handles potential markdown formatting)
        if "```json" in response_text:
            import json
            import re
            json_str = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_str:
                solutions = json.loads(json_str.group(1))
                # Print solutions in backend console
                print("\n===== TEXT-ONLY SOLUTIONS =====")
                for idx, solution in enumerate(solutions):
                    print(f"{idx+1}. {solution['title']}: {solution['description']}")
                print("==============================\n")
                return solutions
        
        # Try direct JSON parsing
        import json
        try:
            solutions = json.loads(response_text)
            # Print solutions in backend console
            print("\n===== TEXT-ONLY SOLUTIONS =====")
            for idx, solution in enumerate(solutions):
                print(f"{idx+1}. {solution['title']}: {solution['description']}")
            print("==============================\n")
            return solutions
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error: {str(json_err)}")
            print(f"Response text: {response_text}")
            raise Exception(f"Invalid JSON response: {str(json_err)}")
    except Exception as e:
        print(f"Error getting solutions from Gemini: {str(e)}")
        fallback_solutions = [
            {"title": "Redirect Traffic", "description": "Guide visitors to less crowded areas using digital signage and staff at key junctions to maintain smooth flow patterns."},
            {"title": "Staff Deployment", "description": "Position additional personnel at crowd hotspots to manage flow, address concerns and maintain security presence."},
            {"title": "Entry Control", "description": "Temporarily regulate access to affected areas using a one-in-one-out system until density decreases to safe levels."},
            {"title": "Open Alternate Routes", "description": "Unlock emergency or staff pathways to create additional movement options and reduce pressure in main corridors."}
        ]
        # Print fallback solutions
        print("\n===== FALLBACK SOLUTIONS =====")
        for idx, solution in enumerate(fallback_solutions):
            print(f"{idx+1}. {solution['title']}: {solution['description']}")
        print("============================\n")
        return fallback_solutions

def detect_people():
    global output_frame, lock, detector, alert_status, threshold, current_solutions, last_frame_analysis_time, grid_frames, webcam, view_mode
    
    frame_number = 0
    frame_skip = 5
    
    # Initialize grid view if not already done
    initialize_grid_view()
    
    # Try to initialize webcam, but don't stop execution if it fails
    try:
        initialize_webcam()
    except Exception as e:
        print(f"Warning: Could not initialize webcam: {str(e)}")
    
    while True:
        if view_mode == 'single':
            # Single floor view
            if detector is None:
                time.sleep(0.1)
                continue
                
            ret, frame = detector.cap.read()
            if not ret:
                # Restart video when it ends
                detector.cap.release()
                detector.cap = cv2.VideoCapture(current_video)
                continue
                
            if frame_number % frame_skip != 0:
                frame_number += 1
                continue
                
            frame = cv2.resize(frame, (1280, 720))
            results = detector.model(frame)
            human_boxes = [box for box in results[0].boxes if box.cls == 0]
            count = sum(1 for box in human_boxes if box.conf >= detector.confidence_threshold)
            
            for box in human_boxes:
                if box.conf >= detector.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf.cpu().numpy().item()
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person: {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            detector.log_detections(frame_number, count)
            cv2.putText(frame, f"People: {count}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Floor: {current_floor}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Update alert status
            new_alert_status = count > threshold
            
            # If alert status changes to True (crowd exceeds threshold) or 
            # it's been more than the cooldown period since the last analysis
            current_time = time.time()
            if (new_alert_status and 
                (not alert_status or current_time - last_frame_analysis_time > frame_analysis_cooldown)):
                # Use the current frame to analyze with Gemini
                current_solutions = analyze_frame_with_gemini(frame.copy(), count, current_floor)
                last_frame_analysis_time = current_time
                
            alert_status = new_alert_status
            
            if alert_status:
                cv2.putText(frame, "ALERT: CROWD DETECTED!", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            with lock:
                output_frame = frame.copy()
        
        elif view_mode == 'grid':
            # Grid view of all floors
            if not grid_frames:
                time.sleep(0.1)
                continue
            
            # Create a grid of all three floors
            grid_output = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Process each floor
            row_height = 240  # Fixed height for each row (720 / 3)
            for i, (floor, data) in enumerate(grid_frames.items()):
                detector = data['detector']
                
                if detector is None or not detector.cap.isOpened():
                    # If detector not available, use a blank frame with message
                    frame = np.zeros((row_height, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"{floor}: No video available", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    # Get frame from floor video
                    ret, frame = detector.cap.read()
                    if not ret:
                        detector.cap.release()
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        video_path = os.path.join(current_dir, floor_options[floor])
                        if not os.path.isfile(video_path):
                            video_path = os.path.join(current_dir, 'videos', floor_options[floor])
                        detector.cap = cv2.VideoCapture(video_path)
                        
                        # Create a blank frame for this iteration
                        frame = np.zeros((row_height, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, f"Restarting {floor} video...", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        # Resize and process the frame
                        frame = cv2.resize(frame, (640, row_height))
                        try:
                            results = detector.model(frame)
                            human_boxes = [box for box in results[0].boxes if box.cls == 0]
                            count = sum(1 for box in human_boxes if box.conf >= detector.confidence_threshold)
                            
                            for box in human_boxes:
                                if box.conf >= detector.confidence_threshold:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    conf = box.conf.cpu().numpy().item()
                                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            cv2.putText(frame, f"{floor}: {count} people", (10, 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            
                            # Alert indication
                            if count > threshold:
                                cv2.putText(frame, "ALERT!", (10, 40), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        except Exception as e:
                            print(f"Error processing {floor} frame: {str(e)}")
                            cv2.putText(frame, f"Error: {str(e)[:20]}...", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Place the frame in the grid by row
                row = i  # One floor per row
                start_y = row * row_height
                # Center the frame horizontally
                start_x = (1280 - 640) // 2
                
                # Place in grid
                grid_output[start_y:start_y + row_height, start_x:start_x + 640] = frame
                
                # Store the processed frame
                data['frame'] = frame
            
            # Add a title to the grid view
            cv2.putText(grid_output, "Mall Monitoring - All Floors", (450, 710), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            with lock:
                output_frame = grid_output.copy()
        
        elif view_mode == 'webcam':
            # Webcam monitoring
            if webcam is None or not webcam.isOpened():
                try:
                    success = initialize_webcam()
                    if not success:
                        # If webcam cannot be opened, create a message frame
                        error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(error_frame, "Error: Cannot access webcam", (400, 360), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(error_frame, "Please ensure your webcam is connected and not in use by another application", 
                                   (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        with lock:
                            output_frame = error_frame.copy()
                        time.sleep(1)
                        continue
                except Exception as e:
                    error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(error_frame, f"Error: {str(e)}", (400, 360), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    with lock:
                        output_frame = error_frame.copy()
                    time.sleep(1)
                    continue
            
            try:
                ret, frame = webcam.read()
                if not ret:
                    error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "Error: Failed to read from webcam", (350, 360), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    with lock:
                        output_frame = error_frame.copy()
                    time.sleep(0.5)
                    continue
                
                # Flip the frame horizontally for a more natural view
                frame = cv2.flip(frame, 1)
                
                frame = cv2.resize(frame, (1280, 720))
                
                # Use the detector model for the webcam feed if detector is available
                if detector is not None:
                    try:
                        results = detector.model(frame)
                        human_boxes = [box for box in results[0].boxes if box.cls == 0]
                        count = sum(1 for box in human_boxes if box.conf >= detector.confidence_threshold)
                        
                        for box in human_boxes:
                            if box.conf >= detector.confidence_threshold:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf.cpu().numpy().item()
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(frame, f'Person: {conf:.2f}', (int(x1), int(y1) - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        cv2.putText(frame, f"People: {count}", (15, 22), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Update alert status for webcam
                        new_alert_status = count > threshold
                        alert_status = new_alert_status
                        
                        if alert_status:
                            cv2.putText(frame, "ALERT: CROWD DETECTED!", (15, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    except Exception as e:
                        # Log the error but continue processing
                        print(f"Error processing webcam frame with model: {str(e)}")
                        cv2.putText(frame, f"Processing error: {str(e)[:30]}...", (15, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Add current date/time
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"Real-time Monitoring (Webcam) - {current_time}", (15, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                with lock:
                    output_frame = frame.copy()
            except Exception as e:
                print(f"General error in webcam processing: {str(e)}")
                error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Webcam Error: {str(e)}", (350, 360), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                with lock:
                    output_frame = error_frame.copy()
                time.sleep(0.5)
        
        frame_number += 1
        time.sleep(0.03)  # Slight delay to reduce CPU usage

def generate():
    global output_frame, lock
    
    while True:
        try:
            with lock:
                if output_frame is None:
                    # Create a black frame with message if no output frame is available
                    temp_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(temp_frame, "Waiting for video...", (400, 360), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    (flag, encoded_image) = cv2.imencode(".jpg", temp_frame)
                else:
                    (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
                
                if not flag:
                    continue
            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encoded_image) + b'\r\n')
            time.sleep(0.05)
        except Exception as e:
            print(f"Error in generate: {str(e)}")
            time.sleep(0.1)
            continue

@app.route('/')
def index():
    return render_template('index.html', floor_options=floor_options, current_floor=current_floor, threshold=threshold, view_modes=view_modes)

@app.route('/video_feed')
def video_feed():
    try:
        if view_mode == 'webcam' and (webcam is None or not webcam.isOpened()):
            print("Webcam is not accessible for video feed")
            blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Webcam not available", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                          bytearray(buffer) + b'\r\n',
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        if view_mode == 'single' and (not detector or not detector.cap.isOpened()):
            print(f"Video file not accessible for single view mode: {current_video}")
            blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Video not available: {current_video}", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                          bytearray(buffer) + b'\r\n',
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        if view_mode == 'grid' and not grid_frames:
            print("Grid frames not initialized")
            blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Grid view not available", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                          bytearray(buffer) + b'\r\n',
                          mimetype='multipart/x-mixed-replace; boundary=frame')
            
        print(f"Starting video feed in {view_mode} mode")
        return Response(generate(),
                      mimetype='multipart/x-mixed-replace; boundary=frame',
                      headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                              'Pragma': 'no-cache',
                              'Expires': '0'})
    except Exception as e:
        print(f"Error in video_feed: {str(e)}")
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(blank_frame, f"Error: {str(e)}", (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank_frame)
        return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                      bytearray(buffer) + b'\r\n',
                      mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global alert_status, current_solutions, view_mode
    return jsonify({
        'alert': alert_status,
        'current_floor': current_floor,
        'threshold': threshold,
        'solutions': current_solutions,
        'view_mode': view_mode
    })

@app.route('/change_floor', methods=['POST'])
def change_floor():
    global current_floor, current_video, detector
    
    floor = request.json.get('floor')
    if floor in floor_options:
        current_floor = floor
        current_video = floor_options[floor]
        
        # Release current detector and initialize new one
        if detector is not None:
            detector.cap.release()
        
        initialize_detector(video_source=current_video)
        
        return jsonify({'success': True, 'message': f'Changed to {floor}'})
    
    return jsonify({'success': False, 'message': 'Invalid floor option'})

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    global threshold
    
    new_threshold = request.json.get('threshold')
    try:
        threshold = int(new_threshold)
        return jsonify({'success': True, 'message': f'Threshold updated to {threshold}'})
    except:
        return jsonify({'success': False, 'message': 'Invalid threshold value'})

@app.route('/change_view_mode', methods=['POST'])
def change_view_mode():
    global view_mode, webcam, grid_frames
    
    new_mode = request.json.get('view_mode')
    if new_mode in view_modes.values():
        # Prepare resources for the new view mode
        if new_mode == 'webcam' and (webcam is None or not webcam.isOpened()):
            print("Initializing webcam for webcam view mode")
            initialize_webcam()
        
        if new_mode == 'grid' and not grid_frames:
            print("Initializing grid frames for grid view mode")
            initialize_grid_view()
        
        view_mode = new_mode
        print(f"View mode changed to: {view_mode}")
        return jsonify({'success': True, 'message': f'View mode updated to {view_mode}'})
    
    return jsonify({'success': False, 'message': 'Invalid view mode'})

def check_video_files():
    # Create a videos directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(current_dir, 'videos')
    
    if not os.path.exists(videos_dir):
        try:
            os.makedirs(videos_dir)
            print(f"Created videos directory at: {videos_dir}")
        except Exception as e:
            print(f"Error creating videos directory: {str(e)}")
    
    # Check if the video files exist
    missing_videos = []
    for floor, video in floor_options.items():
        # Check various locations
        found = False
        paths_to_check = [
            video,  # Direct path
            os.path.join(current_dir, video),  # In app directory
            os.path.join(videos_dir, video)  # In videos subdirectory
        ]
        
        for path in paths_to_check:
            if os.path.isfile(path):
                print(f"Found video for {floor} at: {path}")
                found = True
                break
        
        if not found:
            missing_videos.append((floor, video))
    
    if missing_videos:
        print("\n===== MISSING VIDEO FILES =====")
        print("The following video files are missing:")
        for floor, video in missing_videos:
            print(f"- {floor}: {video}")
        print("Creating test videos as replacements...")
        
        # Create test videos for missing files
        for floor, video in missing_videos:
            create_test_video_if_missing(video)
        
        print("Test videos created successfully")
        print("===============================\n")
    
    return True  # We're handling missing videos by creating test videos

def create_test_video_if_missing(video_name):
    """Creates a test video with moving shapes if the actual video file is missing"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(current_dir, 'videos')
    video_path = os.path.join(videos_dir, video_name)
    
    if os.path.exists(video_path):
        return video_path
    
    try:
        print(f"Creating test video {video_name}")
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 720))

        # Create some random shapes that move
        shapes = []
        for i in range(10):  # Create 10 "people"
            # x, y, radius, dx, dy, color
            shape = {
                'x': np.random.randint(50, 1230),
                'y': np.random.randint(50, 670),
                'radius': np.random.randint(20, 40),
                'dx': np.random.randint(-5, 5),
                'dy': np.random.randint(-5, 5),
                'color': (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            }
            if shape['dx'] == 0:
                shape['dx'] = 1
            if shape['dy'] == 0:
                shape['dy'] = 1
            shapes.append(shape)
        
        # Generate 200 frames (10 seconds at 20fps)
        for i in range(200):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Draw "floor name" on the video
            cv2.putText(frame, f"Test Video: {video_name}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Update and draw all shapes
            for shape in shapes:
                # Move the shape
                shape['x'] += shape['dx']
                shape['y'] += shape['dy']
                
                # Bounce off edges
                if shape['x'] <= shape['radius'] or shape['x'] >= 1280 - shape['radius']:
                    shape['dx'] *= -1
                if shape['y'] <= shape['radius'] or shape['y'] >= 720 - shape['radius']:
                    shape['dy'] *= -1
                
                # Draw the shape
                cv2.circle(frame, (shape['x'], shape['y']), shape['radius'], shape['color'], -1)
                # Draw humanoid shape
                cv2.line(frame, (shape['x'], shape['y'] + shape['radius']), 
                        (shape['x'], shape['y'] + shape['radius'] * 3), shape['color'], 2)
                # Arms
                cv2.line(frame, (shape['x'], shape['y'] + shape['radius'] * 1.5), 
                        (shape['x'] - shape['radius'], shape['y'] + shape['radius']), shape['color'], 2)
                cv2.line(frame, (shape['x'], shape['y'] + shape['radius'] * 1.5), 
                        (shape['x'] + shape['radius'], shape['y'] + shape['radius']), shape['color'], 2)
                # Legs
                cv2.line(frame, (shape['x'], shape['y'] + shape['radius'] * 3), 
                        (shape['x'] - shape['radius'], shape['y'] + shape['radius'] * 4), shape['color'], 2)
                cv2.line(frame, (shape['x'], shape['y'] + shape['radius'] * 3), 
                        (shape['x'] + shape['radius'], shape['y'] + shape['radius'] * 4), shape['color'], 2)
            
            # Write the frame
            out.write(frame)
        
        # Release VideoWriter
        out.release()
        print(f"Created test video at {video_path}")
        return video_path
    except Exception as e:
        print(f"Error creating test video: {str(e)}")
        return None

if __name__ == '__main__':
    try:
        # Check if video files exist
        check_video_files()
        
        # Start a thread that will perform people detection
        t = threading.Thread(target=detect_people)
        t.daemon = True
        t.start()
        
        # Initialize detector with default values
        initialize_detector()
        
        port = 49152  # Using a high port number less likely to be blocked
        print(f"Starting server on http://127.0.0.1:{port}")
        app.run(
            host='127.0.0.1',
            port=port,
            debug=True,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        input("Press Enter to exit...")
    finally:
        # Cleanup resources
        if detector and hasattr(detector, 'cap') and detector.cap.isOpened():
            detector.cap.release()
        
        # Release webcam if it was initialized
        if webcam and webcam.isOpened():
            webcam.release()
            
        # Release grid view resources
        for floor_data in grid_frames.values():
            if 'detector' in floor_data and hasattr(floor_data['detector'], 'cap') and floor_data['detector'].cap.isOpened():
                floor_data['detector'].cap.release()