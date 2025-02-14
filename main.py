import os
import argparse
from people_detector import PeopleDetector

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pedestrian Detection using YOLO")
    parser.add_argument('--video', type=str, default='railwayvid.mp4', help="Path to the video file.")
    parser.add_argument('--model', type=str, default='yolov5s.pt', help="Path to the YOLO model.")
    parser.add_argument('--output', type=str, default='output.avi', help="Path to save the processed video.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    if not os.path.exists(args.video):
        raise ValueError(f"Error: The video source '{args.video}' does not exist.")

    detector = PeopleDetector(model_path=args.model, video_source=args.video)
    detector.process_video(output_path=args.output)

if __name__ == "__main__":
    main()
