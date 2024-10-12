import os
from people_detector import PeopleDetector

def main():
    video_source = './IMG_4561.mp4' 
    if not os.path.exists(video_source):
        raise ValueError(f"Error: The video source '{video_source}' does not exist.")

    detector = PeopleDetector(model_version='face', video_source=video_source)

    detector.process_video()

if __name__ == "__main__":
    main()
