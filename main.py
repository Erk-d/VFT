import sys
import os
import database
from processor import VideoProcessor
from profiler import Profiler

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_video> [sample_rate]")
        return

    video_path = sys.argv[1]
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 10 # Default skip 10 frames

    if not os.path.isfile(video_path):
        print(f"File not found: {video_path}")
        return

    print("--- Video Face Tracker ---")
    
    # 1. Init DB
    if not os.path.exists("data"):
        os.makedirs("data")
    database.init_db()
    
    # 2. Register Video
    print(f"Registering video: {video_path}")
    video_id = database.add_video(video_path)
    
    # 3. Process
    processor = VideoProcessor(video_path, sample_rate=sample_rate, output_dir="data/crops")
    detections = processor.process()
    
    if not detections:
        print("No faces found or video failed to load.")
        return

    # 4. Profile & Match
    print("Profiling and Clustering...")
    profiler = Profiler()
    profiler.run(video_id, detections)
    
    print("Done! Data saved to data/face_tracker.db")

if __name__ == "__main__":
    main()
