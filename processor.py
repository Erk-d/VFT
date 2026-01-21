import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

class VideoProcessor:
    def __init__(self, video_path, sample_rate=10, output_dir="data/crops"):
        self.video_path = video_path
        self.sample_rate = sample_rate # Process every Nth frame
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process(self):
        """
        Reads video and returns a list of dictionaries:
        [{'embedding': np.array, 'timestamp': float, 'crop_path': str}, ...]
        """
        if not os.path.exists(self.video_path):
            print(f"Error: File {self.video_path} not found.")
            return []

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        detections = []
        
        print(f"Processing {self.video_path} at {fps} FPS (sampling every {self.sample_rate} frames)...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % self.sample_rate != 0:
                continue

            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb_frame = frame[:, :, ::-1] # Optimized conversion (faster than cvtColor) or just frame[:,:,::-1]
            
            # Detect faces
            # model="hog" is faster than "cnn" (but less accurate). Use "hog" for CPU speed.
            video_timestamp = frame_count / fps
            
            # Reduce image size for faster detection if resolution is huge (e.g. 4k)
            # small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5) 
            # locations = face_recognition.face_locations(small_frame)
            # For now, full RES for accuracy:
            locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            if not locations:
                continue
                
            encodings = face_recognition.face_encodings(rgb_frame, locations)

            for box, enc in zip(locations, encodings):
                top, right, bottom, left = box
                
                # Save crop
                face_image = frame[top:bottom, left:right]
                crop_filename = f"crop_{int(datetime.now().timestamp())}_{frame_count}_{top}.jpg"
                crop_path = os.path.join(self.output_dir, crop_filename)
                cv2.imwrite(crop_path, face_image)
                
                detections.append({
                    'embedding': enc,
                    'timestamp': video_timestamp,
                    'crop_path': crop_path,
                    'box': box
                })
        
        cap.release()
        print(f"Finished processing. Found {len(detections)} face appearances.")
        return detections
