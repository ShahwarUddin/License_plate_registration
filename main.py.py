import cv2
import numpy as np
import os
import base64
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError  # Import this at the top
from dotenv import load_dotenv
import traceback
import cloudinary
import cloudinary.uploader
import io

load_dotenv()

class VehiclePlateRecognition:
    def __init__(self, camera_id, **kwargs):
        super().__init__(**kwargs)
        self.camera_id = camera_id
        self.camera_category = self.get_camera_category(camera_id)
        self.collection = self.connect_to_db()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.vehicle_model = YOLO("../yolov8n.pt")
        self.plate_model = YOLO("../model.pt")
        self.VEHICLE_CLASSES = {2, 3, 5, 7}  # COCO dataset classes
        
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET")
        )
    def get_camera_category(self, camera_id):
        if camera_id in [1, 2]:
            return "enter"
        elif camera_id in [4, 5]:
            return "exit"
        else:
            return "in"


    def connect_to_db(self):
        """Connects to MongoDB and returns the collection object."""
        try:
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri:
                raise ValueError("MONGO_URI is not set in the .env file")

            client = MongoClient(mongo_uri)
            db = client["numberplates"]
            collection = db["plate_records"]

            # Ensure 'numberplate' field is unique to prevent duplicates
            collection.create_index("numberplate", unique=True)

            return collection
        except Exception as err:
            print(f"Error connecting to database: {err}")
            raise



    def perform_ocr(self, image_array):
        """Performs OCR on the given image and returns the extracted text."""
        if image_array is None:
            raise ValueError("Image is None")
        if isinstance(image_array, np.ndarray):
            results = self.ocr.ocr(image_array, rec=True)
        else:
            raise TypeError("Input image is not a valid numpy array")
        
        text = ' '.join([result[1][0] for result in results[0]] if results[0] else "")
        pattern = r"^[A-Za-z]{2,3}-\d{3,4}$"
        return text if re.match(pattern, text) else ""
    
    def upload_to_cloudinary(self, image_array):
        """Uploads an OpenCV image array to Cloudinary and returns the image URL."""
        try:
            _, buffer = cv2.imencode(".jpg", image_array)
            image_file = io.BytesIO(buffer.tobytes())
            response = cloudinary.uploader.upload(image_file, resource_type="image")
            return response["secure_url"]
        except Exception as e:
            print(f"Error uploading image: {e}")
            return None


# improve checking system
    def save_to_database(self, date, time, class_name, numberplate, vehicle_image_path, plate_image_path):
        """Saves the detected number plate and images to MongoDB, preventing duplicates using unique index."""
        vehicle_image_url = self.upload_to_cloudinary(vehicle_image_path)
        plate_image_url = self.upload_to_cloudinary(plate_image_path)
        
        record = {
            "date": date,
            "time": time,
            "class_name": class_name,
            "numberplate": numberplate,
            "registration_number": numberplate,
            "state": "SD",
            "district": "Karachi",
            "camera_id": self.camera_id,
            "camera_category": self.camera_category,
            "vehicle_image": vehicle_image_url,
            "plate_image": plate_image_url 
        }
        
        try:
            self.collection.insert_one(record)
            print(f"New number plate {numberplate} saved to MongoDB.")
        except DuplicateKeyError:
            print(f"Number plate {numberplate} already exists. Skipping insertion.")
        except Exception as err:
            print(f"Error saving to database: {err}")
            traceback.print_exc()

        
    def process_video(self, video_path):
        """Processes a video to detect vehicles and number plates."""
        cap = cv2.VideoCapture(video_path)
        c = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            c += 1
            if c % 5 != 0:
                continue

            vehicle_results = self.vehicle_model(frame)[0]
            vehicle_boxes = vehicle_results.boxes.xyxy.cpu().numpy()
            vehicle_classes = vehicle_results.boxes.cls.cpu().numpy()
            vehicle_confidences = vehicle_results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls, conf in zip(vehicle_boxes, vehicle_classes, vehicle_confidences):
                if int(cls) not in self.VEHICLE_CLASSES or conf < 0.50:
                    continue
                
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                vehicle_roi = frame[y1:y2, x1:x2]
                plate_results = self.plate_model(vehicle_roi)[0]
                plate_boxes = plate_results.boxes.xyxy.cpu().numpy()
                
                for (px1, py1, px2, py2) in plate_boxes:
                    px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
                    px1, px2 = px1 + x1, px2 + x1
                    py1, py2 = py1 + y1, py2 + y1
                    plate_roi = frame[py1:py2, px1:px2]
                    
                    text = self.perform_ocr(plate_roi)
                    if text:
                        date = datetime.now().strftime("%Y-%m-%d")
                        time = datetime.now().strftime("%H:%M:%S")
                        self.save_to_database(date, time, f"Vehicle {int(cls)}", text, vehicle_roi, plate_roi)
                        
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                        cv2.putText(frame, text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_id=1
    recognizer = VehiclePlateRecognition(camera_id=camera_id)
    
