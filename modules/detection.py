"""
Module phát hiện biển số xe sử dụng YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO


class LicensePlateDetector:
    """
    Class phát hiện biển số xe sử dụng YOLO model
    """
    
    def __init__(self, model_path="models/best.pt", fallback_model="yolov8n.pt"):
        """
        Khởi tạo detector với YOLO model
        
        Args:
            model_path: Đường dẫn đến model custom
            fallback_model: Model dự phòng nếu không load được model custom
        """
        self.model = None
        self.model_path = model_path
        self.fallback_model = fallback_model
        self.load_model()
    
    def load_model(self):
        """
        Load YOLO model
        """
        try:
            self.model = YOLO(self.model_path)
            print(f"✓ Đã load model custom: {self.model_path}")
        except Exception as e:
            print(f"⚠ Không load được model custom: {e}")
            try:
                self.model = YOLO(self.fallback_model)
                print(f"✓ Đã load model dự phòng: {self.fallback_model}")
            except Exception as e2:
                print(f"✗ Lỗi khi load model: {e2}")
                raise
    
    def detect(self, image):
        """
        Phát hiện biển số trong ảnh
        
        Args:
            image: Ảnh đầu vào (PIL Image hoặc numpy array)
            
        Returns:
            results: Kết quả detection từ YOLO
        """
        if self.model is None:
            raise RuntimeError("Model chưa được load!")
        
        # Chuyển đổi PIL Image sang numpy array nếu cần
        if hasattr(image, 'mode'):  # PIL Image
            image_np = np.array(image)
        else:
            image_np = image
        
        # Thực hiện detection
        results = self.model(image_np)
        return results
    
    def get_plate_regions(self, image):
        """
        Lấy các vùng ROI (Region of Interest) của biển số
        
        Args:
            image: Ảnh đầu vào (PIL Image hoặc numpy array)
            
        Returns:
            List các tuple (roi, bbox) với:
                - roi: Ảnh vùng biển số (numpy array)
                - bbox: Tọa độ bounding box (x1, y1, x2, y2)
        """
        # Chuyển đổi PIL Image sang numpy array nếu cần
        if hasattr(image, 'mode'):  # PIL Image
            image_np = np.array(image)
        else:
            image_np = image
        
        results = self.detect(image_np)
        plate_regions = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = image_np[y1:y2, x1:x2]
                bbox = (x1, y1, x2, y2)
                plate_regions.append((roi, bbox))
        
        return plate_regions
    
    def draw_detections(self, image, detections, color=(0, 255, 0), thickness=3):
        """
        Vẽ các detection lên ảnh
        
        Args:
            image: Ảnh đầu vào (numpy array)
            detections: List các detection (bbox, text, vehicle_type)
            color: Màu của bounding box
            thickness: Độ dày của bounding box
            
        Returns:
            Ảnh đã được vẽ detection
        """
        image_copy = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            text = detection.get('text', '')
            vehicle_type = detection.get('vehicle_type', '')
            
            x1, y1, x2, y2 = bbox
            
            # Chọn màu theo loại xe
            if vehicle_type == "XE MÁY":
                box_color = (0, 255, 0)  # Xanh lá
            elif vehicle_type == "Ô TÔ":
                box_color = (0, 165, 255)  # Cam
            else:
                box_color = color
            
            # Vẽ bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), box_color, thickness)
            
            # Vẽ text nếu có
            if text:
                # Xóa các ký tự định dạng để vẽ lên ảnh
                display_text = text.replace("-", "").replace(".", "")
                
                # Tính kích thước chữ để vẽ nền
                (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(image_copy, (x1, y1 - 40), (x1 + w, y1), box_color, -1)
                
                # Vẽ text
                cv2.putText(image_copy, display_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return image_copy
