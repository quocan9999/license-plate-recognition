"""
Module phát hiện biển số xe sử dụng YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO
from .config import (
    MODEL_PATH, 
    FALLBACK_MODEL_PATH, 
    COLOR_DEFAULT, 
    COLOR_MOTO, 
    COLOR_CAR, 
    BBOX_THICKNESS,
    TEXT_FONT_SCALE,
    TEXT_THICKNESS
)


class LicensePlateDetector:
    """
    Class phát hiện biển số xe sử dụng YOLO model
    """
    
    def __init__(self, model_path=MODEL_PATH, fallback_model=FALLBACK_MODEL_PATH):
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
    
    def draw_detections(self, image, detections, color=COLOR_DEFAULT, thickness=BBOX_THICKNESS):
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
        num_detections = len(detections)
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            text = detection.get('text', '')
            vehicle_type = detection.get('vehicle_type', '')
            
            x1, y1, x2, y2 = bbox
            
            # Chọn màu theo loại xe
            if vehicle_type == "XE MÁY":
                box_color = COLOR_MOTO
            elif vehicle_type == "Ô TÔ":
                box_color = COLOR_CAR
            else:
                box_color = color
            
            # Tính toán scale dựa trên kích thước ảnh
            img_h, img_w = image_copy.shape[:2]
            
            # Scale factor: chuẩn hóa theo chiều rộng 640px (kích thước chuẩn của YOLO)
            # Nếu ảnh rộng 640px -> scale = 1.0
            # Nếu ảnh rộng 1920px -> scale = 3.0
            scale_factor = max(1.0, img_w / 640.0)
            
            # Tính độ dày nét vẽ động
            dynamic_thickness = max(2, int(thickness * scale_factor))
            
            # Vẽ bounding box với độ dày động
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), box_color, dynamic_thickness)
            
            # Vẽ text nếu có
            if text:
                # Xóa các ký tự định dạng để vẽ lên ảnh
                display_text = text.replace("-", "").replace(".", "")
                
                # Thêm số thứ tự nếu có nhiều hơn 1 biển số
                if num_detections > 1:
                    display_text = f"#{i+1} {display_text}"
                
                # Tính font scale và độ dày chữ động
                dynamic_font_scale = max(1.2, TEXT_FONT_SCALE * scale_factor * 1.8)
                dynamic_text_thickness = max(2, int(TEXT_THICKNESS * scale_factor))

                # Tính kích thước chữ để vẽ nền
                (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, dynamic_text_thickness)
                
                # Tính toán vị trí vẽ text
                text_x = x1
                text_y = y1 - 10
                
                # Nếu text bị che ở cạnh trên (y1 quá nhỏ) -> vẽ xuống dưới box
                if y1 < h + 10:
                    text_y = y2 + h + 10
                    
                # Nếu text bị che ở cạnh phải (x1 + w quá lớn) -> dời sang trái
                if text_x + w > img_w:
                    text_x = img_w - w - 5
                
                # Vẽ nền cho text
                # Điều chỉnh tọa độ nền dựa trên vị trí text_y
                padding = int(5 * scale_factor)
                cv2.rectangle(image_copy, (text_x, text_y - h - padding), (text_x + w, text_y + padding), box_color, -1)
                
                # Vẽ text
                cv2.putText(image_copy, display_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, (255, 255, 255), dynamic_text_thickness)
        
        return image_copy
