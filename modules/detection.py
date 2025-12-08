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

    def _preprocess_image(self, image):
        """
        Chuyển đổi ảnh sang định dạng numpy RGB chuẩn
        """
        # Chuyển đổi PIL Image sang numpy array nếu cần
        if hasattr(image, 'mode'):  # PIL Image
            image_np = np.array(image)
        else:
            image_np = image
        
        # Kiểm tra nếu ảnh là grayscale (2 chiều) -> convert sang RGB
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        # Kiểm tra nếu ảnh có 4 kênh (RGBA) -> convert sang RGB
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
        return image_np

    def detect(self, image, image_index=None):
        """
        Phát hiện biển số trong ảnh
        
        Args:
            image: Ảnh đầu vào (PIL Image hoặc numpy array)
            image_index: Số thứ tự ảnh (optional)
            
        Returns:
            results: Kết quả detection từ YOLO
        """
        if self.model is None:
            raise RuntimeError("Model chưa được load!")
        
        image_np = self._preprocess_image(image)
        
        # Thực hiện detection với verbose=False để tắt output tự động
        # Thêm conf=0.25 để lọc các box có độ tin cậy thấp
        # Thêm classes=[0] để chỉ nhận diện class 0 (biển số)
        results = self.model(image_np, conf=0.25, classes=[0], verbose=False)
        
        # In thông tin detection với STT tùy chỉnh
        if results and len(results) > 0:
            result = results[0]  # Lấy kết quả đầu tiên
            if hasattr(result, 'boxes') and result.boxes is not None:
                num_detections = len(result.boxes)
                orig_height, orig_width = image_np.shape[:2]
                
                # Lấy kích thước inference từ model (thường là 640x640 cho YOLOv8)
                model_imgsz = getattr(self.model, 'imgsz', 640)
                if isinstance(model_imgsz, (list, tuple)):
                    yolo_size = f"{model_imgsz[0]}x{model_imgsz[1]}" if len(model_imgsz) > 1 else f"{model_imgsz[0]}x{model_imgsz[0]}"
                else:
                    yolo_size = f"{model_imgsz}x{model_imgsz}"
                
                if image_index is not None:
                    print(f"{image_index}: {orig_width}x{orig_height} (resized to: {yolo_size}) {num_detections} bien_so")
                else:
                    print(f"0: {orig_width}x{orig_height} (resized to: {yolo_size}) {num_detections} bien_so")
        
        return results
    
    def get_plate_regions(self, image, image_index=None):
        """
        Lấy các vùng ROI (Region of Interest) của biển số
        
        Args:
            image: Ảnh đầu vào (PIL Image hoặc numpy array)
            image_index: Số thứ tự ảnh (optional)
            
        Returns:
            List các tuple (roi, bbox) với:
                - roi: Ảnh vùng biển số (numpy array)
                - bbox: Tọa độ bounding box (x1, y1, x2, y2)
        """
        image_np = self._preprocess_image(image)
        
        results = self.detect(image_np, image_index=image_index)
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
        img_h, img_w = image_copy.shape[:2]
        
        # Tính toán scale factor dựa trên kích thước ảnh
        scale_factor = max(1.0, img_w / 640.0)
        
        # Lưu lại các vùng đã vẽ text để tránh đè lên nhau
        used_text_regions = []
        
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
                dynamic_font_scale = max(0.8, TEXT_FONT_SCALE * scale_factor * 1.2)
                dynamic_text_thickness = max(2, int(TEXT_THICKNESS * scale_factor * 0.8))

                # Tính kích thước chữ để vẽ nền
                (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, dynamic_text_thickness)
                
                padding = int(3 * scale_factor)
                
                # Tím vị trí tốt nhất để vẽ text (không bị đè)
                # Thử các vị trí theo thứ tự ưu tiên: trên, dưới, trái, phải
                positions = [
                    (x1, y1 - h - padding * 2, 'top'),      # Phía trên bbox
                    (x1, y2 + h + padding * 2, 'bottom'),   # Phía dưới bbox
                    (x1, y1 + h + padding, 'inside_top'),   # Bên trong bbox (ở trên)
                ]
                
                best_pos = None
                for text_x, text_y, pos_type in positions:
                    # Điều chỉnh nếu text vượt quá biên ảnh
                    if text_x + w > img_w:
                        text_x = img_w - w - 5
                    if text_x < 0:
                        text_x = 5
                    if text_y < h + padding:
                        continue  # Bỏ qua nếu vượt quá cạnh trên
                    if text_y > img_h - padding:
                        continue  # Bỏ qua nếu vượt quá cạnh dưới
                    
                    # Tạo vùng text
                    text_region = (text_x, text_y - h - padding, text_x + w + padding, text_y + padding)
                    
                    # Kiểm tra xem có bị đè lên các vùng đã vẽ không
                    is_overlapping = False
                    for used_region in used_text_regions:
                        if self._regions_overlap(text_region, used_region):
                            is_overlapping = True
                            break
                    
                    if not is_overlapping:
                        best_pos = (text_x, text_y, text_region)
                        break
                
                # Nếu tất cả vị trí đều bị đè, sử dụng vị trí mặc định
                if best_pos is None:
                    text_x = x1
                    text_y = y1 - h - padding * 2 if y1 > h + padding * 2 else y2 + h + padding * 2
                    text_region = (text_x, text_y - h - padding, text_x + w + padding, text_y + padding)
                    best_pos = (text_x, text_y, text_region)
                
                text_x, text_y, text_region = best_pos
                used_text_regions.append(text_region)
                
                # Vẽ nền cho text
                cv2.rectangle(image_copy, 
                             (int(text_region[0]), int(text_region[1])), 
                             (int(text_region[2]), int(text_region[3])), 
                             box_color, -1)
                
                # Vẽ text
                cv2.putText(image_copy, display_text, (int(text_x), int(text_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale, (255, 255, 255), dynamic_text_thickness)
        
        return image_copy
    
    def _regions_overlap(self, region1, region2):
        """
        Kiểm tra xem 2 vùng có giao nhau không
        Mỗi region là tuple (x1, y1, x2, y2)
        """
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2
        
        # Kiểm tra không giao nhau
        if x2_1 < x1_2 or x2_2 < x1_1:
            return False
        if y2_1 < y1_2 or y2_2 < y1_1:
            return False
        return True
        
        return image_copy
