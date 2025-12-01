"""
Module quản lý việc ghi log và lưu lịch sử nhận diện
"""

import os
import csv
from datetime import datetime
from PIL import Image
import numpy as np

class HistoryLogger:
    """
    Class quản lý việc lưu trữ lịch sử nhận diện, bao gồm ảnh và file CSV
    """
    
    def __init__(self, base_dir="History"):
        """
        Khởi tạo logger
        
        Args:
            base_dir: Thư mục gốc để lưu lịch sử
        """
        self.base_dir = base_dir
        # Đảm bảo thư mục gốc tồn tại
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def save_result(self, original_image_path, original_image_pil, detections):
        """
        Lưu kết quả nhận diện vào thư mục History và ghi log CSV
        
        Args:
            original_image_path: Đường dẫn file ảnh gốc
            original_image_pil: Ảnh gốc (PIL Image)
            detections: Danh sách kết quả nhận diện
        """
        try:
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            
            # 1. Lưu ảnh gốc
            # Lấy tên file gốc để dễ truy xuất
            original_filename = os.path.basename(original_image_path)
            name_no_ext = os.path.splitext(original_filename)[0]
            
            # Tạo thư mục riêng cho ảnh này: History/{Timestamp}_{OriginalName}
            image_folder_name = f"{timestamp}_{name_no_ext}"
            save_dir = os.path.join(self.base_dir, image_folder_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Tên file: YYYYMMDD_HHMMSS_OriginalName.jpg
            save_original_name = f"{timestamp}_{name_no_ext}.jpg"
            save_original_path = os.path.join(save_dir, save_original_name)
            
            # Lưu ảnh gốc
            original_image_pil.save(save_original_path)
            
            # File log CSV
            csv_file = os.path.join(self.base_dir, "history.csv")
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                # Header
                if not file_exists:
                    writer.writerow(['Timestamp', 'License Plate', 'Vehicle Type', 'Original Image Path', 'ROI Image Path', 'Preprocessed Image Path'])
                
                # Nếu không có biển số nào
                if not detections:
                     writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), "No Plate", "", save_original_path, "", ""])
                
                # 2. Lưu từng biển số cắt được (ROI)
                for i, det in enumerate(detections):
                    plate_text = det['text']
                    vehicle_type = det['vehicle_type']
                    roi = det['roi'] # numpy array (RGB)
                    preprocessed_image = det.get('preprocessed_image')
                    preprocessing_method = det.get('preprocessing_method', 'unknown')
                    intermediate_images = det.get('intermediate_images', {})
                    
                    # Clean text cho tên file
                    clean_text = "".join(c for c in plate_text if c.isalnum())
                    
                    # Tên file ROI: YYYYMMDD_HHMMSS_BienSo_Index.jpg
                    save_roi_name = f"{timestamp}_{clean_text}_{i}.jpg"
                    save_roi_path = os.path.join(save_dir, save_roi_name)
                    
                    # Lưu ảnh ROI
                    roi_pil = Image.fromarray(roi)
                    roi_pil.save(save_roi_path)
                    
                    # Lưu ảnh Preprocessed (nếu có)
                    save_preprocessed_path = ""
                    if preprocessed_image is not None:
                        # 1. Lưu ảnh kết quả cuối cùng: ..._processed.jpg
                        save_final_name = f"{timestamp}_{clean_text}_{i}_processed.jpg"
                        save_preprocessed_path = os.path.join(save_dir, save_final_name)
                        Image.fromarray(preprocessed_image).save(save_preprocessed_path)
                        
                        # 2. Lưu từng bước trung gian
                        if intermediate_images:
                            for step_name, step_img in intermediate_images.items():
                                save_step_name = f"{timestamp}_{clean_text}_{i}_processed_{step_name}.jpg"
                                save_step_path = os.path.join(save_dir, save_step_name)
                                Image.fromarray(step_img).save(save_step_path)
                    
                    # Ghi log
                    writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), plate_text, vehicle_type, save_original_path, save_roi_path, save_preprocessed_path])
                    
        except Exception as e:
            print(f"Lỗi khi lưu lịch sử: {e}")
