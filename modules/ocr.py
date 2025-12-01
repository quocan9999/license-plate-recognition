"""
Module OCR cho nhận diện ký tự biển số xe
Sử dụng EasyOCR với Warping (nắn thẳng biển số)
"""


import re
import easyocr
from .preprocessing import preprocess_for_ocr
from .utils import classify_vehicle, fix_plate_chars, format_plate
from .config import OCR_LANGUAGES, OCR_GPU


class LicensePlateOCR:
    """
    Class OCR cho nhận diện ký tự biển số xe Việt Nam
    Sử dụng EasyOCR với Warping
    """
    
    def __init__(self, languages=OCR_LANGUAGES, gpu=OCR_GPU):
        """
        Khởi tạo EasyOCR reader
        
        Args:
            languages: Danh sách ngôn ngữ hỗ trợ
            gpu: Sử dụng GPU hay không
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print(f"✓ Đã khởi tạo EasyOCR (GPU: {gpu}) với Warping")
    
    def read_text(self, image, detail=0):
        """
        Đọc text từ ảnh sử dụng EasyOCR
        
        Args:
            image: Ảnh đầu vào (numpy array)
            detail: 0 = chỉ text, 1 = full detail
            
        Returns:
            List các dòng text đã nhận diện
        """
        return self.reader.readtext(image, detail=detail)
    
    def process_plate(self, roi, apply_warping=True):
        """
        Xử lý và nhận diện biển số từ ROI
        
        Args:
            roi: Ảnh vùng biển số (numpy array)
            apply_warping: Có áp dụng warping (nắn thẳng) hay không
            
        Returns:
            Dict chứa thông tin biển số:
                - raw_text: Text thô từ OCR
                - vehicle_type: Loại xe
                - clean_text: Text đã sửa lỗi
                - formatted_text: Text đã format
                - is_50cc: Có phải xe máy 50cc không
        """
        # Tiền xử lý ảnh (bao gồm warping nếu apply_warping=True)
        preprocessed, method, intermediates = preprocess_for_ocr(roi, apply_warping=apply_warping)
        
        # OCR với EasyOCR
        ocr_result = self.read_text(preprocessed, detail=0)
        
        if len(ocr_result) == 0:
            return None
        
        # Phân loại loại xe
        vehicle_type = classify_vehicle(ocr_result)
        
        # Kiểm tra xe máy 50cc
        is_50cc = False
        if vehicle_type == "XE MÁY":
            line1 = ocr_result[0]
            line1_clean = re.sub(r'[^A-Z0-9]', '', line1.upper())
            # Nếu dòng 1 kết thúc bằng chữ và dài >= 4 -> xe máy 50cc
            if len(line1_clean) >= 4 and not line1_clean[-1].isdigit():
                is_50cc = True
        
        # Ghép và sửa lỗi
        raw_text = "".join(ocr_result)
        clean_text = fix_plate_chars(raw_text, is_50cc=is_50cc)
        formatted_text = format_plate(clean_text, vehicle_type)
        
        return {
            'raw_text': raw_text,
            'vehicle_type': vehicle_type,
            'clean_text': clean_text,
            'formatted_text': formatted_text,
            'is_50cc': is_50cc,
            'ocr_lines': ocr_result,
            'preprocessed_image': preprocessed,
            'preprocessing_method': method,
            'intermediate_images': intermediates
        }
    
    def is_valid_plate(self, plate_info):
        """
        Kiểm tra biển số có hợp lệ không
        
        Args:
            plate_info: Dict thông tin biển số từ process_plate()
            
        Returns:
            True nếu hợp lệ, False nếu không
        """
        if plate_info is None:
            return False
        
        formatted_text = plate_info.get('formatted_text', '')
        
        # Kiểm tra độ dài tối thiểu
        if len(formatted_text) <= 5:
            return False
        
        # Kiểm tra loại xe
        vehicle_type = plate_info.get('vehicle_type', '')
        if vehicle_type == "KHÔNG RÕ":
            return False
        
        return True
