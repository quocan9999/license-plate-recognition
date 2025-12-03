"""
Module OCR cho nhận diện ký tự biển số xe
Sử dụng EasyOCR với Warping (nắn thẳng biển số)
"""


import re
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import easyocr
from .preprocessing import preprocess_for_ocr
from .utils import classify_vehicle, fix_plate_chars, format_plate
from .config import OCR_LANGUAGES, OCR_GPU


class LicensePlateOCR:
    """
    Class OCR cho nhận diện ký tự biển số xe Việt Nam
    Sử dụng EasyOCR với Warping
    """
    
    def __init__(self, languages: List[str] = OCR_LANGUAGES, gpu: bool = OCR_GPU):
        """
        Khởi tạo EasyOCR reader
        
        Args:
            languages: Danh sách ngôn ngữ hỗ trợ
            gpu: Sử dụng GPU hay không
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print(f"✓ Đã khởi tạo EasyOCR (GPU: {gpu}) với Warping")
    
    def read_text(self, image: np.ndarray, detail: int = 1) -> List[Any]:
        """
        Đọc text từ ảnh sử dụng EasyOCR
        
        Args:
            image: Ảnh đầu vào (numpy array)
            detail: 0 = chỉ text, 1 = full detail (bbox, text, conf)
            
        Returns:
            List kết quả
        """
        return self.reader.readtext(image, detail=detail)
    
    def _sort_ocr_results_top_to_bottom(self, ocr_output: List[Any]) -> List[Any]:
        """
        Sắp xếp kết quả OCR theo thứ tự từ trên xuống dưới, trái qua phải
        
        Đối với biển số 2 dòng, cần đọc dòng trên trước, sau đó dòng dưới.
        
        Args:
            ocr_output: Kết quả từ EasyOCR [[bbox, text, conf], ...]
            
        Returns:
            Kết quả đã được sắp xếp
        """
        if len(ocr_output) == 0:
            return ocr_output
        
        # Sắp xếp theo tọa độ Y (top) của bbox, sau đó theo X (left)
        # bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Lấy y_center = (y1 + y3) / 2, x_center = (x1 + x3) / 2
        
        def get_sort_key(item):
            bbox = item[0]
            # Tính tọa độ trung tâm
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            x_center = (bbox[0][0] + bbox[2][0]) / 2
            # Sắp xếp theo Y trước (trên -> dưới), sau đó X (trái -> phải)
            return (y_center, x_center)
        
        sorted_output = sorted(ocr_output, key=get_sort_key)
        return sorted_output
    

    def _process_ocr_result(self, ocr_output: List[Any], preprocessed: np.ndarray, method: str, intermediates: Dict[str, np.ndarray]) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Xử lý kết quả raw từ EasyOCR -> plate_info
        """
        if len(ocr_output) == 0:
            return None, 0.0
        
        # Sắp xếp kết quả OCR theo thứ tự từ trên xuống dưới, trái qua phải
        ocr_output = self._sort_ocr_results_top_to_bottom(ocr_output)
            
        # Tách text và confidence
        # ocr_output format: [[bbox, text, conf], ...]
        text_lines = [item[1] for item in ocr_output]
        confidences = [item[2] for item in ocr_output]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Phân loại loại xe
        vehicle_type = classify_vehicle(text_lines)
        
        # Kiểm tra xe máy 50cc
        is_50cc = False
        if vehicle_type == "XE MÁY":
            line1 = text_lines[0]
            line1_clean = re.sub(r'[^A-Z0-9]', '', line1.upper())
            if len(line1_clean) >= 4 and not line1_clean[-1].isdigit():
                is_50cc = True
        
        # Ghép và sửa lỗi
        raw_text = "".join(text_lines)
        clean_text = fix_plate_chars(raw_text, is_50cc=is_50cc)
        formatted_text = format_plate(clean_text, vehicle_type)
        
        plate_info = {
            'raw_text': raw_text,
            'vehicle_type': vehicle_type,
            'clean_text': clean_text,
            'formatted_text': formatted_text,
            'is_50cc': is_50cc,
            'ocr_lines': text_lines,
            'preprocessed_image': preprocessed,
            'preprocessing_method': method,
            'intermediate_images': intermediates,
            'confidence': avg_conf
        }
        
        return plate_info, avg_conf

    def process_plate(self, roi: np.ndarray, apply_warping: bool = True) -> Optional[Dict[str, Any]]:
        """
        Xử lý và nhận diện biển số từ ROI
        Chiến lược: Multi-Hypothesis (Thử nhiều cách tiền xử lý và chọn kết quả tốt nhất)
        """
        # Lấy danh sách các phiên bản ảnh đã tiền xử lý
        variants = preprocess_for_ocr(roi, apply_warping=apply_warping)
        
        candidates = []
        
        for image, method in variants:
            # OCR
            ocr_output = self.read_text(image, detail=1)
            
            # Xử lý kết quả
            # Lưu ý: intermediates giờ không còn trả về từ preprocess_for_ocr, 
            # nên ta truyền dict rỗng hoặc tạo dict chứa ảnh hiện tại để debug
            intermediates = {method: image}
            
            plate_info, conf = self._process_ocr_result(ocr_output, image, method, intermediates)
            
            if plate_info and self.is_valid_plate(plate_info):
                candidates.append(plate_info)
                
        # Chọn kết quả tốt nhất
        if not candidates:
            return None
            
        # Sắp xếp theo confidence giảm dần
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        best_result = candidates[0]
        
        # Log debug
        if len(candidates) > 1:
            print(f"Selected '{best_result['preprocessing_method']}' ({best_result['confidence']:.2f}) from {len(candidates)} candidates.")
            
        return best_result
    
    def is_valid_plate(self, plate_info: Optional[Dict[str, Any]]) -> bool:
        """
        Kiểm tra biển số có hợp lệ không
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
