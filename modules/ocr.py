"""
Module OCR cho nhận diện ký tự biển số xe
Sử dụng PaddleOCR với Warping (nắn thẳng biển số)
"""


import re
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from paddleocr import PaddleOCR
from .preprocessing import preprocess_for_ocr
from .utils import classify_vehicle, fix_plate_chars, format_plate
from .config import OCR_LANGUAGES, OCR_GPU


class LicensePlateOCR:
    """
    Class OCR cho nhận diện ký tự biển số xe Việt Nam
    Sử dụng PaddleOCR với Warping
    """
    
    def __init__(self, languages: List[str] = OCR_LANGUAGES, gpu: bool = OCR_GPU):
        """
        Khởi tạo PaddleOCR reader
        
        Args:
            languages: Danh sách ngôn ngữ hỗ trợ (PaddleOCR sử dụng 'en', 'vi', etc.)
            gpu: Sử dụng GPU hay không
        """
        # PaddleOCR hỗ trợ nhiều ngôn ngữ: 'en' (English), 'vi' (Vietnamese), etc.
        # Sử dụng 'en' cho biển số xe vì chủ yếu là ký tự Latin và số
        use_gpu = gpu if gpu else False
        
        # Xác định ngôn ngữ cho PaddleOCR
        # PaddleOCR chỉ hỗ trợ 1 ngôn ngữ tại một thời điểm
        lang = 'en'  # Mặc định tiếng Anh cho biển số xe
        if isinstance(languages, list) and len(languages) > 0:
            if 'vi' in languages:
                lang = 'vi'
            elif 'en' in languages:
                lang = 'en'
        elif isinstance(languages, str):
            lang = languages
        
        # PaddleOCR v3.x sử dụng API mới
        # Tắt các tính năng không cần thiết để tăng tốc
        self.reader = PaddleOCR(
            use_doc_orientation_classify=False,  # Tắt phân loại hướng tài liệu
            use_doc_unwarping=False,  # Tắt unwarping tài liệu (đã xử lý trước đó)
            use_textline_orientation=False,  # Tắt xoay text tự động
            lang=lang,
            text_det_thresh=0.3,  # Ngưỡng phát hiện text
            text_det_box_thresh=0.5,  # Ngưỡng box
        )
        self.lang = lang
        print(f"✓ Đã khởi tạo PaddleOCR (GPU: {use_gpu}, Lang: {lang}) với Warping")
    
    def read_text(self, image: np.ndarray, detail: int = 1) -> List[Any]:
        """
        Đọc text từ ảnh sử dụng PaddleOCR
        
        Args:
            image: Ảnh đầu vào (numpy array)
            detail: 0 = chỉ text, 1 = full detail (bbox, text, conf)
            
        Returns:
            List kết quả theo format: [[bbox, text, conf], ...]
        """
        # Đảm bảo ảnh là numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # PaddleOCR yêu cầu ảnh 3 channel (RGB/BGR)
        # Chuyển đổi ảnh grayscale sang BGR nếu cần
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # PaddleOCR v3.x sử dụng method predict()
        # Output format: [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...], ...}]
        result = list(self.reader.predict(image))
        
        # Chuyển đổi kết quả PaddleOCR v3.x sang format tương thích EasyOCR
        # EasyOCR format: [[bbox, text, conf], ...]
        converted_result = []
        
        if result and len(result) > 0:
            res = result[0]  # Lấy kết quả đầu tiên
            rec_texts = res.get('rec_texts', [])
            rec_scores = res.get('rec_scores', [])
            rec_polys = res.get('rec_polys', [])
            
            for i, text in enumerate(rec_texts):
                bbox = rec_polys[i].tolist() if i < len(rec_polys) else []
                conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                
                # Lọc chỉ giữ lại ký tự A-Z và 0-9
                allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                filtered_text = ''.join(c for c in text.upper() if c in allowlist)
                
                if filtered_text:  # Chỉ thêm nếu có text hợp lệ
                    converted_result.append([bbox, filtered_text, conf])
        
        if detail == 0:
            return [item[1] for item in converted_result]
        
        return converted_result
    
    def _sort_ocr_results_top_to_bottom(self, ocr_output: List[Any]) -> List[Any]:
        """
        Sắp xếp kết quả OCR theo thứ tự từ trên xuống dưới, trái qua phải
        
        Đối với biển số 2 dòng, cần đọc dòng trên trước, sau đó dòng dưới.
        
        Args:
            ocr_output: Kết quả từ PaddleOCR [[bbox, text, conf], ...]
            
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

    def process_plate(self, roi: np.ndarray, apply_warping: bool = True) -> Optional[Dict[str, Any]]:
        """
        Xử lý và nhận diện biển số từ ROI
        Chiến lược: Multi-Hypothesis (Thử nhiều cách tiền xử lý và chọn kết quả tốt nhất)
        """
        # Lấy danh sách các phiên bản ảnh đã tiền xử lý
        variants = preprocess_for_ocr(roi, apply_warping=apply_warping)
        
        candidates = []
        all_intermediates = {}  # Collect all intermediate images
        
        for image, method in variants:
            # Lưu tất cả intermediate images
            all_intermediates[method] = image
            
            # OCR
            ocr_output = self.read_text(image, detail=1)
            
            # Xử lý kết quả
            plate_info, conf = self._process_ocr_result(ocr_output, image, method, all_intermediates)
            
            if plate_info and self.is_valid_plate(plate_info):
                # Ensure all intermediates are included
                plate_info['intermediate_images'] = all_intermediates
                candidates.append(plate_info)
                
                # --- EARLY EXIT (Dừng sớm) ---
                # Nếu độ tin cậy cao (> 0.8), chấp nhận ngay và không thử các phương pháp khác
                if conf > 0.8:
                    print(f"⚡ Early exit with '{method}' ({conf:.2f})")
                    return plate_info
                
        # Chọn kết quả tốt nhất với SMART RANKING
        if not candidates:
            return None
            
        # Smart ranking: Ưu tiên warped methods và binary images
        def calculate_smart_score(candidate):
            """
            CẢI TIẾN: Tính điểm thông minh cho candidate với xác thực chất lượng
            
            Ưu tiên:
            1. Điểm tin cậy (quan trọng nhất)
            2. Độ hoàn chỉnh văn bản (phạt văn bản bị cắt)
            3. Điểm thưởng phương pháp (vừa phải)
            """
            method = candidate['preprocessing_method']
            confidence = candidate['confidence']
            # raw_text = candidate.get('raw_text', '')  # Unused
            clean_text = candidate.get('clean_text', '')
            
            # Điểm cơ bản = confidence (0.0-1.0)
            score = confidence
            
            # KIỂM TRA CHẤT LƯỢNG
            # 1. Kiểm tra độ hoàn chỉnh văn bản
            if len(clean_text) < 6:  # Quá ngắn (phát hiện không đầy đủ)
                score -= 0.2  # Phạt nặng
            elif len(clean_text) < 8:  # Có thể không đầy đủ
                score -= 0.1  # Phạt vừa
                
            # 2. Kiểm tra ngưỡng tin cậy
            if confidence < 0.2:  # Tin cậy rất thấp
                score -= 0.15  # Phạt bổ sung
            elif confidence < 0.3:  # Tin cậy thấp
                score -= 0.05  # Phạt nhỏ
                
            # ĐIỂM THƯỞNG PHƯƠNG PHÁP (GIẢM - bảo thủ hơn)
            # Điểm thưởng vừa cho phương pháp warped (chỉ khi tin cậy tốt VÀ văn bản đầy đủ)
            if ('warped' in method.lower() and confidence > 0.25 and len(clean_text) >= 7):
                score += 0.08  # Giảm từ 0.15 xuống 0.08
            # Phạt cho phương pháp warped với kết quả kém
            elif 'warped' in method.lower() and (confidence < 0.3 or len(clean_text) < 6):
                score -= 0.1  # Phạt cho warping kém
                
            # Điểm thưởng vừa cho phương pháp binary
            if 'otsu' in method.lower():
                score += 0.08  # Giảm từ 0.15 xuống 0.08
                
            # Điểm thưởng nhỏ kết hợp (chỉ khi cả tin cậy và độ dài văn bản tốt)
            if ('warped' in method.lower() and 'otsu' in method.lower() and 
                confidence > 0.25 and len(clean_text) >= 7):
                score += 0.05  # Giảm từ 0.10 xuống 0.05
                
            # Phạt nhỏ cho grayscale thuần (không otsu)
            if 'gray' in method.lower() and 'otsu' not in method.lower() and 'clahe' not in method.lower():
                score -= 0.02  # Giảm phạt
                
            return score
        
        # Sort by smart score (descending)
        candidates.sort(key=calculate_smart_score, reverse=True)
        
        best_result = candidates[0]
        # Ensure all intermediates are included in final result
        best_result['intermediate_images'] = all_intermediates
        
        # Enhanced debug log
        smart_score = calculate_smart_score(best_result)
        print(f"Selected '{best_result['preprocessing_method']}' (conf: {best_result['confidence']:.2f}, smart_score: {smart_score:.2f}) from {len(candidates)} candidates.")
        
        # Show all candidates for debugging
        if len(candidates) > 1:
            print("📊 All candidates:")
            for i, candidate in enumerate(candidates[:3]):  # Show top 3
                c_score = calculate_smart_score(candidate)
                print(f"  {i+1}. {candidate['preprocessing_method']}: conf={candidate['confidence']:.2f}, smart_score={c_score:.2f}")
            
        return best_result
