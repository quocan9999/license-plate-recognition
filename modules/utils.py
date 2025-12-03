"""
Module các hàm hỗ trợ cho nhận diện biển số xe Việt Nam
Bao gồm: validation, formatting, character mapping, và phân loại xe
"""

import re
from typing import List, Dict, Set
from .config import VALID_PROVINCE_START, VALID_PROVINCE_END

# --- MÃ TỈNH THÀNH VIỆT NAM (11-99, trừ 13) ---
VALID_PROVINCE_CODES = set(range(VALID_PROVINCE_START, VALID_PROVINCE_END))  # 11-99
VALID_PROVINCE_CODES.discard(13)  # Loại bỏ mã 13

# --- CHỮ CÁI SERIES HỢP LỆ (20 chữ, không có I, J, O, Q, R, W) ---
VALID_SERIES_LETTERS = set('ABCDEFGHKLMNPSTUVXYZ')

# --- MAPPING CẢI TIẾN ---
# Mapping: Chữ -> Số (dùng cho vị trí phải là SỐ)
# LƯU Ý: Biển số Việt Nam KHÔNG dùng chữ I, nên I luôn được mapping thành 1
dict_char_to_num = {
    'I': '1',  # Biển số VN không dùng I, nên I → 1
    'L': '1', 'T': '1',
    'O': '0',  # Biển số VN không dùng O, nên O → 0
    'Q': '0',  # Biển số VN không dùng Q, nên Q → 0
    'D': '0', 'U': '0',
    'B': '8', 'E': '8',
    'S': '5', 'F': '5',
    'Z': '2', 
    'R': '2',  # Biển số VN không dùng R, nên R → 2
    'G': '6', 'C': '6',
    'A': '4',
    'J': '3',  # Biển số VN không dùng J, nên J → 3
    'W': '2',  # Biển số VN không dùng W, nên W → 2
}

# Mapping: Số -> Chữ (dùng cho vị trí phải là CHỮ)
# Ưu tiên các chữ cái phổ biến trong biển số VN
dict_num_to_char = {
    '0': 'D',  # D là chữ phổ biến
    '1': 'L',  # KHÔNG mapping thành I vì biển số VN không dùng I
    '2': 'Z',
    '3': 'B',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'P'
}


def classify_vehicle(ocr_list: List[str]) -> str:
    """
    Phân loại Xe máy vs Ô tô
    
    CẤU TRÚC BIỂN SỐ VIỆT NAM:
    - Ô tô: 
      + 1 dòng: 30A12345 (mã tỉnh + chữ + 5 số) - 8 ký tự
      + 2 dòng: 37A / 555.55 (dòng 1: mã tỉnh + chữ, dòng 2: số) - dòng 1 có 3 ký tự
      
    - Xe máy thường: 
      + 2 dòng: 29A1 / 123.45 (dòng 1: mã tỉnh + chữ + số, dòng 2: số) - dòng 1 có 4 ký tự
      
    - Xe máy 50cc:
      + 2 dòng: 29AA / 12345 (dòng 1: mã tỉnh + 2 chữ, dòng 2: số) - dòng 1 có 4 ký tự, 2 chữ cuối
    
    LOGIC PHÂN LOẠI (CẢI TIẾN):
    - 1 dòng -> Ô TÔ
    - 2+ dòng:
      + Dòng 1 có 3 ký tự (NN-L) -> Ô TÔ 2 dòng
      + Dòng 1 có 4+ ký tự:
        * Nếu 2 ký tự cuối đều là CHỮ -> XE MÁY 50cc
        * Nếu ký tự cuối là SỐ -> XE MÁY thường
        * Nếu ký tự cuối là CHỮ nhưng ký tự thứ 3 là SỐ -> XE MÁY thường (OCR thiếu số)
    
    Args:
        ocr_list: Danh sách các dòng text từ OCR
        
    Returns:
        Loại xe: "Ô TÔ", "XE MÁY", hoặc "KHÔNG RÕ"
    """
    if len(ocr_list) == 1:
        # 1 dòng -> Ô tô (ví dụ: 30A12345)
        return "Ô TÔ"

    elif len(ocr_list) >= 2:
        line1 = ocr_list[0]
        line1_clean = re.sub(r'[^A-Z0-9]', '', line1.upper())

        if len(line1_clean) == 0: 
            return "KHÔNG RÕ"

        # LOGIC MỚI: Dựa vào độ dài dòng 1
        if len(line1_clean) == 3:
            # Dòng 1 có 3 ký tự (NN-L) -> Ô TÔ 2 dòng
            # Ví dụ: 30A / 123.45, 37R / 555.55
            return "Ô TÔ"
        
        elif len(line1_clean) == 4:
            # Dòng 1 có 4 ký tự -> Kiểm tra 2 ký tự cuối
            char3 = line1_clean[2]  # Ký tự thứ 3 (index 2)
            char4 = line1_clean[3]  # Ký tự thứ 4 (index 3)
            
            # Nếu 2 ký tự cuối đều là CHỮ -> XE MÁY 50cc
            # Ví dụ: 29AA / 12345
            if char3.isalpha() and char4.isalpha():
                return "XE MÁY"
            
            # Nếu ký tự thứ 3 là SỐ -> XE MÁY thường
            # Ví dụ: 29A1 / 123.45
            elif char3.isdigit():
                return "XE MÁY"
            
            # Trường hợp khác (char3 là chữ, char4 là số hoặc ngược lại)
            # Có thể là lỗi OCR, mặc định là XE MÁY
            else:
                return "XE MÁY"
        
        elif len(line1_clean) > 4:
            # Dòng 1 quá dài -> Có thể là OCR nhận diện sai
            # Kiểm tra ký tự cuối
            last_char = line1_clean[-1]
            if last_char.isdigit():
                return "XE MÁY"
            else:
                # Kiểm tra 2 ký tự cuối
                if len(line1_clean) >= 4 and line1_clean[-2].isalpha():
                    return "XE MÁY"  # 50cc
                else:
                    return "XE MÁY"
        
        elif len(line1_clean) < 3:
            # Dòng 1 quá ngắn -> Không rõ
            return "KHÔNG RÕ"

    return "KHÔNG RÕ"


def validate_province_code(code_str: str) -> bool:
    """
    Kiểm tra mã tỉnh có hợp lệ không (11-99)
    
    Args:
        code_str: Chuỗi mã tỉnh (2 ký tự)
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    try:
        code = int(code_str)
        return code in VALID_PROVINCE_CODES
    except:
        return False


def fix_plate_chars(raw_text: str, is_50cc: bool = False) -> str:
    """
    Sửa lỗi ký tự dựa trên pattern biển số Việt Nam
    
    CẤU TRÚC BIỂN SỐ VIỆT NAM:
    - Ô tô: NN-L NNNNN
      + NN: Mã tỉnh (2 số, 11-99)
      + L: Chữ cái series (A-Z)
      + NNNNN: 5 chữ số (00001-99999)
      
    - Xe máy thường: NN-LN NNN.NN
      + NN: Mã tỉnh (2 số, 11-99)
      + L: Chữ cái series (A-Z)
      + N: 1 số (0-9)
      + NNN.NN: 3 số + dấu chấm + 2 số
      
    - Xe máy 50cc: NN-LL NNNNN
      + NN: Mã tỉnh (2 số, 11-99)
      + LL: 2 chữ cái series (AA-ZZ)
      + NNNNN: 5 chữ số
      
    Args:
        raw_text: Text thô từ OCR
        is_50cc: True nếu là xe máy 50cc
        
    Returns:
        Text đã được sửa lỗi
    """
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    chars = list(text)

    if len(chars) < 6: 
        return text

    # === VỊ TRÍ 0-1: MÃ TỈNH (Luôn là SỐ) ===
    for i in [0, 1]:
        if chars[i] in dict_char_to_num:
            chars[i] = dict_char_to_num[chars[i]]

    # Validate mã tỉnh
    province_code = ''.join(chars[0:2])
    if not validate_province_code(province_code):
        # Nếu mã tỉnh không hợp lệ, thử sửa một số trường hợp phổ biến
        # Ví dụ: 3U -> 30, 31 -> 31 (hợp lệ)
        pass

    # === VỊ TRÍ 2: CHỮ CÁI SERIES (Luôn là CHỮ) ===
    if chars[2] in dict_num_to_char:
        chars[2] = dict_num_to_char[chars[2]]

    # === XỬ LÝ PHẦN CÒN LẠI ===
    start_index_for_numbers = 3

    if is_50cc:
        # Xe máy 50cc: NN-LL NNNNN
        # Vị trí 3 cũng là CHỮ
        if len(chars) > 3:
            if chars[3] in dict_num_to_char:
                chars[3] = dict_num_to_char[chars[3]]
        start_index_for_numbers = 4
    
    # Các vị trí còn lại luôn là SỐ
    for i in range(start_index_for_numbers, len(chars)):
        if chars[i] in dict_char_to_num:
            chars[i] = dict_char_to_num[chars[i]]

    return "".join(chars)


def format_plate(text: str, vehicle_type: str) -> str:
    """
    Format biển số theo chuẩn Việt Nam
    
    BIỂN SỐ MỚI (5 số cuối):
    - Ô TÔ: NN-L NNN.NN  (ví dụ: 30A-123.45)
    - XE MÁY: NN-LN NNN.NN  (ví dụ: 29-A1 123.45)
    - XE MÁY 50CC: NN-LL NNN.NN  (ví dụ: 29-AA 123.45)
    
    BIỂN SỐ CŨ (4 số cuối):
    - Ô TÔ: NN-L NNNN  (ví dụ: 30A-4264)
    - XE MÁY: NN-LN NNNN  (ví dụ: 29-A1 4264)
    
    Args:
        text: Text biển số đã được sửa lỗi
        vehicle_type: Loại xe ("XE MÁY" hoặc "Ô TÔ")
        
    Returns:
        Text biển số đã được format
    """
    # Giới hạn độ dài
    if len(text) > 9: 
        text = text[:9]
    
    if vehicle_type == "XE MÁY":
        # Xe máy 50cc
        if len(text) >= 6 and not text[3].isdigit():
            # 29AA12345 -> 29-AA 123.45 (9 ký tự - biển mới)
            if len(text) == 9:
                return f"{text[:2]}-{text[2:4]} {text[4:7]}.{text[7:]}"
            # 29AA1234 -> 29-AA 1234 (8 ký tự - biển cũ)
            elif len(text) == 8:
                return f"{text[:2]}-{text[2:4]} {text[4:]}"
            else:
                return f"{text[:2]}-{text[2:4]} {text[4:]}"
        
        # Xe máy thường
        else:
            # 29A112345 -> 29-A1 123.45 (9 ký tự - biển mới)
            if len(text) == 9:
                return f"{text[:2]}-{text[2:4]} {text[4:7]}.{text[7:]}"
            # 29A11234 -> 29-A1 1234 (8 ký tự - biển cũ)
            elif len(text) == 8:
                return f"{text[:2]}-{text[2:4]} {text[4:]}"
            # 29A1123 -> 29-A1 123 (7 ký tự - biển cũ thiếu)
            elif len(text) == 7:
                return f"{text[:2]}-{text[2:4]} {text[4:]}"
            # Độ dài khác
            elif len(text) >= 6:
                return f"{text[:2]}-{text[2:4]} {text[4:]}"

    else:  # Ô TÔ
        # 30A12345 -> 30A-123.45 (8 ký tự - biển mới)
        if len(text) == 8:
            return f"{text[:2]}{text[2]}-{text[3:6]}.{text[6:]}"
        
        # 30A1234 -> 30A-4264 (7 ký tự - biển cũ)
        elif len(text) == 7:
            return f"{text[:2]}{text[2]}-{text[3:]}"
        
        # 30A123456 -> 30A-123.45 (9 ký tự - cắt bớt)
        elif len(text) == 9:
            return f"{text[:2]}{text[2]}-{text[3:6]}.{text[6:8]}"

    return text
