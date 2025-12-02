"""
Module các hàm hỗ trợ cho nhận diện biển số xe Việt Nam
Bao gồm: validation, formatting, character mapping, và phân loại xe
"""

import re
from .config import VALID_PROVINCE_START, VALID_PROVINCE_END

# --- MÃ TỈNH THÀNH VIỆT NAM (11-99) ---
VALID_PROVINCE_CODES = set(range(VALID_PROVINCE_START, VALID_PROVINCE_END))  # 11-99

# --- MAPPING CẢI TIẾN ---
# Mapping: Chữ -> Số (dùng cho vị trí phải là SỐ)
dict_char_to_num = {
    'I': '1', 'L': '1', 'T': '1',
    'O': '0', 'Q': '0', 'D': '0', 'U': '0',
    'B': '8', 'E': '8', 'R': '8',
    'S': '5', 'F': '5',
    'Z': '2', 
    'G': '6', 'C': '6',
    'A': '4',
    'J': '3',
    'Y': '7',
}

# Mapping: Số -> Chữ (dùng cho vị trí phải là CHỮ)
dict_num_to_char = {
    '0': 'D', '1': 'I', '2': 'Z',
    '3': 'B', '4': 'A', '5': 'S',
    '6': 'G', '8': 'B', '7': 'T',
    '9': 'P' # Optional but helpful
}


def classify_vehicle(ocr_list):
    """
    Phân loại Xe máy vs Ô tô
    
    CẤU TRÚC BIỂN SỐ VIỆT NAM:
    - Ô tô: 
      + 1 dòng: 30A12345 (mã tỉnh + chữ + 5 số)
      + 2 dòng: 37A / 555.55 (dòng 1: mã tỉnh + chữ, dòng 2: số)
      
    - Xe máy thường: 
      + 2 dòng: 29A1 / 123.45 (dòng 1: mã tỉnh + chữ + số, dòng 2: số)
      
    - Xe máy 50cc:
      + 2 dòng: 29AA / 12345 (dòng 1: mã tỉnh + 2 chữ, dòng 2: số)
    
    LOGIC PHÂN LOẠI:
    - 1 dòng -> Ô TÔ
    - 2+ dòng:
      + Dòng 1 kết thúc bằng SỐ -> XE MÁY thường
      + Dòng 1 kết thúc bằng CHỮ:
        * Nếu có 2 chữ liên tiếp (4+ ký tự) -> XE MÁY 50cc
        * Nếu chỉ 1 chữ (3 ký tự: NN-L) -> Ô TÔ
    
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

        last_char = line1_clean[-1]
        
        # Nếu ký tự cuối của dòng 1 là số -> xe máy thông thường
        # Ví dụ: 29A1 / 123.45
        if last_char.isdigit():
            return "XE MÁY"
        else:
            # Ký tự cuối là CHỮ
            # Kiểm tra xem có 2 chữ liên tiếp không (xe máy 50cc)
            # Ví dụ: 29AA -> len=4, 2 ký tự cuối đều là chữ
            if len(line1_clean) >= 4 and not line1_clean[-2].isdigit():
                # Xe máy 50cc: 29AA / 12345
                return "XE MÁY"
            else:
                # Ô tô 2 dòng: 37A / 555.55
                # Dòng 1 chỉ có 3 ký tự (NN + L)
                # Tuy nhiên, nếu L không phải là A, B, C, D (các series ô tô phổ biến),
                # thì khả năng cao là Xe máy bị nhận diện thiếu số (ví dụ: 59-V3 -> 59-V)
                series_char = line1_clean[-1]
                if series_char in ['A', 'B', 'C', 'D']:
                    return "Ô TÔ"
                else:
                    return "XE MÁY"

    return "KHÔNG RÕ"


def validate_province_code(code_str):
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


def fix_plate_chars(raw_text, is_50cc=False):
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


def format_plate(text, vehicle_type):
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
        text: Text biển số đã được sửa lỗi (chỉ gồm A-Z, 0-9)
        vehicle_type: Loại xe ("XE MÁY" hoặc "Ô TÔ")
        
    Returns:
        Text biển số đã được format
    """
    # Giới hạn độ dài tối đa
    if len(text) > 9: 
        text = text[:9]
    
    if vehicle_type == "XE MÁY":
        # Xe máy 50cc (NN-LL...)
        # Ví dụ: 29AA12345
        if len(text) >= 4 and not text[3].isdigit():
            # Biển 5 số: 29AA12345 -> 29-AA 123.45 (9 ký tự)
            if len(text) == 9:
                return f"{text[:2]}-{text[2:4]} {text[4:7]}.{text[7:]}"
            # Biển 4 số: 29AA1234 -> 29-AA 1234 (8 ký tự)
            elif len(text) == 8:
                return f"{text[:2]}-{text[2:4]} {text[4:]}"
            else:
                # Fallback cho độ dài lạ
                return f"{text[:2]}-{text[2:4]} {text[4:]}"
        
        # Xe máy thường (NN-LN...)
        # Ví dụ: 29A112345
        else:
            # Biển 5 số: 29A112345 -> 29-A1 123.45 (9 ký tự)
            if len(text) == 9:
                return f"{text[:2]}-{text[2:4]} {text[4:7]}.{text[7:]}"
            # Biển 4 số: 29A11234 -> 29-A1 1234 (8 ký tự)
            elif len(text) == 8:
                return f"{text[:2]}-{text[2:4]} {text[4:]}"
            # Trường hợp thiếu số nhưng vẫn format tạm
            elif len(text) >= 6:
                return f"{text[:2]}-{text[2:4]} {text[4:]}"

    else:  # Ô TÔ
        # Biển 5 số: 30A12345 -> 30A-123.45 (8 ký tự)
        if len(text) == 8:
            return f"{text[:2]}{text[2]}-{text[3:6]}.{text[6:]}"
        
        # Biển 4 số: 30A1234 -> 30A-4264 (7 ký tự)
        elif len(text) == 7:
            return f"{text[:2]}{text[2]}-{text[3:]}"
        
        # Trường hợp thừa ký tự (ví dụ đọc nhầm) -> Cắt về 5 số
        elif len(text) >= 9:
            # Lấy 8 ký tự đầu tiên hợp lệ nhất
            return f"{text[:2]}{text[2]}-{text[3:6]}.{text[6:8]}"

    return text
