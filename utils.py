# PHÂN LOẠI XE MÁY & Ô TÔ - Phiên bản cải tiến

import cv2
import numpy as np
import easyocr
import re

# Khởi tạo EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# --- MÃ TỈNH THÀNH VIỆT NAM (11-99) ---
VALID_PROVINCE_CODES = set(range(11, 100))  # 11-99

# --- MAPPING CẢI TIẾN ---
# Mapping: Chữ -> Số (dùng cho vị trí phải là SỐ)
dict_char_to_num = {
    'I': '1', 'L': '1', 'T': '1',
    'O': '0', 'Q': '0', 'D': '0', 'U': '0',
    'B': '8', 'E': '8',
    'S': '5',
    'Z': '2', 'R': '2',
    'G': '6', 'C': '6',
    'A': '4',
    'J': '3',
}

# Mapping: Số -> Chữ (dùng cho vị trí phải là CHỮ)
dict_num_to_char = {
    '0': 'D', '1': 'I', '2': 'Z',
    '3': 'B', '4': 'A', '5': 'S',
    '6': 'G', '8': 'B',
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
                return "Ô TÔ"

    return "KHÔNG RÕ"


def validate_province_code(code_str):
    """
    Kiểm tra mã tỉnh có hợp lệ không (11-99)
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


def process_and_predict(image, model_yolo):
    """
    Xử lý ảnh và nhận diện biển số xe
    """
    image_np = np.array(image)
    results = model_yolo(image_np)
    detected_plates = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = image_np[y1:y2, x1:x2]

            # OCR với preprocessing đơn giản
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            ocr_result = reader.readtext(gray, detail=0)

            if len(ocr_result) > 0:
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
                final_text = format_plate(clean_text, vehicle_type)

                # Chỉ hiển thị nếu độ dài hợp lý
                if len(final_text) > 5:
                    # Chuẩn bị text cho UI
                    info_for_ui = f"[{vehicle_type}] {final_text}"
                    detected_plates.append(info_for_ui)

                    # Text để vẽ lên ảnh (không có loại xe)
                    text_for_drawing = final_text

                    # Màu sắc: Xanh lá (xe máy), Cam (ô tô)
                    color = (0, 255, 0) if vehicle_type == "XE MÁY" else (0, 165, 255)

                    # Vẽ khung bao quanh biển số
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)

                    # Tính kích thước chữ để vẽ nền
                    (w, h), _ = cv2.getTextSize(text_for_drawing, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(image_np, (x1, y1 - 40), (x1 + w, y1), color, -1)

                    # Vẽ text biển số
                    cv2.putText(image_np, text_for_drawing, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return image_np, detected_plates