# PHÂN LOẠI XE MÁY & Ô TÔ

import cv2
import numpy as np
import easyocr
import re

# Khởi tạo EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# ---MAPPING ---
dict_char_to_num = {
    'J': '3', 'I': '1', 'L': '1',
    'O': '0', 'Q': '0', 'D': '0', 'U': '0', 'C': '0',
    'B': '8', 'S': '5', 'Z': '7',
    'G': '9', 'A': '4',
    'T': '1'
}

dict_num_to_char = {
    '0': 'O', '1': 'I', '2': 'Z',
    '4': 'A', '8': 'B', '5': 'S',
    '7': 'Z', '9': 'G', '6': 'G'
}

def classify_vehicle(ocr_list):
    """Phân loại Xe máy vs Ô tô"""
    if len(ocr_list) == 1:
        return "Ô TÔ"

    elif len(ocr_list) >= 2:
        line1 = ocr_list[0]
        line1_clean = re.sub(r'[^A-Z0-9]', '', line1.upper())

        if len(line1_clean) == 0: return "KHÔNG RÕ"

        last_char = line1_clean[-1]
        
        if last_char.isdigit(): # Nếu kí cuối của dòng 1 (kí tự thứ 3) là số thì là xe máy
            return "XE MÁY"
        else:
            if len(line1_clean) >= 4: # Nếu là chữ mà len >= 4 --> xe máy 50ccc
                return "XE MÁY"
            else:
                return "Ô TÔ"

    return "KHÔNG RÕ"


def fix_plate_chars(raw_text, is_50cc=False):
    """
    Sửa lỗi ký tự dựa trên pattern biển số Việt Nam
    Pattern: NN L N... (2 Số - 1 Chữ - Các số còn lại)
    Pattern 50cc: NN L L N... (2 Số - 2 Chữ - Các số còn lại)
    """
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    chars = list(text)

    if len(chars) < 6: return text

    # Hai vị trí đầu (Index 0, 1) luôn là số
    # Ví dụ: 3U -> 30
    for i in [0, 1]:
        if chars[i] in dict_char_to_num:
            chars[i] = dict_char_to_num[chars[i]]

    # Vị trí thứ 2 (Index 2) luôn là chữ
    if chars[2] in dict_num_to_char:
        chars[2] = dict_num_to_char[chars[2]]

    # Xử lý vị trí thứ 3 (Index 3)
    start_index_for_numbers = 3

    # Trường hợp nếu là xe máy 50cc
    if is_50cc:
        # Nếu là xe máy 50cc, vị trí thứ 3 là CHỮ (Ví dụ: 29AA)
        if len(chars) > 3:
            if chars[3] in dict_num_to_char:
                chars[3] = dict_num_to_char[chars[3]]
        start_index_for_numbers = 4
    
    # Các vị trí còn lại luôn là số
    for i in range(start_index_for_numbers, len(chars)):
        if chars[i] in dict_char_to_num:
            chars[i] = dict_char_to_num[chars[i]]

    return "".join(chars)


def format_plate(text, vehicle_type):
    """Format dấu chấm và gạch ngang"""
    if len(text) > 9: text = text[:9]

    if vehicle_type == "XE MÁY":
        if len(text) == 9:
            return f"{text[:2]}-{text[2:4]} {text[4:7]}.{text[7:]}"
        elif len(text) == 8:
            return f"{text[:2]}-{text[2:4]} {text[4:]}"

    else:  # Ô TÔ
        if len(text) >= 8:
            return f"{text[:2]}{text[2]}-{text[3:6]}.{text[6:]}"
        elif len(text) == 7:
            return f"{text[:2]}{text[2]}-{text[3:]}"

    return text


def process_and_predict(image, model_yolo):
    image_np = np.array(image)
    results = model_yolo(image_np)
    detected_plates = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = image_np[y1:y2, x1:x2]

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            ocr_result = reader.readtext(gray, detail=0)

            if len(ocr_result) > 0:
                vehicle_type = classify_vehicle(ocr_result)

                # Check 50cc logic
                is_50cc = False
                if vehicle_type == "XE MÁY":
                    line1 = ocr_result[0]
                    line1_clean = re.sub(r'[^A-Z0-9]', '', line1.upper())
                    # Nếu dòng 1 kết thúc bằng chữ và dài >= 4 thì là 50cc
                    if len(line1_clean) >= 4 and not line1_clean[-1].isdigit():
                        is_50cc = True

                raw_text = "".join(ocr_result)
                clean_text = fix_plate_chars(raw_text, is_50cc=is_50cc)
                final_text = format_plate(clean_text, vehicle_type)

                if len(final_text) > 5:
                    # Biến này để gửi về UI (Vẫn cần [Ô TÔ] để UI biết mà cắt chuỗi)
                    info_for_ui = f"[{vehicle_type}] {final_text}"
                    detected_plates.append(info_for_ui)

                    # Biến này để vẽ lên ảnh
                    text_for_drawing = final_text

                    # Màu sắc
                    color = (0, 255, 0) if vehicle_type == "XE MÁY" else (0, 165, 255)

                    # Vẽ khung
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)

                    # Tính kích thước chữ để vẽ nền đen vừa vặn
                    (w, h), _ = cv2.getTextSize(text_for_drawing, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(image_np, (x1, y1 - 40), (x1 + w, y1), color, -1)

                    # Chỉ vẽ text biển số
                    cv2.putText(image_np, text_for_drawing, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return image_np, detected_plates