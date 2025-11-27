# PHÂN LOẠI XE MÁY & Ô TÔ

import cv2
import numpy as np
import easyocr
import re

# Khởi tạo EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# --- 1. CẬP NHẬT TỪ ĐIỂN MAPPING ---
# Thêm 'U': '0' và các ký tự dễ nhầm khác
dict_char_to_num = {
    'J': '3', 'I': '1', 'L': '1',
    'O': '0', 'Q': '0', 'D': '0', 'U': '0', 'C': '0',  # Thêm U, C -> 0
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

        if last_char.isdigit():
            return "XE MÁY"
        else:
            return "Ô TÔ"

    return "KHÔNG RÕ"


def fix_plate_chars(raw_text):
    """
    Sửa lỗi ký tự dựa trên pattern biển số Việt Nam
    Pattern: NN L N... (2 Số - 1 Chữ - Các số còn lại)
    """
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    chars = list(text)

    if len(chars) < 6: return text

    # --- SỬA LỖI MÃ TỈNH (QUAN TRỌNG) ---
    # 1. Hai vị trí đầu (Index 0, 1): LUÔN LÀ SỐ
    # Ví dụ: 3U... -> 30..., 5Z... -> 57...
    for i in [0, 1]:
        if chars[i] in dict_char_to_num:
            chars[i] = dict_char_to_num[chars[i]]
        # Nếu ký tự đó là chữ nhưng không nằm trong dict map (ví dụ 'K'),
        # ta vẫn có thể cân nhắc thay thế nếu cần, nhưng tạm thời map các lỗi phổ biến là đủ.

    # 2. Vị trí thứ 2 (Index 2): LUÔN LÀ CHỮ (Series)
    if chars[2] in dict_num_to_char:
        chars[2] = dict_num_to_char[chars[2]]

    # 3. Từ vị trí thứ 3 (Index 3) trở đi: LUÔN LÀ SỐ
    for i in range(3, len(chars)):
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

                raw_text = "".join(ocr_result)
                clean_text = fix_plate_chars(raw_text)
                final_text = format_plate(clean_text, vehicle_type)

                if len(final_text) > 5:
                    # --- 2. SỬA PHẦN HIỂN THỊ ---

                    # Biến này để gửi về UI (Vẫn cần [Ô TÔ] để UI biết mà cắt chuỗi)
                    info_for_ui = f"[{vehicle_type}] {final_text}"
                    detected_plates.append(info_for_ui)

                    # Biến này để vẽ lên ảnh (Chỉ vẽ biển số)
                    text_for_drawing = final_text

                    # Màu sắc
                    color = (0, 255, 0) if vehicle_type == "XE MÁY" else (0, 165, 255)

                    # Vẽ khung
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)

                    # Tính kích thước chữ để vẽ nền đen vừa vặn
                    (w, h), _ = cv2.getTextSize(text_for_drawing, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(image_np, (x1, y1 - 40), (x1 + w, y1), color, -1)

                    # Chỉ vẽ text biển số (30F-789.07)
                    cv2.putText(image_np, text_for_drawing, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return image_np, detected_plates