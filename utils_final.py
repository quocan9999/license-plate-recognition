# KHÔNG PHÂN LOẠI XE MÁY & Ô TÔ

import cv2
import numpy as np
import easyocr
import re

# Khởi tạo EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# --- TỪ ĐIỂN QUY ĐỔI NHẦM LẪN (Mapping) ---
# Quy đổi khi vị trí đó CẦN LÀ SỐ nhưng lại nhận diện ra chữ
dict_char_to_num = {
    'J': '3', 'I': '1', 'L': '1',
    'O': '0', 'Q': '0', 'D': '0',
    'B': '8', 'S': '5', 'Z': '7',
    'G': '9', 'A': '4'
}

# Quy đổi khi vị trí đó CẦN LÀ CHỮ nhưng lại nhận diện ra số
dict_num_to_char = {
    '0': 'O', '1': 'I', '2': 'Z',
    '4': 'A', '8': 'B', '5': 'S',
    '7': 'Z', '9': 'G', '6': 'G'  # Đôi khi 6 nhìn nhầm thành G
}


def fix_plate_chars(raw_text):
    """
    Hàm logic cốt lõi để sửa lỗi nhận diện sai (G->9, Z->7...)
    Dựa trên pattern: NN L N... (2 Số - 1 Chữ - Các số còn lại)
    """
    # 1. Xóa hết ký tự đặc biệt, chỉ để lại Chữ và Số liền nhau
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    # Nếu ngắn quá hoặc dài quá thì trả về nguyên gốc để tránh lỗi index
    if len(text) < 7:
        return raw_text  # Không đủ dữ kiện để sửa

    # Nếu dài hơn 9 ký tự (ví dụ đọc thừa rác), cắt bớt
    if len(text) > 9:
        text = text[:9]

    # Chuyển string thành list ký tự để dễ thay thế
    chars = list(text)

    # --- LOGIC SỬA LỖI THEO VỊ TRÍ ---

    # Vị trí 0, 1: Luôn là SỐ (Mã tỉnh, vd: 59, 30)
    for i in [0, 1]:
        if chars[i] in dict_char_to_num:
            chars[i] = dict_char_to_num[chars[i]]

    # Vị trí 2: Luôn là CHỮ (Series, vd: A, B, P, C)
    if chars[2] in dict_num_to_char:
        chars[2] = dict_num_to_char[chars[2]]

    # Vị trí 3 trở đi: Luôn là SỐ (Số biển số)
    # Lưu ý: Xe máy (59-P1) thì ký tự thứ 3 (index 3) là số 1 -> Vẫn là SỐ.
    # Xe ô tô (30A-12345) thì ký tự thứ 3 (index 3) là số 1 -> Vẫn là SỐ.
    # -> Kết luận: Từ index 3 trở về sau CHẮC CHẮN LÀ SỐ.
    for i in range(3, len(chars)):
        if chars[i] in dict_char_to_num:
            chars[i] = dict_char_to_num[chars[i]]

    return "".join(chars)


def format_plate(text):
    """
    Hàm định dạng lại dấu chấm và gạch ngang cho đẹp
    Input: chuỗi liền (vd: 59P112345)
    Output: chuỗi format (vd: 59-P1 123.45)
    """
    # Xử lý cắt chuỗi thừa (như bạn nói đôi khi ra 6 số cuối)
    # Pattern VN tối đa chỉ có 5 số cuối.
    # Tổng độ dài tối đa của biển VN hiện tại là 9 ký tự (59-P1 123.45 -> 59P112345)
    if len(text) > 9:
        text = text[:9]  # Cắt bỏ phần thừa

    # --- FORMAT ---
    # Case 1: Biển xe máy mới / Biển 5 số (Tổng 9 ký tự: NN L N NNNNN)
    # VD: 59P112345 -> 59-P1 123.45
    if len(text) == 9:
        return f"{text[:2]}-{text[2:4]} {text[4:7]}.{text[7:]}"

    # Case 2: Biển ô tô mới (Tổng 8 ký tự: NN L NNNNN)
    # VD: 30K12345 -> 30K-123.45
    # Hoặc Biển xe máy cũ (Tổng 8 ký tự: NN L N NNNN) -> 59P1 1234 (ít gặp hơn, ưu tiên format ô tô hoặc tùy chỉnh)
    elif len(text) == 8:
        # Check logic: Nếu ký tự thứ 3 (index 3) là số, khả năng cao là ô tô 5 số hoặc xe máy 4 số
        # Để an toàn và đẹp, ta format kiểu chung: 30K-123.45
        return f"{text[:2]}{text[2]}-{text[3:6]}.{text[6:]}"

    # Case 3: Biển ô tô cũ (Tổng 7 ký tự: NN L NNNN)
    # VD: 30A1234 -> 30A-1234
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

            # Tiền xử lý ảnh (Quan trọng để OCR chuẩn hơn)
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

            # OCR đọc raw text
            ocr_result = reader.readtext(gray, detail=0)

            # Gộp text lại thành 1 dòng liền mạch
            raw_text = "".join(ocr_result)

            # --- BƯỚC HẬU XỬ LÝ (POST PROCESS) ---
            # 1. Sửa lỗi ký tự (G->9, Z->7...)
            clean_chars = fix_plate_chars(raw_text)

            # 2. Định dạng dấu chấm, gạch ngang
            final_text = format_plate(clean_chars)

            if len(final_text) > 4:
                detected_plates.append(final_text)
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Vẽ nền đen cho chữ dễ đọc
                cv2.rectangle(image_np, (x1, y1 - 40), (x2, y1), (0, 0, 0), -1)
                cv2.putText(image_np, final_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return image_np, detected_plates