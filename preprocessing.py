"""
Module tiền xử lý ảnh cho nhận diện biển số xe
Preprocessing đơn giản để giữ độ chính xác cao
"""

import cv2


def preprocess_for_ocr(roi):
    # Chuyển sang grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi.copy()
    
    return gray
