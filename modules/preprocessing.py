"""
Module tiền xử lý ảnh cho nhận diện biển số xe
Preprocessing đơn giản để giữ độ chính xác cao
"""

import cv2
import numpy as np


def preprocess_for_ocr(roi):
    """
    Tiền xử lý ảnh ROI (Region of Interest) của biển số
    
    Args:
        roi: Ảnh vùng biển số (numpy array)
        
    Returns:
        Ảnh đã được tiền xử lý (grayscale)
    """
    # Chuyển sang grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi.copy()
    
    return gray


# def apply_adaptive_threshold(gray_image):
#     """
#     Áp dụng adaptive thresholding cho ảnh grayscale
    
#     Args:
#         gray_image: Ảnh grayscale
        
#     Returns:
#         Ảnh đã được threshold
#     """
#     return cv2.adaptiveThreshold(
#         gray_image, 
#         255, 
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY, 
#         11, 
#         2
#     )


# def denoise_image(image):
#     """
#     Khử nhiễu cho ảnh
    
#     Args:
#         image: Ảnh đầu vào
        
#     Returns:
#         Ảnh đã được khử nhiễu
#     """
#     return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


# def enhance_contrast(image):
#     """
#     Tăng cường độ tương phản của ảnh
    
#     Args:
#         image: Ảnh đầu vào (grayscale)
        
#     Returns:
#         Ảnh đã được tăng cường độ tương phản
#     """
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     return clahe.apply(image)
