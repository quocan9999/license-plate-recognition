"""
Module tiền xử lý ảnh cho nhận diện biển số xe
Bao gồm: Grayscale conversion, Warping (nắn thẳng), và các kỹ thuật nâng cao
"""

import cv2
import numpy as np
from .config import (
    CLAHE_CLIP_LIMIT, 
    CLAHE_TILE_GRID_SIZE, 
    UPSCALE_SCALE, 
    WARP_PADDING,
    ADAPTIVE_THRESH_BLOCK_SIZE,
    ADAPTIVE_THRESH_C
)


def order_points(pts):
    """
    Sắp xếp các điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: Array 4 điểm
        
    Returns:
        Array 4 điểm đã được sắp xếp
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left có tổng nhỏ nhất, bottom-right có tổng lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right có diff nhỏ nhất, bottom-left có diff lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image, pts):
    """
    Perspective transform để nắn thẳng ảnh dựa trên 4 điểm
    
    Args:
        image: Ảnh đầu vào
        pts: 4 điểm góc của vùng cần transform
        
    Returns:
        Ảnh đã được nắn thẳng
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Tính chiều rộng của ảnh mới
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Tính chiều cao của ảnh mới
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Tạo điểm đích cho perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Tính ma trận perspective transform và áp dụng
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def apply_clahe(image):
    """
    Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Giúp cân bằng sáng cục bộ, khắc phục bóng che hoặc lóa.
    """
    if len(image.shape) == 3:
        # Chuyển sang LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Áp dụng CLAHE cho kênh L
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        cl = clahe.apply(l)
        
        # Merge lại
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final
    else:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        return clahe.apply(image)


def apply_super_resolution(image, scale=UPSCALE_SCALE):
    """
    Phóng to ảnh (Upscaling) dùng Bicubic Interpolation.
    Giúp EasyOCR nhận diện tốt hơn với text nhỏ.
    """
    # Sử dụng Bicubic cho chất lượng tốt hơn Linear
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def detect_and_warp_plate(roi):
    """
    Tự động phát hiện góc biển số và nắn thẳng
    Cải tiến: Thêm Padding & Morphology để bắt contour tốt hơn
    
    Args:
        roi: Ảnh vùng biển số (numpy array)
        
    Returns:
        Ảnh đã được nắn thẳng, hoặc ảnh gốc nếu không phát hiện được góc
    """
    # 1. Thêm Padding để tránh contour bị dính viền
    h, w = roi.shape[:2]
    padding = WARP_PADDING
    padded_roi = cv2.copyMakeBorder(roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    # 2. Chuyển sang grayscale
    if len(padded_roi.shape) == 3:
        gray = cv2.cvtColor(padded_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = padded_roi.copy()
    
    # 3. Tiền xử lý để tìm contour (Làm "đặc" biển số)
    # Dùng Adaptive Threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C)
    
    # Morphology Close để nối liền các ký tự thành 1 khối đặc
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Tìm contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_contour = None
    
    if len(contours) > 0:
        # Sắp xếp contours theo diện tích (lớn nhất trước)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                plate_contour = approx
                break
    
    # 5. Warping
    if plate_contour is not None:
        # Trừ đi padding để lấy tọa độ gốc trên ảnh roi ban đầu
        pts = plate_contour.reshape(4, 2)
        pts = pts - padding
        
        # Đảm bảo tọa độ không âm
        pts = np.maximum(pts, 0)
        
        # Thực hiện warp trên ảnh gốc (roi)
        warped = four_point_transform(roi, pts)
        return warped, "warped"
    
    # Nếu không tìm thấy, trả về ảnh gốc
    return roi, "original"


def preprocess_for_ocr(roi, apply_warping=True):
    """
    Tiền xử lý ảnh ROI (Region of Interest) của biển số
    Pipeline: Warping -> Grayscale -> CLAHE -> Super-Resolution
    
    Args:
        roi: Ảnh vùng biển số (numpy array)
        apply_warping: Có áp dụng warping hay không
        
    Returns:
        Ảnh đã được tiền xử lý
    """
    method_str = []
    processed = roi
    
    intermediates = {}
    
    # 1. Warping
    if apply_warping:
        processed, method = detect_and_warp_plate(roi)
        if method == "warped":
            method_str.append("warped")
            intermediates["warped"] = processed.copy()
    
    # 2. Grayscale (EasyOCR hoạt động tốt trên Gray/Binary)
    if len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        method_str.append("gray")
        intermediates["gray"] = processed.copy()
    
    # 3. CLAHE (Cân bằng sáng)
    processed = apply_clahe(processed)
    method_str.append("clahe")
    intermediates["clahe"] = processed.copy()
    
    # 4. Super-Resolution (Phóng to)
    processed = apply_super_resolution(processed, scale=UPSCALE_SCALE)
    method_str.append("upscale")
    intermediates["upscale"] = processed.copy()
    
    return processed, "+".join(method_str), intermediates


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
