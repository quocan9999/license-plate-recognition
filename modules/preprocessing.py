"""
Module tiền xử lý ảnh cho nhận diện biển số xe
Bao gồm: Grayscale conversion, Warping (nắn thẳng), và các kỹ thuật nâng cao
"""

import cv2
import numpy as np


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


def detect_and_warp_plate(roi):
    """
    Tự động phát hiện góc biển số và nắn thẳng
    
    Args:
        roi: Ảnh vùng biển số (numpy array)
        
    Returns:
        Ảnh đã được nắn thẳng, hoặc ảnh gốc nếu không phát hiện được góc
    """
    # Chuyển sang grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi.copy()
    
    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Phát hiện cạnh
    edged = cv2.Canny(blurred, 50, 150)
    
    # Tìm contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Nếu không tìm thấy contour, trả về ảnh gốc
    if len(contours) == 0:
        return gray
    
    # Sắp xếp contours theo diện tích (lớn nhất trước)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    plate_contour = None
    
    # Tìm contour có 4 góc (hình chữ nhật)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            plate_contour = approx
            break
    
    # Nếu tìm thấy contour 4 góc, thực hiện warping
    if plate_contour is not None:
        warped = four_point_transform(roi, plate_contour.reshape(4, 2))
        return warped
    
    # Nếu không tìm thấy, trả về ảnh grayscale gốc
    return gray


def preprocess_for_ocr(roi, apply_warping=True):
    """
    Tiền xử lý ảnh ROI (Region of Interest) của biển số
    
    Args:
        roi: Ảnh vùng biển số (numpy array)
        apply_warping: Có áp dụng warping hay không
        
    Returns:
        Ảnh đã được tiền xử lý
    """
    if apply_warping:
        # Áp dụng warping để nắn thẳng biển số
        processed = detect_and_warp_plate(roi)
    else:
        # Chỉ chuyển sang grayscale
        if len(roi.shape) == 3:
            processed = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            processed = roi.copy()
    
    return processed


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
