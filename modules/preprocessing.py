"""
Module tiền xử lý ảnh cho nhận diện biển số xe
Bao gồm: Grayscale conversion, Warping (nắn thẳng), và các kỹ thuật nâng cao
"""

import cv2
import numpy as np
from typing import List, Tuple
from .config import (
    CLAHE_CLIP_LIMIT, 
    CLAHE_TILE_GRID_SIZE, 
    UPSCALE_SCALE, 
    WARP_PADDING
)


def order_points(pts: np.ndarray) -> np.ndarray:
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


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Perspective transform để nắn thẳng ảnh dựa trên 4 điểm
    IMPROVED: Thêm error handling và validation
    
    Args:
        image: Ảnh đầu vào
        pts: 4 điểm góc của vùng cần transform
        
    Returns:
        Ảnh đã được nắn thẳng
    """
    try:
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
        
        # Minimum size validation
        if maxWidth < 10 or maxHeight < 10:
            print(f"⚠️ Warped size too small: {maxWidth}x{maxHeight}")
            return image
        
        # Maximum size validation (prevent memory issues)
        if maxWidth > 2000 or maxHeight > 2000:
            print(f"⚠️ Warped size too large: {maxWidth}x{maxHeight}")
            return image
        
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
    except Exception as e:
        print(f"⚠️ Four point transform error: {e}")
        return image


def apply_clahe(image: np.ndarray) -> np.ndarray:
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


def apply_threshold(image: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """
    Áp dụng phân ngưỡng (Thresholding) để chuyển sang ảnh nhị phân.
    Giúp tách chữ khỏi nền nhiễu.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    if method == 'otsu':
        # Otsu's binarization
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    elif method == 'adaptive':
        # Adaptive thresholding
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return gray

def apply_super_resolution(image: np.ndarray, scale: int = UPSCALE_SCALE) -> np.ndarray:
    """
    Phóng to ảnh (Upscaling) dùng Bicubic Interpolation.
    ...
    """
    h, w = image.shape[:2]
    if h < 64: # Chỉ upscale nếu ảnh nhỏ
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image


def detect_and_warp_plate(roi: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    ENHANCED: Tự động phát hiện góc biển số và nắn thẳng với thuật toán mạnh hơn
    
    CÁC CẢI TIẾN MỚI:
    1. Edge detection + Hough lines (HIỆU QUẢ NHẤT cho biển số nghiêng)
    2. Corner detection với Harris & Shi-Tomasi
    3. Improved contour detection với tham số tích cực
    4. Multiple fallback strategies
    5. Rotation analysis & validation
    
    Args:
        roi: Ảnh vùng biển số (numpy array)
        
    Returns:
        Ảnh đã được nắn thẳng, hoặc ảnh gốc nếu không phát hiện được góc
    """
    # Phương pháp 1: Edge-based warping (HIỆU QUẢ NHẤT)
    warped, method = edge_based_warping(roi)
    if method == "edge_warped":
        return warped, method
    
    # Phương pháp 2: Corner-based warping
    warped, method = corner_based_warping(roi)
    if method == "corner_warped":
        return warped, method
    
    # Phương pháp 3: Improved contour-based (dự phòng)
    warped, method = improved_contour_warping(roi)
    if method == "contour_warped":
        return warped, method
    
    return roi, "original"


def edge_based_warping(roi: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Warping dựa trên edge detection và Hough lines - MOST EFFECTIVE for tilted plates
    """
    
    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    h, w = gray.shape[:2]
    
    # Edge detection với nhiều phương pháp
    edges_methods = [
        ("canny", lambda img: cv2.Canny(img, 50, 150, apertureSize=3)),
        ("canny_strong", lambda img: cv2.Canny(img, 100, 200, apertureSize=3)),
        ("sobel", lambda img: cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)))
    ]
    
    for _, edge_func in edges_methods:
        edges = edge_func(gray)
        
        # Hough Line Detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=w//4, maxLineGap=h//4)
        
        if lines is not None and len(lines) >= 4:
            # Phân loại lines thành horizontal và vertical
            h_lines = []  # Horizontal lines
            v_lines = []  # Vertical lines
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Tính góc
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 30 or abs(angle) > 150:  # Horizontal-ish
                    h_lines.append(line[0])
                elif 60 < abs(angle) < 120:  # Vertical-ish
                    v_lines.append(line[0])
            
            if len(h_lines) >= 2 and len(v_lines) >= 2:
                # Tìm 4 góc từ intersection của lines
                corners = find_rectangle_corners_from_lines(h_lines, v_lines, gray.shape)
                
                if corners is not None:
                    try:
                        warped = four_point_transform_enhanced(roi, corners)
                        return warped, "edge_warped"
                    except:
                        pass
    
    return roi, "original"


def corner_based_warping(roi: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Warping dựa trên corner detection
    """
    
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    # Corner detection với Shi-Tomasi (đáng tin cậy nhất)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    if corners is not None and len(corners) >= 4:
        # Tìm 4 góc xa nhất từ center
        center = np.array([gray.shape[1]//2, gray.shape[0]//2])
        corners = corners.reshape(-1, 2)
        
        # Sort theo khoảng cách từ center
        distances = np.sqrt(np.sum((corners - center)**2, axis=1))
        farthest_indices = np.argsort(distances)[-4:]
        four_corners = corners[farthest_indices]
        
        try:
            warped = four_point_transform_enhanced(roi, four_corners.astype(np.float32))
            return warped, "corner_warped"
        except:
            pass
    
    return roi, "original"


def improved_contour_warping(roi: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Improved contour-based warping với aggressive parameters
    """
    
    h, w = roi.shape[:2]
    padding = WARP_PADDING
    padded_roi = cv2.copyMakeBorder(roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    if len(padded_roi.shape) == 3:
        gray = cv2.cvtColor(padded_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = padded_roi.copy()
    
    # Tiền xử lý tích cực hơn
    methods = [
        {
            'name': 'aggressive_adaptive',
            'thresh': lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5),
            'morph_kernel': (7, 7),
            'morph_iterations': 3,
            'approx_factors': [0.005, 0.01, 0.015, 0.02, 0.025]  # Nhiều hệ số hơn
        },
        {
            'name': 'strong_otsu',
            'thresh': lambda img: cv2.threshold(cv2.GaussianBlur(img, (5,5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            'morph_kernel': (9, 9),
            'morph_iterations': 4,
            'approx_factors': [0.01, 0.02]
        }
    ]
    
    best_contour = None
    best_score = 0
    
    for method in methods:
        thresh = method['thresh'](gray)
        
        # Hình thái học mạnh hơn
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, method['morph_kernel'])
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=method['morph_iterations'])
        
        # Các phép hình thái học bổ sung
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
            
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        roi_area = h * w
        
        for contour in contours:
            cnt_area = cv2.contourArea(contour)
            if cnt_area < 0.03 * roi_area:  # Ngưỡng thậm chí thấp hơn
                continue
            
            for approx_factor in method['approx_factors']:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, approx_factor * peri, True)
                
                # Cho phép số đỉnh linh hoạt hơn
                if 4 <= len(approx) <= 12:
                    (bx, by, bw, bh) = cv2.boundingRect(approx)
                    aspect_ratio = bw / float(bh)
                    
                    if 0.3 <= aspect_ratio <= 8.0:  # Tỷ lệ khung hình linh hoạt hơn
                        # Tính điểm tốt hơn
                        vertex_score = max(0, (12 - len(approx)) / 8.0)
                        area_score = min(1.0, cnt_area / (0.2 * roi_area))
                        ar_score = min(1.0, min(aspect_ratio / 1.5, 3.0 / aspect_ratio))
                        
                        total_score = vertex_score * 0.3 + area_score * 0.5 + ar_score * 0.2
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_contour = approx.copy()
    
    if best_contour is not None and best_score > 0.2:  # Ngưỡng điểm thấp hơn
        if len(best_contour) > 4:
            rect = cv2.minAreaRect(best_contour)
            box = cv2.boxPoints(rect)
            best_contour = np.array(box, dtype=np.float32).reshape(-1, 1, 2)
        
        pts = best_contour.reshape(-1, 2) - padding
        pts = np.maximum(pts, 0)
        
        if len(pts) == 4:
            try:
                warped = four_point_transform_enhanced(roi, pts.astype(np.float32))
                return warped, "contour_warped"
            except:
                pass
    
    return roi, "original"


def find_rectangle_corners_from_lines(h_lines, v_lines, shape):
    """
    Tìm 4 góc hình chữ nhật từ horizontal và vertical lines
    """
    
    try:
        h, w = shape[:2]
        
        # Sort lines
        h_lines = sorted(h_lines, key=lambda line: (line[1] + line[3]) // 2)  # Sort by y
        v_lines = sorted(v_lines, key=lambda line: (line[0] + line[2]) // 2)  # Sort by x
        
        # Take top, bottom horizontal lines và left, right vertical lines
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            top_line = h_lines[0]
            bottom_line = h_lines[-1]
            left_line = v_lines[0]
            right_line = v_lines[-1]
            
            # Find intersections
            corners = []
            
            # Top-left
            tl = line_intersection(top_line, left_line)
            if tl is not None:
                corners.append(tl)
            
            # Top-right
            tr = line_intersection(top_line, right_line)
            if tr is not None:
                corners.append(tr)
            
            # Bottom-right
            br = line_intersection(bottom_line, right_line)
            if br is not None:
                corners.append(br)
            
            # Bottom-left
            bl = line_intersection(bottom_line, left_line)
            if bl is not None:
                corners.append(bl)
            
            if len(corners) == 4:
                return np.array(corners, dtype=np.float32)
        
        return None
    except:
        return None


def line_intersection(line1, line2):
    """
    Tìm giao điểm của 2 đường thẳng
    """
    
    try:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6:
            return None
        
        px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
        py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
        
        return [px, py]
    except:
        return None


def four_point_transform_enhanced(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Enhanced four point transform với validation tốt hơn
    """
    
    # Order points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate dimensions
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Validation
    if maxWidth < 20 or maxHeight < 20:
        raise ValueError("Warped size too small")
    
    if maxWidth > 1500 or maxHeight > 1500:
        raise ValueError("Warped size too large")
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def preprocess_for_ocr(roi: np.ndarray, apply_warping: bool = True) -> List[Tuple[np.ndarray, str]]:
    """
    Tiền xử lý ảnh ROI (Region of Interest) của biển số
    Trả về nhiều phiên bản xử lý khác nhau để OCR thử nghiệm.
    
    Args:
        roi: Ảnh vùng biển số (numpy array)
        apply_warping: Có áp dụng warping hay không
        
    Returns:
        List các tuple (image, method_name)
    """
    variants = []
    
    # 1. Warped (Nếu được yêu cầu)
    warped_roi = None
    warped_method = None
    if apply_warping:
        warped, method = detect_and_warp_plate(roi)
        if method != "original":  # Any successful warping method
            warped_roi = warped
            warped_method = method
            # Thêm bản Warped + Gray (Ưu tiên cao nhất)
            if len(warped.shape) == 3:
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            else:
                warped_gray = warped
            variants.append((warped_gray, f"{method}_gray"))
            # IMPORTANT: Thêm ảnh warped màu gốc để lưu vào history
            variants.append((warped, f"{method}_color"))

    # 2. Original (Gray) (Ưu tiên nhì)
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    variants.append((gray, "gray"))
    
    # 3. Gray + CLAHE (Cho ảnh tối/bóng - Rất hiệu quả với EasyOCR)
    clahe = apply_clahe(gray)
    variants.append((clahe, "gray_clahe"))

    # 4. Các biến thể Otsu (Chỉ dùng khi ảnh xám thất bại)
    if warped_roi is not None:
        warped_otsu = apply_threshold(warped_gray, 'otsu')
        variants.append((warped_otsu, f"{warped_method}_otsu"))
        
    otsu = apply_threshold(gray, 'otsu')
    variants.append((otsu, "gray_otsu"))
    


    return variants
