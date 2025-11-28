# Modules - Hệ thống Nhận diện Biển số xe Việt Nam

Package này chứa các module chính cho hệ thống nhận diện biển số xe Việt Nam.

## Cấu trúc Module

```
modules/
├── __init__.py          # Package initialization và exports
├── detection.py         # Module phát hiện biển số (YOLO)
├── ocr.py               # Module OCR và xử lý text
├── preprocessing.py     # Module tiền xử lý ảnh
└── utils.py             # Module các hàm hỗ trợ
```

## Chi tiết các Module

### 1. `detection.py` - Module Phát hiện Biển số

**Class: `LicensePlateDetector`**

Chức năng:
- Load và quản lý YOLO model
- Phát hiện vùng biển số trong ảnh
- Trích xuất ROI (Region of Interest)
- Vẽ bounding box lên ảnh

Ví dụ sử dụng:
```python
from modules.detection import LicensePlateDetector

detector = LicensePlateDetector(model_path="models/best.pt")
plate_regions = detector.get_plate_regions(image)
```

### 2. `ocr.py` - Module OCR

**Class: `LicensePlateOCR`**

Chức năng:
- Khởi tạo EasyOCR reader
- Đọc text từ ảnh biển số
- Xử lý và sửa lỗi ký tự
- Phân loại loại xe (Ô tô/Xe máy)
- Format biển số theo chuẩn Việt Nam

Ví dụ sử dụng:
```python
from modules.ocr import LicensePlateOCR

ocr = LicensePlateOCR(languages=['en'], gpu=False)
plate_info = ocr.process_plate(roi)
```

### 3. `preprocessing.py` - Module Tiền xử lý

**Functions:**
- `preprocess_for_ocr(roi)` - Tiền xử lý cơ bản (grayscale)
- `apply_adaptive_threshold(gray_image)` - Adaptive thresholding
- `denoise_image(image)` - Khử nhiễu
- `enhance_contrast(image)` - Tăng cường độ tương phản

Ví dụ sử dụng:
```python
from modules.preprocessing import preprocess_for_ocr, enhance_contrast

preprocessed = preprocess_for_ocr(roi)
enhanced = enhance_contrast(preprocessed)
```

### 4. `utils.py` - Module Hỗ trợ

**Functions:**

#### Phân loại và Validation
- `classify_vehicle(ocr_list)` - Phân loại Ô tô/Xe máy
- `validate_province_code(code_str)` - Kiểm tra mã tỉnh hợp lệ

#### Sửa lỗi và Format
- `fix_plate_chars(raw_text, is_50cc=False)` - Sửa lỗi ký tự OCR
- `format_plate(text, vehicle_type)` - Format biển số theo chuẩn VN

#### Constants
- `VALID_PROVINCE_CODES` - Set mã tỉnh hợp lệ (11-99)
- `dict_char_to_num` - Mapping chữ -> số
- `dict_num_to_char` - Mapping số -> chữ

Ví dụ sử dụng:
```python
from modules.utils import classify_vehicle, fix_plate_chars, format_plate

vehicle_type = classify_vehicle(ocr_result)
clean_text = fix_plate_chars(raw_text, is_50cc=False)
formatted = format_plate(clean_text, vehicle_type)
```

## Cấu trúc Biển số Việt Nam

### Ô tô
- **1 dòng**: `30A12345` (mã tỉnh + chữ + 5 số)
- **2 dòng**: `37A / 555.55` (dòng 1: mã tỉnh + chữ, dòng 2: số)
- **Format**: `30A-123.45` hoặc `30A-4264`

### Xe máy thường
- **2 dòng**: `29A1 / 123.45` (dòng 1: mã tỉnh + chữ + số, dòng 2: số)
- **Format**: `29-A1 123.45` hoặc `29-A1 1234`

### Xe máy 50cc
- **2 dòng**: `29AA / 12345` (dòng 1: mã tỉnh + 2 chữ, dòng 2: số)
- **Format**: `29-AA 123.45` hoặc `29-AA 1234`

## Thuật toán Sửa lỗi Ký tự

Hệ thống sử dụng mapping thông minh để sửa lỗi OCR:

### Chữ -> Số (cho vị trí phải là số)
- `I, L, T` → `1`
- `O, Q, D, U` → `0`
- `B, E` → `8`
- `S` → `5`
- `Z, R` → `2`
- `G, C` → `6`
- `A` → `4`
- `J` → `3`

### Số -> Chữ (cho vị trí phải là chữ)
- `0` → `D`
- `1` → `I`
- `2` → `Z`
- `3, 8` → `B`
- `4` → `A`
- `5` → `S`
- `6` → `G`

## Sử dụng trong GUI

File `gui_multi.py` đã được refactor để sử dụng các module này:

```python
from modules.detection import LicensePlateDetector
from modules.ocr import LicensePlateOCR

# Trong class MultiPlateApp
self.detector = LicensePlateDetector(model_path="models/best.pt")
self.ocr = LicensePlateOCR(languages=['en'], gpu=False)

# Xử lý ảnh
plate_regions = self.detector.get_plate_regions(image)
for roi, bbox in plate_regions:
    plate_info = self.ocr.process_plate(roi)
```

## Lợi ích của Refactoring

1. **Tách biệt trách nhiệm**: Mỗi module có một chức năng rõ ràng
2. **Dễ bảo trì**: Code được tổ chức tốt hơn, dễ tìm và sửa lỗi
3. **Tái sử dụng**: Các module có thể được import và sử dụng ở nhiều nơi
4. **Mở rộng**: Dễ dàng thêm tính năng mới vào từng module
5. **Testing**: Dễ dàng viết unit test cho từng module riêng biệt

## Dependencies

- `opencv-python` (cv2)
- `numpy`
- `easyocr`
- `ultralytics` (YOLO)
- `Pillow` (PIL)
