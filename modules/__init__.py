"""
Module nhận diện biển số xe Việt Nam
Bao gồm: Detection, OCR, Preprocessing, và các hàm hỗ trợ
"""

from .detection import LicensePlateDetector
from .ocr import LicensePlateOCR
from .preprocessing import preprocess_for_ocr
from .utils import (
    classify_vehicle,
    validate_province_code,
    fix_plate_chars,
    format_plate,
    VALID_PROVINCE_CODES,
    dict_char_to_num,
    dict_num_to_char
)

from .logger import HistoryLogger

__all__ = [
    'LicensePlateDetector',
    'LicensePlateOCR',
    'HistoryLogger',
    'preprocess_for_ocr',
    'classify_vehicle',
    'validate_province_code',
    'fix_plate_chars',
    'format_plate',
    'VALID_PROVINCE_CODES',
    'dict_char_to_num',
    'dict_num_to_char'
]
