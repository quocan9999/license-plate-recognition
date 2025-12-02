"""
Module chứa các cấu hình và hằng số cho toàn bộ dự án
"""
import os

# --- PATHS ---
# Thư mục gốc của project (giả sử file này nằm trong modules/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn Model
MODEL_PATH = "models/best_yolov8l_new.pt"
FALLBACK_MODEL_PATH = "yolov8l.pt"

# Thư mục lưu lịch sử
HISTORY_DIR = "History"
HISTORY_CSV_FILE = "history.csv"

# --- OCR SETTINGS ---
OCR_LANGUAGES = ['en']
OCR_GPU = False

# --- PREPROCESSING SETTINGS ---
# CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Upscaling
UPSCALE_SCALE = 2

# Warping
WARP_PADDING = 10

# Adaptive Threshold
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_C = 9

# --- VISUALIZATION SETTINGS ---
COLOR_DEFAULT = (0, 255, 0)      # Green
COLOR_MOTO = (0, 255, 0)         # Green
COLOR_CAR = (0, 165, 255)        # Orange
BBOX_THICKNESS = 3
TEXT_FONT_SCALE = 0.8
TEXT_THICKNESS = 4

# --- UTILS SETTINGS ---
VALID_PROVINCE_START = 11
VALID_PROVINCE_END = 100
