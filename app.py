# app.py
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from utils import process_and_predict

# --- Cấu hình trang ---
st.set_page_config(page_title="Nhận diện biển số xe Việt Nam", layout="wide")

# --- CSS tùy chỉnh để chữ to rõ ---
st.markdown("""
<style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: #FF4B4B;
    }
    .label-font {
        font-size:20px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Model (Cache để không load lại mỗi lần upload) ---
@st.cache_resource
def load_model():
    # Thay đường dẫn này bằng đường dẫn file best.pt của bạn sau khi train
    return YOLO("models/best.pt")


try:
    model = load_model()
except:
    st.warning("Chưa tìm thấy file weights custom. Đang sử dụng yolov8n.pt mặc định để demo.")
    model = YOLO("yolov8n.pt")

# --- Giao diện chính ---
st.title("📸 Hệ Thống Nhận Diện Biển Số Xe (Việt Nam)")
st.write("Hỗ trợ nhận diện biển số xe máy và ô tô.")

uploaded_file = st.file_uploader("Tải ảnh xe lên tại đây...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Chia cột: Bên trái ảnh gốc, Bên phải kết quả
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh gốc', use_container_width=True)

    with col2:
        # 1. TẠO MỘT PLACEHOLDER (Chỗ giữ chỗ)
        status_text = st.empty()

        # 2. GHI CHỮ VÀO PLACEHOLDER ĐÓ
        status_text.markdown('<p class="label-font">Đang xử lý...</p>', unsafe_allow_html=True)

        # 3. THỰC HIỆN XỬ LÝ (Máy tính chạy nặng ở bước này)
        processed_image, plates = process_and_predict(image, model)

        # 4. XÓA CHỮ "Đang xử lý..." ĐI (Quan trọng)
        status_text.empty()

        # --- HIỂN THỊ KẾT QUẢ ---
        # Hiển thị ảnh đã vẽ khung
        st.image(processed_image, caption='Kết quả nhận diện', use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="label-font">KẾT QUẢ BIỂN SỐ:</p>', unsafe_allow_html=True)

        if plates:
            for plate in plates:
                st.markdown(f'<p class="big-font">{plate}</p>', unsafe_allow_html=True)
        else:
            st.error("Không tìm thấy biển số nào trong ảnh.")