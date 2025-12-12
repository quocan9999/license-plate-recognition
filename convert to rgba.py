import cv2

# 1. Đọc ảnh (Mặc định OpenCV đọc là BGR - 3 kênh)
duong_dan_anh = '595853116_122135385248972121_856348337674392417_n.jpg' # Có thể là jpg hoặc png
hinh_anh = cv2.imread(duong_dan_anh)

# 2. Chuyển đổi sang BGRA (Thêm kênh Alpha)
# Lúc này shape của ảnh sẽ đổi từ (H, W, 3) -> (H, W, 4)
hinh_anh_bgra = cv2.cvtColor(hinh_anh, cv2.COLOR_BGR2BGRA)

# 3. Kiểm tra số kênh để chắc chắn
print(f"Số kênh màu: {hinh_anh_bgra.shape[2]}") # Kết quả sẽ là 4

# 4. Lưu ảnh (BẮT BUỘC lưu đuôi .png để giữ kênh Alpha)
cv2.imwrite('anh_ket_qua_4_kenh.png', hinh_anh_bgra)