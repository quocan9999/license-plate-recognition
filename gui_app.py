import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
from utils import process_and_predict  # Import hàm xử lý từ file utils cũ


class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phần mềm Nhận diện Biển số xe")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")

        # --- LOAD MODEL ---
        self.model = None
        self.load_model()

        # --- GIAO DIỆN ---
        # 1. Tiêu đề
        self.lbl_title = tk.Label(root, text="HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE",
                                  font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#cc0000")
        self.lbl_title.pack(pady=20)

        # 2. Khu vực hiển thị ảnh (Dùng Canvas hoặc Label)
        self.frame_img = tk.Frame(root, bg="white", bd=2, relief="sunken")
        self.frame_img.pack(pady=10)

        self.lbl_image = tk.Label(self.frame_img, text="Chưa chọn ảnh", bg="#e0e0e0", width=80, height=20)
        self.lbl_image.pack()

        # 3. Khu vực nút bấm
        self.frame_controls = tk.Frame(root, bg="#f0f0f0")
        self.frame_controls.pack(pady=20)

        self.btn_select = tk.Button(self.frame_controls, text="📂 Chọn Ảnh", command=self.select_image,
                                    font=("Arial", 12), bg="#4CAF50", fg="white", width=15)
        self.btn_select.grid(row=0, column=0, padx=10)

        # Nút xử lý (ban đầu ẩn hoặc disable, khi có ảnh mới cho bấm)
        self.btn_process = tk.Button(self.frame_controls, text="⚡ Nhận diện ngay", command=self.run_detection,
                                     font=("Arial", 12), bg="#2196F3", fg="white", width=15, state="disabled")
        self.btn_process.grid(row=0, column=1, padx=10)

        # 4. Khu vực kết quả
        self.lbl_result_title = tk.Label(root, text="KẾT QUẢ:", font=("Arial", 14, "bold"), bg="#f0f0f0")
        self.lbl_result_title.pack()

        self.lbl_result_text = tk.Label(root, text="---", font=("Arial", 30, "bold"), fg="#2E7D32", bg="#f0f0f0")
        self.lbl_result_text.pack(pady=10)

        # Biến lưu trữ
        self.current_image_path = None
        self.current_image_pil = None  # Ảnh gốc dạng PIL

    def load_model(self):
        try:
            # Đường dẫn tương đối, đảm bảo file best.pt nằm đúng chỗ
            self.model = YOLO("models/best.pt")
            print("Đã load model thành công!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không tìm thấy model best.pt\nChi tiết: {e}")
            # Load tạm model n mặc định để không crash app
            self.model = YOLO("yolov8n.pt")

    def select_image(self):
        # Mở hộp thoại chọn file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.current_image_path = file_path

            # Load ảnh bằng PIL
            self.current_image_pil = Image.open(file_path)

            # Hiển thị ảnh lên giao diện (Resize cho vừa khung nhìn)
            self.display_image(self.current_image_pil)

            # Reset kết quả cũ
            self.lbl_result_text.config(text="---")
            self.btn_process.config(state="normal")  # Cho phép bấm nút xử lý

    def display_image(self, img_pil):
        # Resize ảnh để hiển thị vừa vặn trong GUI (Thumbnail)
        # Giữ nguyên tỉ lệ khung hình
        base_width = 600
        w_percent = (base_width / float(img_pil.size[0]))
        h_size = int((float(img_pil.size[1]) * float(w_percent)))

        # Giới hạn chiều cao tối đa
        if h_size > 400:
            h_size = 400
            w_percent = (h_size / float(img_pil.size[1]))
            base_width = int((float(img_pil.size[0]) * float(w_percent)))

        img_resized = img_pil.resize((base_width, h_size), Image.Resampling.LANCZOS)

        # Chuyển sang định dạng Tkinter hỗ trợ
        self.tk_image = ImageTk.PhotoImage(img_resized)

        self.lbl_image.config(image=self.tk_image, width=0, height=0)  # Reset width/height text
        self.lbl_image.image = self.tk_image  # Giữ tham chiếu để không bị Garbage Collection xóa mất

    def run_detection(self):
        if self.model is None or self.current_image_pil is None:
            return

        # Cập nhật UI báo đang chạy
        self.lbl_result_text.config(text="Đang xử lý...", fg="orange")
        self.root.update_idletasks()  # Bắt buộc lệnh này để GUI vẽ lại chữ ngay lập tức

        try:
            # Gọi hàm xử lý từ utils.py
            # Lưu ý: utils trả về (ảnh_numpy, list_biển_số)
            processed_img_np, plates = process_and_predict(self.current_image_pil, self.model)

            # 1. Hiển thị ảnh kết quả (đã vẽ khung)
            # Vì OpenCV dùng BGR, PIL dùng RGB -> Cần convert màu
            # Nhưng utils của bạn có thể trả về RGB sẵn nếu logic vẽ dùng PIL,
            # tuy nhiên utils ở câu trước dùng cv2 vẽ nên là numpy array.

            # Convert Numpy Array -> PIL Image
            result_pil = Image.fromarray(processed_img_np)
            self.display_image(result_pil)

            # 2. Hiển thị text biển số
            if plates:
                # Nối các biển số lại nếu có nhiều xe (xuống dòng)
                text_result = "\n".join(plates)
                self.lbl_result_text.config(text=text_result, fg="#2E7D32")  # Màu xanh lá
            else:
                self.lbl_result_text.config(text="Không tìm thấy biển số", fg="red")

        except Exception as e:
            messagebox.showerror("Lỗi Xử Lý", f"Có lỗi xảy ra: {e}")
            self.lbl_result_text.config(text="Lỗi", fg="red")


# --- CHẠY APP ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()