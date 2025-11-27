import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import platform
import subprocess
from ultralytics import YOLO
from utils import process_and_predict


class MultiPlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Nhận diện Biển số Hàng loạt")

        # --- CẤU HÌNH CỬA SỔ ---
        # Tự động phóng to toàn màn hình khi mở
        try:
            self.root.state('zoomed')  # Dành cho Windows
        except:
            self.root.attributes('-zoomed', True)  # Dành cho Linux/Mac

        # --- 1. LOAD MODEL ---
        self.model = None
        self.load_model()

        self.image_refs = []

        # --- 2. GIAO DIỆN CHÍNH ---
        self.top_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
        self.top_frame.pack(fill="x")

        self.btn_select = tk.Button(self.top_frame, text="📂 Chọn nhiều ảnh (Batch)",
                                    command=self.select_images,
                                    font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", padx=20, pady=5)
        self.btn_select.pack()

        tk.Label(self.top_frame, text="(Mẹo: Click đúp vào ảnh để mở xem chi tiết)", bg="#f0f0f0",
                 font=("Arial", 10, "italic")).pack()

        # --- 3. TẠO VÙNG CUỘN (SCROLLABLE AREA) ---
        self.canvas = tk.Canvas(root, bg="white")
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="white")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.bind_mouse_scroll()

    def bind_mouse_scroll(self):
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def load_model(self):
        try:
            self.model = YOLO("models/best.pt")
            print("Đã load model custom!")
        except:
            self.model = YOLO("yolov8n.pt")

    def select_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Chọn các ảnh xe cần xử lý",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if file_paths:
            self.process_batch(file_paths)

    def open_image_external(self, event, file_path):
        """Hàm mở ảnh bằng phần mềm mặc định của hệ thống khi click đúp"""
        try:
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', file_path))
            else:  # Linux
                subprocess.call(('xdg-open', file_path))
        except Exception as e:
            print(f"Không mở được file: {e}")

    def process_batch(self, file_paths):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.image_refs = []

        for index, file_path in enumerate(file_paths):
            stt = index + 1

            # Khung chứa 1 dòng
            row_frame = tk.Frame(self.scrollable_frame, bg="white", bd=2, relief="groove")
            row_frame.pack(fill="x", padx=10, pady=10)

            # Header
            lbl_header = tk.Label(row_frame, text=f"Hồ sơ ảnh #{stt}", font=("Arial", 11, "bold"), bg="#ddd",
                                  anchor="w", padx=10)
            lbl_header.pack(fill="x")

            content_frame = tk.Frame(row_frame, bg="white")
            content_frame.pack(pady=10)

            try:
                img_pil = Image.open(file_path)
                processed_img_np, plates = process_and_predict(img_pil, self.model)
                result_pil = Image.fromarray(processed_img_np)

                # --- CỘT 1: ẢNH GỐC ---
                col1 = tk.Frame(content_frame, bg="white")
                col1.grid(row=0, column=0, padx=20)

                # Resize to hơn (450px)
                thumb_orig = self.resize_image(img_pil, fixed_height=450)
                tk_thumb_orig = ImageTk.PhotoImage(thumb_orig)
                self.image_refs.append(tk_thumb_orig)

                lbl_img1 = tk.Label(col1, image=tk_thumb_orig, cursor="hand2")
                lbl_img1.pack()
                # Gán sự kiện Click đúp -> Mở ảnh gốc full size
                lbl_img1.bind("<Double-Button-1>", lambda e, path=file_path: self.open_image_external(e, path))

                tk.Label(col1, text="Ảnh gốc (Click đúp để phóng to)", font=("Arial", 10, "italic"), bg="white").pack()

                # --- CỘT 2: ẢNH XỬ LÝ ---
                col2 = tk.Frame(content_frame, bg="white")
                col2.grid(row=0, column=1, padx=20)

                thumb_res = self.resize_image(result_pil, fixed_height=450)
                tk_thumb_res = ImageTk.PhotoImage(thumb_res)
                self.image_refs.append(tk_thumb_res)

                tk.Label(col2, image=tk_thumb_res).pack()
                tk.Label(col2, text="Ảnh đã nhận diện", font=("Arial", 10, "italic"), bg="white").pack()

                # --- CỘT 3: KẾT QUẢ ---
                col3 = tk.Frame(content_frame, bg="white")
                col3.grid(row=0, column=2, padx=30, sticky="n")  # Sticky n để chữ nằm phía trên

                # Tạo khoảng trống phía trên để chữ ngang tầm mắt hơn
                tk.Frame(col3, height=50, bg="white").pack()

                if plates:
                    result_text = ""
                    for p in plates:
                        if "]" in p:
                            type_part, number_part = p.split("]", 1)
                            type_clean = type_part.replace("[", "")
                            result_text += f"{type_clean} - {number_part.strip()}\n"
                        else:
                            result_text += f"{p}\n"

                    tk.Label(col3, text=result_text, font=("Arial", 20, "bold"), fg="#2E7D32", bg="white",
                             justify="left").pack()
                else:
                    tk.Label(col3, text="Không tìm thấy\nbiển số", font=("Arial", 16), fg="red", bg="white").pack()

            except Exception as e:
                print(f"Lỗi: {e}")

            self.root.update()

    def resize_image(self, img_pil, fixed_height):
        # Hàm resize giữ nguyên tỉ lệ
        h_percent = (fixed_height / float(img_pil.size[1]))
        w_size = int((float(img_pil.size[0]) * float(h_percent)))
        return img_pil.resize((w_size, fixed_height), Image.Resampling.LANCZOS)


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiPlateApp(root)
    root.mainloop()