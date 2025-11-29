import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import platform
import subprocess
import csv
from datetime import datetime
from modules.detection import LicensePlateDetector
from modules.ocr import LicensePlateOCR


class MultiPlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Nhận diện Biển số xe - EasyOCR + Warping")

        # Tự động phóng to toàn màn hình khi mở
        try:
            self.root.state('zoomed')  # Dành cho Windows
        except:
            self.root.attributes('-zoomed', True)  # Dành cho Linux/Mac

        # Khởi tạo detector và OCR (EasyOCR với Warping)
        self.detector = LicensePlateDetector(model_path="models/best.pt")
        self.ocr = LicensePlateOCR(languages=['en'], gpu=False)

        self.image_refs = []

        # Giao diện chính
        self.top_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
        self.top_frame.pack(fill="x")

        self.btn_select = tk.Button(self.top_frame, text="📂 Chọn nhiều ảnh (Batch)",
                                    command=self.select_images,
                                    font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", padx=20, pady=5)
        self.btn_select.pack()

        tk.Label(self.top_frame, text="(Mẹo: Click đúp vào ảnh để mở xem chi tiết | Sử dụng EasyOCR + Warping)", 
                 bg="#f0f0f0",
                 font=("Arial", 10, "italic")).pack()

        self.canvas = tk.Canvas(root, bg="white")
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = tk.Scrollbar(root, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="white")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.h_scrollbar.pack(side="bottom", fill="x")
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bind_mouse_scroll()

    def bind_mouse_scroll(self):
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def _on_shift_mousewheel(event):
            self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)

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

    def process_and_predict(self, image):
        """
        Xử lý ảnh và nhận diện biển số xe
        
        Args:
            image: PIL Image
            
        Returns:
            tuple: (processed_image_np, detected_plates_list, detections)
        """
        image_np = np.array(image)
        detected_plates = []
        
        # Lấy các vùng ROI của biển số
        plate_regions = self.detector.get_plate_regions(image_np)
        
        detections = []
        for roi, bbox in plate_regions:
            # OCR và xử lý biển số (với warping)
            plate_info = self.ocr.process_plate(roi, apply_warping=True)
            
            if plate_info and self.ocr.is_valid_plate(plate_info):
                vehicle_type = plate_info['vehicle_type']
                formatted_text = plate_info['formatted_text']
                
                # Chuẩn bị text cho UI
                info_for_ui = f"[{vehicle_type}] {formatted_text}"
                detected_plates.append(info_for_ui)
                
                # Thêm vào danh sách detection để vẽ
                detections.append({
                    'bbox': bbox,
                    'text': formatted_text,
                    'vehicle_type': vehicle_type,
                    'roi': roi,
                    'preprocessed_image': plate_info.get('preprocessed_image'),
                    'preprocessing_method': plate_info.get('preprocessing_method')
                })
        
        # Vẽ các detection lên ảnh
        processed_image = self.detector.draw_detections(image_np, detections)
        
        return processed_image, detected_plates, detections

    def save_result(self, original_image_path, original_image_pil, detections):
        # Tạo thư mục History nếu chưa có
        base_dir = "History"
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        save_dir = os.path.join(base_dir, date_str)
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        
        # 1. Lưu ảnh gốc
        # Lấy tên file gốc để dễ truy xuất
        original_filename = os.path.basename(original_image_path)
        name_no_ext = os.path.splitext(original_filename)[0]
        
        # Tên file: YYYYMMDD_HHMMSS_OriginalName.jpg
        save_original_name = f"{timestamp}_{name_no_ext}.jpg"
        save_original_path = os.path.join(save_dir, save_original_name)
        
        # Lưu ảnh gốc (convert RGB để đảm bảo đúng màu khi save bằng PIL)
        original_image_pil.save(save_original_path)
        
        # File log CSV
        csv_file = os.path.join(base_dir, "history.csv")
        file_exists = os.path.isfile(csv_file)
        
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Header
                if not file_exists:
                    writer.writerow(['Timestamp', 'License Plate', 'Vehicle Type', 'Original Image Path', 'ROI Image Path', 'Preprocessed Image Path'])
                
                # Nếu không có biển số nào
                if not detections:
                     writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), "No Plate", "", save_original_path, "", ""])
                
                # 2. Lưu từng biển số cắt được (ROI)
                for i, det in enumerate(detections):
                    plate_text = det['text']
                    vehicle_type = det['vehicle_type']
                    roi = det['roi'] # numpy array (RGB)
                    preprocessed_image = det.get('preprocessed_image')
                    preprocessing_method = det.get('preprocessing_method', 'unknown')
                    
                    # Clean text cho tên file
                    clean_text = "".join(c for c in plate_text if c.isalnum())
                    
                    # Tên file ROI: YYYYMMDD_HHMMSS_BienSo_Index.jpg
                    save_roi_name = f"{timestamp}_{clean_text}_{i}.jpg"
                    save_roi_path = os.path.join(save_dir, save_roi_name)
                    
                    # Lưu ảnh ROI
                    roi_pil = Image.fromarray(roi)
                    roi_pil.save(save_roi_path)
                    
                    # Lưu ảnh Preprocessed (nếu có)
                    save_preprocessed_path = ""
                    if preprocessed_image is not None:
                        # Tên file: ..._processed_method.jpg
                        save_preprocessed_name = f"{timestamp}_{clean_text}_{i}_processed_{preprocessing_method}.jpg"
                        save_preprocessed_path = os.path.join(save_dir, save_preprocessed_name)
                        Image.fromarray(preprocessed_image).save(save_preprocessed_path)
                    
                    # Ghi log
                    writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), plate_text, vehicle_type, save_original_path, save_roi_path, save_preprocessed_path])
        except Exception as e:
            print(f"Lỗi khi lưu lịch sử: {e}")

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
                processed_img_np, plates, detections = self.process_and_predict(img_pil)
                
                # Lưu kết quả vào History
                self.save_result(file_path, img_pil, detections)
                
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
                import traceback
                traceback.print_exc()

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