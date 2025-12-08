"""Ứng dụng GUI cho nhận diện biển số xe với xử lý hàng loạt.

Module này cung cấp giao diện người dùng để phát hiện và nhận diện
biển số xe từ ảnh sử dụng YOLOv8 và PaddleOCR.
"""

import os
import platform
import subprocess
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tkinterdnd2 import DND_FILES, TkinterDnD
from modules.detection import LicensePlateDetector
from modules.ocr import LicensePlateOCR
from modules.logger import HistoryLogger
from modules.config import HISTORY_DIR


class MultiPlateApp:
    """Class chính cho ứng dụng GUI nhận diện biển số xe.

    Class này quản lý giao diện người dùng, xử lý ảnh,
    và điều phối giữa các module phát hiện và OCR.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Nhận diện Biển số xe")

        # Tự động phóng to toàn màn hình khi mở
        try:
            self.root.state('zoomed')  # Dành cho Windows
        except:
            self.root.attributes('-zoomed', True)  # Dành cho Linux/Mac

        # Cấu hình Drag & Drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_files)

        # Khởi tạo detector và OCR (PaddleOCR với Warping)
        self.detector = LicensePlateDetector()
        self.ocr = LicensePlateOCR()
        self.logger = HistoryLogger()

        self.image_refs = []
        
        # Biến theo dõi thời gian xử lý
        self.processing_start_time = None
        self.image_processing_times = []
        
        # Dữ liệu kết quả xử lý
        self.results = []  # List of dicts: {file_path, processed_img, cropped_plate, plate_text, processing_time, ...}
        self.current_index = 0  # Index của ảnh đang được chọn
        self.auto_playing = False  # Đang tự động duyệt hay không
        self.auto_play_id = None  # ID của after() để có thể cancel
        
        # Tổng thời gian xử lý
        self.total_processing_time = 0

        # Xây dựng giao diện
        self.build_ui()

    def build_ui(self):
        """Xây dựng giao diện chính"""
        # ========== TOP FRAME - Buttons ==========
        self.top_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        self.top_frame.pack(fill="x")

        # Frame chứa các nút điều khiển
        self.btn_frame = tk.Frame(self.top_frame, bg="#f0f0f0")
        self.btn_frame.pack()

        self.btn_select = tk.Button(self.btn_frame, text="📂 Chọn nhiều ảnh (Batch)",
                                    command=self.select_images,
                                    font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", padx=20, pady=5)
        self.btn_select.pack(side="left", padx=10)

        self.btn_history = tk.Button(self.btn_frame, text="📂 Mở thư mục History",
                                     command=self.open_history_folder,
                                     font=("Arial", 14, "bold"), bg="#FF9800", fg="white", padx=20, pady=5)
        self.btn_history.pack(side="left", padx=10)

        # Label hướng dẫn Drag & Drop
        tk.Label(self.top_frame, text="(Mẹo: Kéo thả ảnh vào đây hoặc Click đúp vào ảnh để mở xem chi tiết)", 
                 bg="#f0f0f0",
                 font=("Arial", 10, "italic")).pack()

        # ========== MAIN CONTENT FRAME ==========
        # Container chính với viền đen
        self.main_container = tk.Frame(self.root, bg="white", bd=2, relief="solid")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Frame nội dung bên trong (không viền)
        self.main_frame = tk.Frame(self.main_container, bg="white")
        self.main_frame.pack(fill="both", expand=True)
        
        # Cấu hình grid cho 3 cột + 2 separator
        self.main_frame.columnconfigure(0, weight=3)  # Cột 1: Ảnh bounding box
        self.main_frame.columnconfigure(1, weight=0, minsize=2)  # Separator 1
        self.main_frame.columnconfigure(2, weight=2)  # Cột 2: Ảnh biển số cắt + info
        self.main_frame.columnconfigure(3, weight=0, minsize=2)  # Separator 2
        self.main_frame.columnconfigure(4, weight=1)  # Cột 3: Danh sách file
        self.main_frame.rowconfigure(0, weight=1)

        # ========== CỘT 1: Ảnh biển số đã vẽ bounding box ==========
        self.col1_frame = tk.Frame(self.main_frame, bg="white")
        self.col1_frame.grid(row=0, column=0, sticky="nsew")
        
        # Canvas để hiển thị ảnh
        self.img_bbox_label = tk.Label(self.col1_frame, bg="white", cursor="hand2")
        self.img_bbox_label.pack(expand=True, fill="both", padx=10, pady=10)
        self.img_bbox_label.bind("<Double-Button-1>", self.open_current_image)

        # ========== SEPARATOR 1 ==========
        self.sep1 = tk.Frame(self.main_frame, bg="black", width=2)
        self.sep1.grid(row=0, column=1, sticky="ns")

        # ========== CỘT 2: Ảnh biển số cắt + thông tin ==========
        self.col2_frame = tk.Frame(self.main_frame, bg="#FFFDE7")
        self.col2_frame.grid(row=0, column=2, sticky="nsew")
        
        # Frame chứa ảnh biển số cắt (sẽ được tạo động trong display_result)
        self.plate_img_frame = tk.Frame(self.col2_frame, bg="#FFFDE7")
        self.plate_img_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # ========== Phần hiển thị thông tin biển số ==========
        self.info_frame = tk.Frame(self.col2_frame, bg="#FFFDE7")
        self.info_frame.pack(fill="x", padx=10, pady=10)
        
        # Text biển số nhận diện được (to, dễ nhìn)
        self.plate_text_label = tk.Label(self.info_frame, text="", 
                                         font=("Arial", 28, "bold"), fg="#1565C0", bg="#FFFDE7")
        self.plate_text_label.pack(pady=10)
        
        # Thời gian xử lý từng ảnh
        self.time_label = tk.Label(self.info_frame, text="Thời gian xử lý: ---", 
                                   font=("Arial", 12), bg="#FFFDE7", fg="#333")
        self.time_label.pack(pady=5)
        
        # Tổng thời gian xử lý
        self.total_time_label = tk.Label(self.info_frame, text="Tổng thời gian: ---", 
                                         font=("Arial", 12), bg="#FFFDE7", fg="#666")
        self.total_time_label.pack(pady=5)
        
        # ========== Các nút điều khiển ==========
        self.control_frame = tk.Frame(self.col2_frame, bg="#FFFDE7")
        self.control_frame.pack(fill="x", padx=10, pady=15)
        
        # Nút Tự động
        self.btn_auto = tk.Button(self.control_frame, text="Tự động", 
                                  command=self.toggle_auto_play,
                                  font=("Arial", 14, "bold"), bg="#2196F3", fg="white", 
                                  padx=20, pady=8, width=10)
        self.btn_auto.pack(pady=5)
        
        # Frame chứa nút Tiếp tục và Lùi
        self.nav_frame = tk.Frame(self.control_frame, bg="#FFFDE7")
        self.nav_frame.pack(pady=5)
        
        self.btn_next = tk.Button(self.nav_frame, text="Tiếp tục", 
                                  command=self.next_image,
                                  font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", 
                                  padx=15, pady=5, width=8)
        self.btn_next.pack(side="left", padx=5)
        
        self.btn_prev = tk.Button(self.nav_frame, text="Lùi", 
                                  command=self.prev_image,
                                  font=("Arial", 12, "bold"), bg="#FF9800", fg="white", 
                                  padx=15, pady=5, width=8)
        self.btn_prev.pack(side="left", padx=5)

        # ========== SEPARATOR 2 ==========
        self.sep2 = tk.Frame(self.main_frame, bg="black", width=2)
        self.sep2.grid(row=0, column=3, sticky="ns")

        # ========== CỘT 3: Danh sách file ảnh ==========
        self.col3_frame = tk.Frame(self.main_frame, bg="#E3F2FD")
        self.col3_frame.grid(row=0, column=4, sticky="nsew")
        
        # Label mô tả
        tk.Label(self.col3_frame, text="Danh sách ảnh", 
                 font=("Arial", 12, "bold"), bg="#E3F2FD", fg="#1565C0").pack(pady=10)
        
        # Frame chứa listbox và scrollbar
        self.list_frame = tk.Frame(self.col3_frame, bg="#E3F2FD")
        self.list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Scrollbar cho listbox
        self.list_scrollbar = tk.Scrollbar(self.list_frame)
        self.list_scrollbar.pack(side="right", fill="y")
        
        # Listbox hiển thị danh sách file
        self.file_listbox = tk.Listbox(self.list_frame, font=("Arial", 11), 
                                        yscrollcommand=self.list_scrollbar.set,
                                        selectmode=tk.SINGLE, activestyle="dotbox",
                                        selectbackground="#1976D2", selectforeground="white")
        self.file_listbox.pack(fill="both", expand=True)
        self.list_scrollbar.config(command=self.file_listbox.yview)
        
        # Bind sự kiện chọn item
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)
        self.file_listbox.bind("<Double-Button-1>", self.on_file_double_click)
        
        # Label trạng thái
        self.status_label = tk.Label(self.col3_frame, text="Chưa có ảnh", 
                                     font=("Arial", 10, "italic"), bg="#E3F2FD", fg="#666")
        self.status_label.pack(pady=5)
        
        # Bind phím mũi tên lên/xuống cho toàn bộ cửa sổ
        self.root.bind("<Up>", self.on_arrow_up)
        self.root.bind("<Down>", self.on_arrow_down)

    def bind_mouse_scroll(self):
        """Bind mouse scroll cho listbox"""
        def _on_mousewheel(event):
            self.file_listbox.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.file_listbox.bind("<MouseWheel>", _on_mousewheel)

    def drop_files(self, event):
        """Xử lý sự kiện kéo thả file"""
        file_paths = self.parse_drop_files(event.data)
        if file_paths:
            self.process_batch(file_paths)

    def parse_drop_files(self, data):
        """Phân tích chuỗi dữ liệu từ sự kiện drop"""
        pattern = r'\{.*?\}|\S+'
        matches = re.findall(pattern, data)
        
        cleaned_paths = []
        for match in matches:
            path = match.strip('{}')
            if os.path.isfile(path):
                cleaned_paths.append(path)
        
        return cleaned_paths

    def select_images(self):
        """Chọn nhiều ảnh từ dialog"""
        file_paths = filedialog.askopenfilenames(
            title="Chọn các ảnh xe cần xử lý",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if file_paths:
            self.process_batch(file_paths)

    def open_history_folder(self):
        """Mở thư mục History"""
        if not os.path.exists(HISTORY_DIR):
            os.makedirs(HISTORY_DIR)
            
        try:
            if platform.system() == 'Windows':
                os.startfile(HISTORY_DIR)
            elif platform.system() == 'Darwin':
                subprocess.call(('open', HISTORY_DIR))
            else:
                subprocess.call(('xdg-open', HISTORY_DIR))
        except Exception as e:
            print(f"Không mở được thư mục history: {e}")

    def open_image_external(self, file_path):
        """Mở ảnh bằng phần mềm mặc định của hệ thống"""
        try:
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':
                subprocess.call(('open', file_path))
            else:
                subprocess.call(('xdg-open', file_path))
        except Exception as e:
            print(f"Không mở được file: {e}")

    def open_current_image(self, event=None):
        """Mở ảnh đang được chọn"""
        if self.results and 0 <= self.current_index < len(self.results):
            self.open_image_external(self.results[self.current_index]['file_path'])

    def process_and_predict(self, image, image_index=None):
        """
        Xử lý ảnh và nhận diện biển số xe
        
        Returns:
            tuple: (processed_image_np, detected_plates_list, detections, cropped_plate_img)
        """
        image_np = np.array(image)
        detected_plates = []
        cropped_plate_img = None
        
        # Lấy các vùng ROI của biển số
        plate_regions = self.detector.get_plate_regions(image_np, image_index=image_index)
        
        detections = []
        valid_plates = []
        
        # Thu thập tất cả các biển số hợp lệ
        for roi, bbox in plate_regions:
            plate_info = self.ocr.process_plate(roi, apply_warping=True)
            
            if plate_info and self.ocr.is_valid_plate(plate_info):
                valid_plates.append((plate_info, bbox, roi))

        # Format kết quả
        num_plates = len(valid_plates)
        
        # List chứa tất cả các ảnh biển số cắt
        cropped_plates_list = []
        
        for i, (plate_info, bbox, roi) in enumerate(valid_plates):
            vehicle_type = plate_info['vehicle_type']
            formatted_text = plate_info['formatted_text']
            
            prefix = f"#{i+1} " if num_plates > 1 else ""
            info_for_ui = f"{prefix}[{vehicle_type}] {formatted_text}"
            detected_plates.append(info_for_ui)
            
            # Lấy tất cả ảnh biển số cắt
            cropped_plates_list.append(roi)
            
            # Giữ lại cropped_plate_img đầu tiên cho backward compatibility
            if cropped_plate_img is None:
                cropped_plate_img = roi
            
            detections.append({
                'bbox': bbox,
                'text': formatted_text,
                'vehicle_type': vehicle_type,
                'roi': roi,
                'preprocessed_image': plate_info.get('preprocessed_image'),
                'preprocessing_method': plate_info.get('preprocessing_method'),
                'intermediate_images': plate_info.get('intermediate_images')
            })
        
        # Vẽ các detection lên ảnh
        processed_image = self.detector.draw_detections(image_np, detections)
        
        return processed_image, detected_plates, detections, cropped_plate_img, cropped_plates_list

    def process_batch(self, file_paths):
        """Bắt đầu xử lý batch trong thread riêng"""
        # Dừng auto play nếu đang chạy
        self.stop_auto_play()
        
        # Reset dữ liệu
        self.results = []
        self.current_index = 0
        self.image_refs = []
        self.total_processing_time = 0
        
        # Xóa listbox
        self.file_listbox.delete(0, tk.END)
        
        # Reset hiển thị
        self.img_bbox_label.config(image='')
        # Xóa các widget trong plate_img_frame
        for widget in self.plate_img_frame.winfo_children():
            widget.destroy()
        self.plate_text_label.config(text="")
        self.time_label.config(text="Thời gian xử lý: ---")
        self.total_time_label.config(text="Tổng thời gian: ---")
        
        # Cập nhật trạng thái
        self.btn_select.config(state="disabled")
        self.btn_history.config(state="disabled")
        self.status_label.config(text=f"Đang xử lý 0/{len(file_paths)} ảnh...")
        
        # Chạy thread xử lý
        threading.Thread(target=self.processing_thread, args=(file_paths,), daemon=True).start()

    def processing_thread(self, file_paths):
        """Hàm thực thi trong background thread"""
        total = len(file_paths)
        
        # Bắt đầu tính tổng thời gian
        self.processing_start_time = time.time()
        self.image_processing_times = []
        
        print(f"\n🚀 Bắt đầu xử lý batch {total} ảnh...")
        print("=" * 60)
        
        for index, file_path in enumerate(file_paths):
            stt = index + 1
            
            # Cập nhật trạng thái
            self.root.after(0, lambda s=stt, t=total: self.status_label.config(
                text=f"Đang xử lý {s}/{t} ảnh..."))
            
            # Bắt đầu tính thời gian cho ảnh này
            image_start_time = time.time()
            
            try:
                print(f"\n📸 ===== ẢNH #{stt} =====\nFile: {os.path.basename(file_path)}")
                
                # Xử lý (Detect + OCR)
                img_pil = Image.open(file_path)
                processed_img_np, plates, detections, cropped_plate, cropped_plates_list = self.process_and_predict(img_pil, image_index=stt)
                
                result_pil = Image.fromarray(processed_img_np)
                
                # Tính thời gian xử lý ảnh này
                image_end_time = time.time()
                image_time = image_end_time - image_start_time
                self.image_processing_times.append(image_time)
                
                print(f"✅ Ảnh #{stt} hoàn thành trong {image_time:.2f}s")
                if plates:
                    print(f"🎯 Kết quả: {', '.join(plates)}")
                else:
                    print("❌ Không phát hiện biển số")
                
                # Lưu kết quả vào History
                self.logger.save_result(file_path, img_pil, detections, processed_image_pil=result_pil)
                
                # Tạo dict kết quả
                # Chuyển đổi tất cả cropped plates sang PIL Image
                cropped_plates_pil = []
                for cp in cropped_plates_list:
                    if cp is not None:
                        cropped_plates_pil.append(Image.fromarray(cp))
                
                result = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'original_img': img_pil,
                    'processed_img': result_pil,
                    'cropped_plate': Image.fromarray(cropped_plate) if cropped_plate is not None else None,
                    'cropped_plates': cropped_plates_pil,  # Danh sách tất cả biển số cắt
                    'plates': plates,
                    'processing_time': image_time
                }
                self.results.append(result)
                
                # Cập nhật UI (gửi về Main Thread)
                self.root.after(0, self.add_result_to_list, index, result)
                
            except Exception as e:
                image_end_time = time.time()
                image_time = image_end_time - image_start_time
                self.image_processing_times.append(image_time)
                
                print(f"❌ Lỗi xử lý ảnh #{stt}: {e}")
                import traceback
                traceback.print_exc()
                
                # Thêm kết quả lỗi
                result = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'original_img': None,
                    'processed_img': None,
                    'cropped_plate': None,
                    'cropped_plates': [],  # Danh sách rỗng khi lỗi
                    'plates': [],
                    'processing_time': image_time,
                    'error': str(e)
                }
                self.results.append(result)
                self.root.after(0, self.add_result_to_list, index, result)

        # Hoàn tất
        self.root.after(0, self.on_processing_finished)

    def add_result_to_list(self, index, result):
        """Thêm kết quả vào listbox"""
        file_name = result['file_name']
        
        # Thêm vào listbox
        self.file_listbox.insert(tk.END, file_name)
        
        # Nếu là ảnh đầu tiên, hiển thị ngay
        if index == 0:
            self.file_listbox.selection_set(0)
            self.display_result(0)

    def on_processing_finished(self):
        """Được gọi khi thread xử lý xong"""
        if self.processing_start_time:
            self.total_processing_time = time.time() - self.processing_start_time
            total_images = len(self.image_processing_times)
            
            print("\n" + "=" * 60)
            print(f"🎉 ĐÃ NHẬN DIỆN XONG {total_images} ẢNH!")
            print(f"⏱️ Tổng thời gian: {self.total_processing_time:.2f}s")
            print("=" * 60)
            
            self.status_label.config(text=f"Hoàn thành {total_images} ảnh!")
            self.total_time_label.config(text=f"Tổng thời gian: {self.total_processing_time:.2f}s")
        else:
            self.status_label.config(text="Đã hoàn thành!")
            
        self.btn_select.config(state="normal")
        self.btn_history.config(state="normal")

    def display_result(self, index):
        """Hiển thị kết quả của ảnh tại index"""
        if not self.results or index < 0 or index >= len(self.results):
            return
        
        self.current_index = index
        result = self.results[index]
        
        # Cập nhật selection trong listbox
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(index)
        self.file_listbox.see(index)
        
        # ========== Hiển thị ảnh bounding box (cột 1) ==========
        if result['processed_img']:
            # Resize để fit vào cột
            processed_img = self.resize_image_to_fit(result['processed_img'], 
                                                      max_width=600, max_height=500)
            tk_processed = ImageTk.PhotoImage(processed_img)
            self.image_refs.append(tk_processed)
            self.img_bbox_label.config(image=tk_processed)
        else:
            self.img_bbox_label.config(image='', text="Lỗi xử lý ảnh")
        
        # ========== Hiển thị ảnh biển số cắt (cột 2) ==========
        # Xóa các widget cũ trong plate_img_frame
        for widget in self.plate_img_frame.winfo_children():
            widget.destroy()
        
        # Lấy danh sách tất cả biển số cắt
        cropped_plates = result.get('cropped_plates', [])
        
        if cropped_plates and len(cropped_plates) > 0:
            num_plates = len(cropped_plates)
            
            # Tính số cột dựa trên số lượng biển số (tối đa 3 cột)
            if num_plates <= 2:
                num_cols = num_plates
            elif num_plates <= 4:
                num_cols = 2
            else:
                num_cols = 3
            
            # Tính kích thước ảnh dựa trên số cột
            max_width_per_plate = max(100, 300 // num_cols)
            max_height_per_plate = max(60, 120)
            
            # Tạo grid container
            grid_container = tk.Frame(self.plate_img_frame, bg="#FFFDE7")
            grid_container.pack(expand=True, fill="both")
            
            for i, plate_img in enumerate(cropped_plates):
                row = i // num_cols
                col = i % num_cols
                
                # Frame chứa từng biển số
                plate_container = tk.Frame(grid_container, bg="#FFFDE7", padx=5, pady=5)
                plate_container.grid(row=row, column=col, sticky="nsew")
                
                # Label số thứ tự
                label_text = f"#{i+1}"
                tk.Label(plate_container, text=label_text, 
                        font=("Arial", 9, "bold"), fg="#1565C0", bg="#FFFDE7").pack()
                
                # Hiển thị ảnh biển số
                cropped_img = self.resize_image_to_fit(plate_img, 
                                                        max_width=max_width_per_plate, 
                                                        max_height=max_height_per_plate)
                tk_cropped = ImageTk.PhotoImage(cropped_img)
                self.image_refs.append(tk_cropped)
                
                img_label = tk.Label(plate_container, image=tk_cropped, bg="#FFFDE7")
                img_label.pack()
            
            # Cấu hình grid để các cột có kích thước đều nhau
            for c in range(num_cols):
                grid_container.columnconfigure(c, weight=1)
        else:
            # Fallback: sử dụng cropped_plate cũ nếu không có cropped_plates
            if result.get('cropped_plate'):
                cropped_img = self.resize_image_to_fit(result['cropped_plate'], 
                                                        max_width=350, max_height=200)
                tk_cropped = ImageTk.PhotoImage(cropped_img)
                self.image_refs.append(tk_cropped)
                
                img_label = tk.Label(self.plate_img_frame, image=tk_cropped, bg="#FFFDE7")
                img_label.pack(expand=True)
            else:
                no_plate_label = tk.Label(self.plate_img_frame, text="Không có biển số", 
                                         font=("Arial", 12), fg="#666", bg="#FFFDE7")
                no_plate_label.pack(expand=True)
        
        # ========== Hiển thị text biển số ==========
        # Hiển thị đầy đủ thông tin: số thứ tự + loại xe + biển số
        if result['plates']:
            num_plates = len(result['plates'])
            # Điều chỉnh font size dựa trên số lượng biển số
            if num_plates <= 2:
                font_size = 30
            elif num_plates <= 4:
                font_size = 18
            else:
                font_size = 16
            
            plate_text = "\n".join(result['plates'])
            self.plate_text_label.config(text=plate_text, fg="#1565C0", 
                                        font=("Arial", font_size, "bold"))
        else:
            self.plate_text_label.config(text="Không nhận diện được", fg="#D32F2F",
                                        font=("Arial", 18, "bold"))
        
        # ========== Hiển thị thời gian xử lý ==========
        self.time_label.config(text=f"Thời gian xử lý: {result['processing_time']:.2f}s")
        
        # ========== Hiển thị tổng thời gian ==========
        if self.total_processing_time > 0:
            self.total_time_label.config(text=f"Tổng thời gian: {self.total_processing_time:.2f}s")

    def resize_image_to_fit(self, img_pil, max_width, max_height):
        """Resize ảnh để vừa với kích thước cho trước, giữ tỷ lệ"""
        width, height = img_pil.size
        
        # Tính tỷ lệ scale
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Không phóng to quá kích thước gốc
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def on_file_select(self, event):
        """Xử lý sự kiện chọn file trong listbox"""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            if index != self.current_index:
                self.display_result(index)

    def on_file_double_click(self, event):
        """Xử lý sự kiện double click vào file trong listbox"""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            if self.results and 0 <= index < len(self.results):
                self.open_image_external(self.results[index]['file_path'])

    def on_arrow_up(self, event):
        """Xử lý phím mũi tên lên"""
        if self.results:
            if self.current_index > 0:
                self.display_result(self.current_index - 1)
            else:
                # Đã ở ảnh đầu, quay lại ảnh cuối
                self.display_result(len(self.results) - 1)
            return "break"

    def on_arrow_down(self, event):
        """Xử lý phím mũi tên xuống"""
        if self.results:
            if self.current_index < len(self.results) - 1:
                self.display_result(self.current_index + 1)
            else:
                # Đã ở ảnh cuối, quay lại ảnh đầu
                self.display_result(0)
            return "break"

    def next_image(self):
        """Di chuyển đến ảnh tiếp theo"""
        if self.results:
            if self.current_index < len(self.results) - 1:
                self.display_result(self.current_index + 1)
            else:
                # Đã ở ảnh cuối, quay lại ảnh đầu
                self.display_result(0)

    def prev_image(self):
        """Di chuyển đến ảnh trước đó"""
        if self.results:
            if self.current_index > 0:
                self.display_result(self.current_index - 1)
            else:
                # Đã ở ảnh đầu, quay lại ảnh cuối
                self.display_result(len(self.results) - 1)

    def toggle_auto_play(self):
        """Bật/tắt chế độ tự động duyệt ảnh"""
        if self.auto_playing:
            self.stop_auto_play()
        else:
            self.start_auto_play()

    def start_auto_play(self):
        """Bắt đầu tự động duyệt ảnh"""
        if not self.results:
            return
        
        self.auto_playing = True
        self.btn_auto.config(text="Dừng", bg="#F44336")
        self.auto_play_next()

    def stop_auto_play(self):
        """Dừng tự động duyệt ảnh"""
        self.auto_playing = False
        self.btn_auto.config(text="Tự động", bg="#2196F3")
        if self.auto_play_id:
            self.root.after_cancel(self.auto_play_id)
            self.auto_play_id = None

    def auto_play_next(self):
        """Tự động chuyển sang ảnh tiếp theo"""
        if not self.auto_playing or not self.results:
            return
        
        if self.current_index < len(self.results) - 1:
            self.display_result(self.current_index + 1)
        else:
            # Đã hết ảnh, quay lại ảnh đầu tiên
            self.display_result(0)
        
        # Đợi 2 giây rồi chuyển tiếp
        self.auto_play_id = self.root.after(2000, self.auto_play_next)


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = MultiPlateApp(root)
    root.mainloop()