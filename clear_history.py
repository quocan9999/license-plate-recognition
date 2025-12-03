import os
import shutil
from modules.config import HISTORY_DIR

def clear_history():
    """
    Xóa toàn bộ dữ liệu trong thư mục History
    """
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(HISTORY_DIR):
        print(f"Thư mục '{HISTORY_DIR}' không tồn tại. Không có gì để xóa.")
        return

    print(f"Đang xóa dữ liệu trong thư mục '{HISTORY_DIR}'...")
    
    # Đếm số lượng file/folder đã xóa
    deleted_files = 0
    deleted_dirs = 0
    
    try:
        # Duyệt qua tất cả các item trong thư mục
        for item in os.listdir(HISTORY_DIR):
            item_path = os.path.join(HISTORY_DIR, item)
            
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path) # Xóa file hoặc symbolic link
                    deleted_files += 1
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path) # Xóa thư mục và nội dung bên trong
                    deleted_dirs += 1
            except Exception as e:
                print(f"Không thể xóa {item_path}. Lỗi: {e}")

        print("--------------------------------------------------")
        print(f"✅ Đã xóa hoàn tất!")
        print(f"   - {deleted_dirs} thư mục con")
        print(f"   - {deleted_files} files (bao gồm cả CSV)")
        print("--------------------------------------------------")
        
    except Exception as e:
        print(f"Đã xảy ra lỗi chung: {e}")

if __name__ == "__main__":
    confirm = input(f"⚠️  CẢNH BÁO: Bạn có chắc chắn muốn xóa TOÀN BỘ dữ liệu trong '{HISTORY_DIR}' không? (y/n): ")
    if confirm.lower() == 'y':
        clear_history()
    else:
        print("Đã hủy thao tác.")
