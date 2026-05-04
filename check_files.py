import os
"""
Module: check_files.py
Purpose:
    Kiểm tra và liệt kê số lượng file trong các thư mục con của một thư mục chính.
Description:
    Script này duyệt qua tất cả các thư mục con trong đường dẫn được chỉ định
    và đếm số lượng file trong mỗi thư mục con, sau đó in ra kết quả.
Usage:
    Cập nhật folder_path với đường dẫn thư mục cần kiểm tra, sau đó chạy script.
Functions:
    - Xác minh sự tồn tại của thư mục chính
    - Lấy danh sách các thư mục con
    - Đếm số file trong mỗi thư mục con
    - Hiển thị kết quả thống kê
"""
from pathlib import Path

folder_path = r"D:\Master DS\Intro_to_DS\data_real"

if os.path.exists(folder_path):
    subfolders = [item for item in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, item))]
    
    print(f"Tổng số folder: {len(subfolders)}\n")
    
    for subfolder in subfolders:
        subfolder_full_path = os.path.join(folder_path, subfolder)
        file_count = len([item for item in os.listdir(subfolder_full_path) 
                         if os.path.isfile(os.path.join(subfolder_full_path, item))])
        print(f"{subfolder}: {file_count} file")
else:
    print("Folder không tồn tại")
    