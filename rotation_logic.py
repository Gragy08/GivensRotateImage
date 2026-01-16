import numpy as np
from PIL import Image
import math

def manual_givens_rotate(image_pil: Image.Image, angle_degree: float):
    # 1. Chuyển ảnh sang ma trận NumPy (Hàng, Cột, Màu)
    img_array = np.array(image_pil)
    height, width, channels = img_array.shape
    
    # 2. Tạo ảnh kết quả (canvas) màu đen
    # Có thể tính toán kích thước mới để không bị cắt góc, nhưng để đơn giản ta giữ nguyên size
    output_array = np.zeros_like(img_array)
    
    # 3. Chuẩn bị các tham số quay
    theta = math.radians(angle_degree)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    # Tâm xoay (thường là giữa ảnh)
    cx, cy = width // 2, height // 2

    # --- ĐÂY LÀ PHẦN TOÁN HỌC BẠN CẦN TỰ VIẾT LẠI/GIẢI THÍCH ---
    
    # Tạo lưới tọa độ cho ảnh đích (Destination Grid)
    # y_coords: ma trận chứa chỉ số dòng, x_coords: ma trận chứa chỉ số cột
    y_idxs, x_idxs = np.indices((height, width))

    # Dời gốc tọa độ về tâm
    x_shifted = x_idxs - cx
    y_shifted = y_idxs - cy

    # Áp dụng công thức QUAY NGƯỢC (Inverse Rotation)
    # Tìm tọa độ gốc (source) tương ứng với tọa độ đích
    # Công thức: P_src = R^(-1) * P_dest
    # Vì R là trực giao nên R^(-1) là R chuyển vị (đổi dấu sin)
    
    x_src = (x_shifted * cos_t + y_shifted * sin_t + cx).astype(int)
    y_src = (-x_shifted * sin_t + y_shifted * cos_t + cy).astype(int)

    # 4. Kiểm tra biên (Boundary Check) - Chỉ lấy những điểm nằm trong ảnh gốc
    mask = (x_src >= 0) & (x_src < width) & (y_src >= 0) & (y_src < height)

    # 5. Gán giá trị màu (Mapping)
    # output[y, x] = input[y_src, x_src] tại những vị trí hợp lệ
    # Đây là phương pháp Nearest Neighbor (láng giềng gần nhất)
    output_array[y_idxs[mask], x_idxs[mask]] = img_array[y_src[mask], x_src[mask]]

    # Chuyển lại thành ảnh để trả về
    return Image.fromarray(output_array)