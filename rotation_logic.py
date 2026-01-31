import numpy as np
from PIL import Image
import math

# Hàm 2D
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

    # --- PHẦN TOÁN HỌC ---
    
    # Tạo lưới tọa độ cho ảnh trả về
    # y_coords: ma trận chứa chỉ số dòng, x_coords: ma trận chứa chỉ số cột
    y_idxs, x_idxs = np.indices((height, width))

    # Dời gốc tọa độ về tâm
    x_shifted = x_idxs - cx
    y_shifted = y_idxs - cy

    # Áp dụng công thức Inverse Rotation
    # Tìm tọa độ source tương ứng với tọa độ đích
    # Công thức: P_src = R^(-1) * P_dest
    # Vì R là trực giao nên R^(-1) là R chuyển vị
    
    x_src = (x_shifted * cos_t + y_shifted * sin_t + cx).astype(int)
    y_src = (-x_shifted * sin_t + y_shifted * cos_t + cy).astype(int)

    # 4. Boundary Check - Chỉ lấy những điểm nằm trong ảnh gốc
    mask = (x_src >= 0) & (x_src < width) & (y_src >= 0) & (y_src < height)

    # 5. Mapping
    # output[y, x] = input[y_src, x_src] tại những vị trí hợp lệ
    # Nearest Neighbor method
    output_array[y_idxs[mask], x_idxs[mask]] = img_array[y_src[mask], x_src[mask]]

    # Chuyển lại thành ảnh để trả về
    return Image.fromarray(output_array)

# Hàm 3D
def manual_rotation_3d(image_pil: Image.Image, angle_x: float, angle_y: float, angle_z: float):
    img_array = np.array(image_pil)
    height, width, channels = img_array.shape
    
    # Tạo ảnh kết quả nền hoặc RGB
    output_array = np.zeros((height, width, channels), dtype=np.uint8)

    # Đổi sang độ radian
    ax = math.radians(angle_x)
    ay = math.radians(angle_y)
    az = math.radians(angle_z)

    # 1. Định nghĩa các Ma trận quay (3x3)
    # Ma trận quay quanh trục X
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(ax), -math.sin(ax)],
        [0, math.sin(ax), math.cos(ax)]
    ])
    
    # Ma trận quay quanh trục Y
    Ry = np.array([
        [math.cos(ay), 0, math.sin(ay)],
        [0, 1, 0],
        [-math.sin(ay), 0, math.cos(ay)]
    ])
    
    # Ma trận quay quanh trục Z
    Rz = np.array([
        [math.cos(az), -math.sin(az), 0],
        [math.sin(az), math.cos(az), 0],
        [0, 0, 1]
    ])

    # Ma trận quay tổng hợp: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    # 2. Tạo lưới tọa độ
    cx, cy = width / 2, height / 2
    f = 500  # Tiêu cự càng nhỏ độ hút càng lớn

    y_idxs, x_idxs = np.indices((height, width))
    
    # Làm phẳng mảng để tính toán vector hóa
    x_flat = x_idxs.flatten() - cx
    y_flat = y_idxs.flatten() - cy
    z_flat = np.zeros_like(x_flat) # Ảnh gốc phẳng nên z = 0

    # Gom thành ma trận tọa độ (3 dòng, N cột)
    coords = np.stack([x_flat, y_flat, z_flat])

    # 3. Áp dụng phép quay: P_new = R * P_old
    rotated_coords = R @ coords
    
    x_new = rotated_coords[0, :]
    y_new = rotated_coords[1, :]
    z_new = rotated_coords[2, :]

    # 4. Chiếu phối cảnh
    # Công thức: x_proj = x * f / (f + z)
    # Thêm 1e-5 để tránh chia cho 0
    scale = f / (f + z_new + 1e-5) 
    
    x_proj = (x_new * scale + cx).astype(int)
    y_proj = (y_new * scale + cy).astype(int)

    # 5. Forward Mapping
    # Lọc những điểm nằm trong khung hình
    valid_mask = (x_proj >= 0) & (x_proj < width) & (y_proj >= 0) & (y_proj < height)
    
    # Lấy tọa độ src và dest
    src_x = x_idxs.flatten()[valid_mask]
    src_y = y_idxs.flatten()[valid_mask]
    dst_x = x_proj[valid_mask]
    dst_y = y_proj[valid_mask]

    # Gán màu
    output_array[dst_y, dst_x] = img_array[src_y, src_x]

    return Image.fromarray(output_array)