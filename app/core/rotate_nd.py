import numpy as np
from PIL import Image
import math

def manual_rotation_nd(image_pil: Image.Image, n_dims: int, axis_i: int, axis_j: int, angle_degree: float):
    img_array = np.array(image_pil)
    height, width, channels = img_array.shape

    # Tạo ảnh nền kết quả
    output_array = np.zeros((height, width, channels), dtype=np.uint8)

    # Chuyển góc sang radian
    theta = math.radians(angle_degree)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # 1. Khởi tạo ma trận đơn vị n x n
    G = np.eye(n_dims)

    # Chuyển đổi Trục i, j từ giao diện (bắt đầu từ 1) sang index mảng của NumPy (bắt đầu từ 0)
    idx_i = axis_i - 1
    idx_j = axis_j - 1

    # 2. Cập nhật 4 phần tử của ma trận Givens tại mặt phẳng (i, j)
    G[idx_i, idx_i] = cos_t
    G[idx_i, idx_j] = -sin_t
    G[idx_j, idx_i] = sin_t
    G[idx_j, idx_j] = cos_t

    # 3. Tạo lưới tọa độ 2D của ảnh gốc
    cx, cy = width / 2, height / 2
    y_idxs, x_idxs = np.indices((height, width))
    
    # Làm phẳng (flatten) và dời tâm
    x_flat = x_idxs.flatten() - cx
    y_flat = y_idxs.flatten() - cy

    # 4. Nhúng vào không gian n-chiều
    N_pixels = len(x_flat)
    # Tạo ma trận tọa độ [n_dims, N_pixels] toàn số 0
    coords = np.zeros((n_dims, N_pixels))
    
    # Gán tọa độ X (chiều 0) và Y (chiều 1). Các chiều h > 1 vẫn là 0.
    coords[0, :] = x_flat
    coords[1, :] = y_flat

    # 5. Thực hiện phép quay: P_new = G * P_old
    rotated_coords = G @ coords

    # 6. Chiếu trực giao xuống 2D (Chỉ lấy lại trục 0 và 1)
    x_new = rotated_coords[0, :]
    y_new = rotated_coords[1, :]

    # Dời tọa độ về lại vị trí góc trái trên cùng
    x_proj = (x_new + cx).astype(int)
    y_proj = (y_new + cy).astype(int)

    # 7. Forward Mapping (Gắn màu từ ảnh gốc sang ảnh đích)
    valid_mask = (x_proj >= 0) & (x_proj < width) & (y_proj >= 0) & (y_proj < height)
    
    src_x = x_idxs.flatten()[valid_mask]
    src_y = y_idxs.flatten()[valid_mask]
    dst_x = x_proj[valid_mask]
    dst_y = y_proj[valid_mask]

    output_array[dst_y, dst_x] = img_array[src_y, src_x]

    return Image.fromarray(output_array)