import numpy as np
from PIL import Image
import math

def manual_rotation_3d(image_pil: Image.Image, angle_x: float, angle_y: float, angle_z: float):
    img_array = np.array(image_pil)
    height, width, channels = img_array.shape
    
    output_array = np.zeros((height, width, channels), dtype=np.uint8)

    ax = math.radians(angle_x)
    ay = math.radians(angle_y)
    az = math.radians(angle_z)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(ax), -math.sin(ax)],
        [0, math.sin(ax), math.cos(ax)]
    ])
    
    Ry = np.array([
        [math.cos(ay), 0, math.sin(ay)],
        [0, 1, 0],
        [-math.sin(ay), 0, math.cos(ay)]
    ])
    
    Rz = np.array([
        [math.cos(az), -math.sin(az), 0],
        [math.sin(az), math.cos(az), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    cx, cy = width / 2, height / 2
    f = 500  

    y_idxs, x_idxs = np.indices((height, width))
    
    x_flat = x_idxs.flatten() - cx
    y_flat = y_idxs.flatten() - cy
    z_flat = np.zeros_like(x_flat) 

    coords = np.stack([x_flat, y_flat, z_flat])

    rotated_coords = R @ coords
    
    x_new = rotated_coords[0, :]
    y_new = rotated_coords[1, :]
    z_new = rotated_coords[2, :]

    scale = f / (f + z_new + 1e-5) 
    
    x_proj = (x_new * scale + cx).astype(int)
    y_proj = (y_new * scale + cy).astype(int)

    valid_mask = (x_proj >= 0) & (x_proj < width) & (y_proj >= 0) & (y_proj < height)
    
    src_x = x_idxs.flatten()[valid_mask]
    src_y = y_idxs.flatten()[valid_mask]
    dst_x = x_proj[valid_mask]
    dst_y = y_proj[valid_mask]

    output_array[dst_y, dst_x] = img_array[src_y, src_x]

    return Image.fromarray(output_array)