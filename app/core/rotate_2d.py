import numpy as np
from PIL import Image
import math

def manual_givens_rotate(image_pil: Image.Image, angle_degree: float):
    img_array = np.array(image_pil)
    height, width, channels = img_array.shape
    
    output_array = np.zeros_like(img_array)
    
    theta = math.radians(angle_degree)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    cx, cy = width // 2, height // 2

    y_idxs, x_idxs = np.indices((height, width))

    x_shifted = x_idxs - cx
    y_shifted = y_idxs - cy

    x_src = (x_shifted * cos_t + y_shifted * sin_t + cx).astype(int)
    y_src = (-x_shifted * sin_t + y_shifted * cos_t + cy).astype(int)

    mask = (x_src >= 0) & (x_src < width) & (y_src >= 0) & (y_src < height)

    output_array[y_idxs[mask], x_idxs[mask]] = img_array[y_src[mask], x_src[mask]]

    return Image.fromarray(output_array)