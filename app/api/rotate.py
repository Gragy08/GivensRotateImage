from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import io

# Import logic từ các file core
from app.core.rotate_2d import manual_givens_rotate
from app.core.rotate_3d import manual_rotation_3d
from app.core.rotate_nd import manual_rotation_nd

router = APIRouter()

@router.post("/rotate")
async def rotate_image(
    file: UploadFile = File(...), 
    mode: str = Form("2d"),             # Hứng mode từ frontend
    angle_x: float = Form(0), 
    angle_y: float = Form(0),
    angle_z: float = Form(0),
    n_dims: int = Form(4),              # Số chiều
    axis_i: int = Form(1),              # Trục i
    axis_j: int = Form(2),              # Trục j
    angle_nd: float = Form(0)           # Góc quay n-chiều
):
    # Load ảnh
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Chọn hàm xử lý dựa trên mode
    if mode == "2d":
        rotated_image = manual_givens_rotate(image, angle_z)
    elif mode == "3d":
        rotated_image = manual_rotation_3d(image, angle_x, angle_y, angle_z)
    elif mode == "nd":
        rotated_image = manual_rotation_nd(image, n_dims, axis_i, axis_j, angle_nd)
    else:
        # Fallback an toàn
        rotated_image = image 

    # Trả ảnh về client
    img_byte_arr = io.BytesIO()
    rotated_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")