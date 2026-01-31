from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from rotation_logic import manual_givens_rotate, manual_rotation_3d
from PIL import Image
import io

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return FileResponse("index.html")

# Call API
@app.post("/api/rotate")
async def rotate_image(
    file: UploadFile = File(...), 
    angle_x: float = Form(0), 
    angle_y: float = Form(0),
    angle_z: float = Form(0)
):
    # Load ảnh
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Chọn hàm xoay
    if angle_x == 0 and angle_y == 0:
        # Dùng hàm 2D 
        rotated_image = manual_givens_rotate(image, angle_z)
    else:
        # Dùng hàm 3D 
        rotated_image = manual_rotation_3d(image, angle_x, angle_y, angle_z)

    # Trả ảnh
    img_byte_arr = io.BytesIO()
    rotated_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")