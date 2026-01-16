from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware 
from rotation_logic import manual_givens_rotate
from PIL import Image
import io

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/rotate")
async def rotate_image(
    file: UploadFile = File(...), 
    angle: float = Form(...)
):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    rotated_image = manual_givens_rotate(image, angle)

    img_byte_arr = io.BytesIO()
    rotated_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")