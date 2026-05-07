from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import router từ file API
from app.api import rotate

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gắn router xoay ảnh vào app chính, tự động thêm tiền tố /api
app.include_router(rotate.router, prefix="/api")

@app.get("/")
async def read_root():
    # Trỏ đường dẫn đến thư mục static mới
    return FileResponse("app/static/index.html")