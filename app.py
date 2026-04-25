"""
Image caption API: receive upload → BLIP → JSON caption.
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Any, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_ID = "Salesforce/blip-image-captioning-base"
processor: Optional[BlipProcessor] = None
model: Optional[BlipForConditionalGeneration] = None
device: str = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BLIP on {device} (first run may download weights)...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print("BLIP ready.")
    yield
    model = None
    processor = None


app = FastAPI(title="BLIP Caption API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


def caption_image(pil: Image.Image) -> str:
    assert processor is not None and model is not None
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    inputs = processor(pil, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_length=50, num_beams=5)
    return processor.decode(out[0], skip_special_tokens=True).strip()


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "device": device, "model": MODEL_ID}


@app.post("/api/caption")
async def caption(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file (e.g. PNG or JPEG).")
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 15MB).")
    try:
        pil = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read image data.")
    try:
        text = caption_image(pil)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Caption failed: {e!s}") from e
    return {"caption": text}
