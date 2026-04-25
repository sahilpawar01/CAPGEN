"""
Image caption API: receive upload → BLIP → JSON caption.
"""
from __future__ import annotations

import io
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import openpyxl
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
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
    WEBCAM_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Webcam / caption file saves: {WEBCAM_CAPTURE_DIR}")
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

app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
    name="static",
)

APP_ROOT = Path(__file__).resolve().parent

MAX_BATCH = 100
MAX_IMAGE_BYTES = 15 * 1024 * 1024
# Always next to app.py, not the shell’s current directory when starting uvicorn
WEBCAM_CAPTURE_DIR = APP_ROOT / "webcam_captures"


def caption_image(pil: Image.Image) -> str:
    assert processor is not None and model is not None
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    inputs = processor(pil, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_length=50, num_beams=5)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def _wrap_caption_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = ""
    for w in words:
        next_part = w if not cur else f"{cur} {w}"
        if len(next_part) <= max_chars:
            cur = next_part
        else:
            if cur:
                lines.append(cur)
            cur = w[:max_chars] if len(w) > max_chars else w
    if cur:
        lines.append(cur)
    return lines or [""]


def _caption_overlay_font(size: int):
    for p in (
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ):
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_caption_on_image(pil: Image.Image, caption: str) -> Image.Image:
    """Tall image: original on top, gray band with caption at bottom."""
    w, h = pil.size
    if pil.mode != "RGB":
        base = pil.convert("RGB")
    else:
        base = pil.copy()
    line_max = max(20, w // 10)
    lines = _wrap_caption_text(caption.strip() or "—", line_max)
    font_size = max(12, min(20, w // 45))
    font = _caption_overlay_font(font_size)
    line_h = max(int(font_size * 1.25), 16)
    pad = line_h
    text_block_h = min(h // 2, line_h * len(lines) + pad * 2)
    out = Image.new("RGB", (w, h + text_block_h), (240, 240, 240))
    out.paste(base, (0, 0))
    draw = ImageDraw.Draw(out)
    y = h + pad
    fill = (20, 20, 20)
    for line in lines:
        draw.text((pad, y), line, fill=fill, font=font)
        y += line_h
    return out


def save_captioned_frame(pil: Image.Image, caption: str) -> dict[str, str]:
    """One JPEG per capture: picture with caption on the image, under webcam_captures/YYYY-MM-DD/."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = WEBCAM_CAPTURE_DIR / day
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now(timezone.utc).strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    path = d / stem
    labeled = render_caption_on_image(pil, caption)
    labeled.save(path, format="JPEG", quality=90)
    return {"image": f"{WEBCAM_CAPTURE_DIR.name}/{day}/{path.name}"}


@app.get("/")
async def root():
    return FileResponse(str(APP_ROOT / "static" / "index.html"))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "device": device, "model": MODEL_ID}


@app.post("/api/caption")
async def caption(
    file: UploadFile = File(...),
    save: Optional[str] = Form(None),
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file (e.g. PNG or JPEG).")
    raw = await file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {MAX_IMAGE_BYTES // (1024 * 1024)}MB).",
        )
    try:
        pil = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read image data.")
    try:
        text = caption_image(pil)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Caption failed: {e!s}") from e
    out: dict[str, Any] = {"caption": text}
    persist = str(save or "").lower() in ("1", "true", "yes", "on")
    if persist:
        try:
            out["saved"] = save_captioned_frame(pil, text)
        except Exception as e:
            out["save_error"] = f"Could not save image: {e!s}"
    return out


class CaptionRowIn(BaseModel):
    filename: str
    caption: str
    error: str = ""


class XlsxExportBody(BaseModel):
    rows: list[CaptionRowIn] = Field(min_length=1)


@app.post("/api/caption/batch")
async def caption_batch(
    files: list[UploadFile] = File(..., description="One or more image files"),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files (max {MAX_BATCH}).",
        )
    items: list[dict[str, str]] = []
    for file in files:
        name = file.filename or "image"
        if not file.content_type or not file.content_type.startswith("image/"):
            items.append(
                {
                    "filename": name,
                    "caption": "",
                    "error": "Not an image (use PNG, JPEG, WebP, etc.).",
                }
            )
            continue
        raw = await file.read()
        if len(raw) > MAX_IMAGE_BYTES:
            items.append(
                {
                    "filename": name,
                    "caption": "",
                    "error": f"File too large (max {MAX_IMAGE_BYTES // (1024 * 1024)} MB).",
                }
            )
            continue
        try:
            pil = Image.open(io.BytesIO(raw))
        except UnidentifiedImageError:
            items.append(
                {
                    "filename": name,
                    "caption": "",
                    "error": "Could not read image data.",
                }
            )
            continue
        try:
            text = caption_image(pil)
        except Exception as e:
            items.append(
                {
                    "filename": name,
                    "caption": "",
                    "error": f"Caption failed: {e!s}",
                }
            )
            continue
        items.append({"filename": name, "caption": text, "error": ""})
    return {"items": items}


@app.post("/api/export/xlsx")
async def export_xlsx(body: XlsxExportBody) -> StreamingResponse:
    wb = openpyxl.Workbook()
    ws = wb.active
    if ws is not None:
        ws.title = "Captions"
        ws.append(["Filename", "Caption", "Error"])
        for row in body.rows:
            ws.append([row.filename, row.caption, row.error])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="captions.xlsx"'},
    )
