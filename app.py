"""
Image caption API: receive upload → CAPGEN → JSON caption.
"""
from __future__ import annotations

import io
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import openpyxl
import torch
import tracker_vision
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BlipForConditionalGeneration, BlipProcessor

APP_ROOT = Path(__file__).resolve().parent
APP_VERSION = "4"
WEBCAM_CAPTURE_DIR = APP_ROOT / "webcam_captures"
ACTIVITY_ROOT = APP_ROOT / "activity_tracker" / "sessions"
MAX_BATCH = 100
MAX_IMAGE_BYTES = 15 * 1024 * 1024


def _env_truthy(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in ("1", "true", "yes", "on")


# Off by default: no tracker routes, no YOLO/FaceNet load. Set ENABLE_ACTIVITY_TRACKER=1 to restore.
ENABLE_ACTIVITY_TRACKER = _env_truthy("ENABLE_ACTIVITY_TRACKER", "0")
_TRACKER_DISABLED_DETAIL = (
    "Activity tracker is disabled. Set ENABLE_ACTIVITY_TRACKER=1 (or true) and restart the server to enable it."
)


def _ensure_activity_tracker_enabled() -> None:
    if not ENABLE_ACTIVITY_TRACKER:
        raise HTTPException(status_code=404, detail=_TRACKER_DISABLED_DETAIL)

MODEL_ID = "Salesforce/blip-image-captioning-base"
processor: Optional[BlipProcessor] = None
model: Optional[BlipForConditionalGeneration] = None
device: str = "cpu"
tracker_vision_init_error: Optional[str] = None
# Per session: last logged (face_match, activity) key — only append log / optional save when it changes
TRACKER_LAST_STATE_KEY: dict[str, str] = {}


def _tracker_state_key(face_match: bool, activity: str) -> str:
    return f"{int(face_match)}|{activity}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, device, tracker_vision_init_error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CAPGEN on {device} (first run may download weights)...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print("CAPGEN ready.")
    WEBCAM_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Webcam / caption file saves: {WEBCAM_CAPTURE_DIR}")
    if ENABLE_ACTIVITY_TRACKER:
        err = tracker_vision.init_models(device)
        if err:
            tracker_vision_init_error = err
            print(f"Tracker (YOLOv8 + FaceNet) not loaded: {err}")
        else:
            tracker_vision_init_error = None
            print("Tracker YOLOv8n + FaceNet (MTCNN) ready.")
        ACTIVITY_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"Activity tracker sessions: {ACTIVITY_ROOT}")
    else:
        tracker_vision_init_error = None
        print("Activity tracker disabled — set ENABLE_ACTIVITY_TRACKER=1 to load YOLO/FaceNet and expose /tracker.")
    print(f"CAPGEN app version {APP_VERSION}")
    yield
    model = None
    processor = None


app = FastAPI(title=f"CAPGEN Caption API v{APP_VERSION}", version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static is mounted at the end of this file (after all routes) so it cannot shadow
# app routes on some ASGI servers.

# CAPGEN caption generation (short, beam search, anti-repetition); tuned for webcam latency.
_CAPGEN_MAX_NEW_TOKENS = 25
_CAPGEN_NUM_BEAMS = 5
_CAPGEN_REPETITION_PENALTY = 1.4


def _dedupe_consecutive_words(text: str) -> str:
    """Remove immediate duplicate tokens: e.g. 'self self self' -> 'self'. O(n) in word count."""
    parts = text.split()
    if len(parts) <= 1:
        return text
    out: list[str] = [parts[0]]
    for w in parts[1:]:
        if w != out[-1]:
            out.append(w)
    return " ".join(out)


def _clean_capgen_caption(raw: str) -> str:
    """Normalize decoded CAPGEN text: strip, lowercase, collapse consecutive repeated words."""
    return _dedupe_consecutive_words(raw.strip().lower())


def caption_image(pil: Image.Image) -> str:
    assert processor is not None and model is not None
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    # Image-only conditioning — no text prompt (conditional CAPGEN encoding).
    inputs = processor(pil, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=_CAPGEN_MAX_NEW_TOKENS,
            num_beams=_CAPGEN_NUM_BEAMS,
            repetition_penalty=_CAPGEN_REPETITION_PENALTY,
            early_stopping=True,
        )
    raw = processor.decode(out[0], skip_special_tokens=True)
    return _clean_capgen_caption(raw)


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


def append_activity_log(
    session_id: str, line: str, face_match: bool, sim: float, log_image: str | None
) -> None:
    root = ACTIVITY_ROOT / session_id
    if not root.is_dir():
        return
    p = root / "activity_log.txt"
    ts = datetime.now(timezone.utc).isoformat()
    extra = f" | match={face_match} | sim={sim:.3f}"
    if log_image:
        extra += f" | image={log_image}"
    with p.open("a", encoding="utf-8") as f:
        f.write(f"{ts} | {line}{extra}\n")


def save_tracker_captioned(pil: Image.Image, line: str, session_id: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = ACTIVITY_ROOT / session_id / day
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now(timezone.utc).strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    path = d / stem
    labeled = render_caption_on_image(pil, line)
    labeled.save(path, format="JPEG", quality=90)
    return f"activity_tracker/sessions/{session_id}/{day}/{stem}"


def _tracker_file_response() -> FileResponse:
    _ensure_activity_tracker_enabled()
    p = APP_ROOT / "static" / "tracker.html"
    if not p.is_file():
        raise HTTPException(
            status_code=500,
            detail="Missing static/tracker.html — copy the project static folder next to app.py.",
        )
    return FileResponse(p, media_type="text/html")


def _tracker_js_file_response() -> FileResponse:
    _ensure_activity_tracker_enabled()
    p = APP_ROOT / "static" / "tracker.js"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="tracker.js not found.")
    return FileResponse(p, media_type="application/javascript")


@app.get("/")
async def root():
    return FileResponse(str(APP_ROOT / "static" / "index.html"))


@app.get("/analyze", tags=["pages"])
async def analyze_page() -> FileResponse:
    p = APP_ROOT / "static" / "analyze.html"
    if not p.is_file():
        raise HTTPException(
            status_code=500,
            detail="Missing static/analyze.html — add the file next to the other static pages.",
        )
    return FileResponse(p, media_type="text/html")


@app.get("/analyze/", include_in_schema=False)
async def analyze_page_trailing_slash() -> RedirectResponse:
    """Same page as /analyze; browsers and links sometimes add a trailing slash."""
    return RedirectResponse(url="/analyze", status_code=307)


@app.get("/tracker", tags=["pages"])
async def tracker_page() -> FileResponse:
    return _tracker_file_response()


@app.get("/activity", tags=["pages"], include_in_schema=False)
async def tracker_page_alias() -> FileResponse:
    return _tracker_file_response()


@app.get("/static/tracker.html", include_in_schema=False)
async def static_tracker_html() -> FileResponse:
    """Shadow StaticFiles so /static/tracker.html respects ENABLE_ACTIVITY_TRACKER."""
    return _tracker_file_response()


@app.get("/static/tracker.js", include_in_schema=False)
async def static_tracker_js() -> FileResponse:
    return _tracker_js_file_response()


@app.get(
    "/api/tracker",
    tags=["tracker"],
    summary="Tracker API index (this is not the HTML page)",
)
async def tracker_api_index() -> dict[str, str]:
    """Clarify /api/tracker vs the browser page at /tracker (common source of 404)."""
    _ensure_activity_tracker_enabled()
    return {
        "open_in_browser": "/tracker",
        "alternate_path": "/activity",
        "enroll": "POST /api/tracker/enroll",
        "analyze": "POST /api/tracker/analyze",
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": APP_VERSION,
        "device": device,
        "model": MODEL_ID,
        "activity_tracker_enabled": ENABLE_ACTIVITY_TRACKER,
        "tracker_vision": tracker_vision.is_ready(),
        "tracker_vision_error": tracker_vision_init_error,
    }


@app.post("/api/tracker/enroll")
async def tracker_enroll(
    name: str = Form(..., min_length=1, max_length=256),
    # Legacy: single part named "file". Current UI sends repeated "files".
    file: Optional[UploadFile] = File(default=None),
    files: list[UploadFile] = File(),
) -> dict[str, Any]:
    _ensure_activity_tracker_enabled()
    if not tracker_vision.is_ready() or tracker_vision_init_error:
        raise HTTPException(
            status_code=503,
            detail="Tracker (YOLO + FaceNet) is not available. Check server logs.",
        )
    name = (name or "").strip() or "Person"
    if files:
        files_in = list(files)
    else:
        files_in = [file] if file is not None else []
    if not files_in:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one face photo; use the same control multiple times (front + side) for best matching.",
        )
    pils: list[Image.Image] = []
    raws: list[bytes] = []
    for file in files_in:
        if not file.content_type or not str(file.content_type).startswith("image/"):
            raise HTTPException(status_code=400, detail="All files must be images (PNG, JPEG, WebP, …).")
        raw = await file.read()
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="One image is too large.")
        try:
            pil = Image.open(io.BytesIO(raw))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Could not read one of the images.")
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pils.append(pil)
        raws.append(raw)
    mat, per_errs = tracker_vision.enroll_many_embeddings(pils)
    if mat is None or mat.size == 0:
        raise HTTPException(
            status_code=400,
            detail="No face found in any image. Add clear face shots (lighting, forward or slight profile).",
        )
    session_id = uuid.uuid4().hex
    root = ACTIVITY_ROOT / session_id
    root.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(raws):
        (root / f"ref_{i}.jpg").write_bytes(b)
    np.save(str(root / "embeddings.npy"), mat)
    n_ref = int(mat.shape[0])
    photo_warnings: list[str] = []
    for i, e in enumerate(per_errs):
        if e:
            photo_warnings.append(f"Photo {i + 1} skipped: {e}")
    (root / "meta.json").write_text(
        json.dumps(
            {
                "name": name or "Person",
                "session_id": session_id,
                "refs_stored": n_ref,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    msg = (
        f"Enrolled {n_ref} face embedding(s) from {len(pils)} photo(s). "
        "Matching picks the best score across your references, which helps when you turn your head. "
        "For reliable side/angle recognition, add a second or third reference photo in profile. "
        "You can set TRACKER_FACE_THRESHOLD in the environment (default ~0.54; lower = looser match)."
    )
    if photo_warnings:
        msg = msg + " " + " ".join(photo_warnings)
    return {
        "session_id": session_id,
        "name": name or "Person",
        "refs_stored": n_ref,
        "photo_warnings": photo_warnings,
        "face_match_threshold": tracker_vision.get_face_match_threshold(),
        "message": msg,
    }


@app.post("/api/tracker/analyze")
async def tracker_analyze(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    save: Optional[str] = Form(None),
) -> dict[str, Any]:
    _ensure_activity_tracker_enabled()
    if not tracker_vision.is_ready() or tracker_vision_init_error:
        raise HTTPException(status_code=503, detail="Tracker models unavailable.")
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required (enroll first).")
    root = ACTIVITY_ROOT / sid
    emb_p = root / "embeddings.npy"
    leg_p = root / "embedding.npy"
    if not (root / "meta.json").is_file() or (not emb_p.is_file() and not leg_p.is_file()):
        raise HTTPException(status_code=404, detail="Unknown session. Enroll again.")
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    display = str(meta.get("name", "Person"))
    ref = np.load(str(emb_p if emb_p.is_file() else leg_p))
    ref = np.atleast_2d(np.asarray(ref, dtype=np.float32))
    if not file.content_type or not str(file.content_type).startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a frame image.")
    raw = await file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large.")
    try:
        pil = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the image.")
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    r = tracker_vision.analyze_frame_pil(pil, ref, display)
    state_key = _tracker_state_key(r.face_match, r.activity)
    prev = TRACKER_LAST_STATE_KEY.get(sid)
    changed = prev != state_key
    if changed:
        TRACKER_LAST_STATE_KEY[sid] = state_key
    out: dict[str, Any] = {
        "session_id": sid,
        "name": r.name,
        "line": r.line,
        "activity": r.activity,
        "face_match": r.face_match,
        "face_score": r.face_score,
        "detections": r.detections,
        "yolo": r.yolo_debug,
        "activity_state_changed": changed,
        "skipped_duplicate_log": not changed,
    }
    if not changed:
        return out
    saved_rel: str | None = None
    if str(save or "").lower() in ("1", "true", "yes", "on"):
        try:
            saved_rel = save_tracker_captioned(pil, r.line, sid)
        except OSError as e:
            out["save_error"] = str(e)
    append_activity_log(sid, r.line, r.face_match, r.face_score, saved_rel)
    if saved_rel:
        out["saved"] = {"image": saved_rel}
    return out


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


_CAPTION_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)

# Short English stopword list (caption-scale text; keeps charts readable).
_CAPTION_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "with",
        "by",
        "from",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "not",
        "no",
    }
)

_sia: Optional[SentimentIntensityAnalyzer] = None


def _sentiment_analyzer() -> SentimentIntensityAnalyzer:
    global _sia
    if _sia is None:
        _sia = SentimentIntensityAnalyzer()
    return _sia


def _tokenize_caption(s: str) -> list[str]:
    return [m.group(0).lower() for m in _CAPTION_TOKEN_RE.finditer(s) if m.group(0)]


def _sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _content_tokens(toks: list[str]) -> list[str]:
    return [w for w in toks if w not in _CAPTION_STOPWORDS and len(w) >= 2]


def analyze_caption_texts(rows: list[CaptionRowIn]) -> dict[str, Any]:
    """Aggregate stats, VADER sentiment, and dataset-quality signals for business / research use."""
    n_total = len(rows)
    n_failed = sum(1 for r in rows if (r.error or "").strip())
    usable: list[tuple[str, str]] = [
        (r.filename, r.caption.strip()) for r in rows if (r.caption or "").strip() and not (r.error or "").strip()
    ]
    n_captions = len(usable)
    if n_captions == 0:
        return {
            "n_rows_total": n_total,
            "n_captions": 0,
            "n_failed": n_failed,
            "message": "No successful captions to analyze. Fix upload errors and run Get captions again.",
        }

    word_counter: Counter[str] = Counter()
    bigram_counter: Counter[tuple[str, str]] = Counter()
    word_counts: list[int] = []
    char_counts: list[int] = []
    per: list[dict[str, Any]] = []
    sia = _sentiment_analyzer()
    for fn, text in usable:
        toks = _tokenize_caption(text)
        wc = len(toks) if toks else (len(text.split()) or 1)
        word_counts.append(wc)
        char_counts.append(len(text))
        content = _content_tokens(toks)
        for w in content:
            word_counter[w] += 1
        for a, b in zip(content, content[1:]):
            bigram_counter[(a, b)] += 1
        vs = sia.polarity_scores(text)
        c = float(vs["compound"])
        per.append(
            {
                "filename": fn,
                "caption": text,
                "word_count": wc,
                "char_count": char_counts[-1],
                "compound": round(c, 4),
                "pos": round(float(vs["pos"]), 4),
                "neu": round(float(vs["neu"]), 4),
                "neg": round(float(vs["neg"]), 4),
                "label": _sentiment_label(c),
            }
        )

    top = [{"word": w, "count": c} for w, c in word_counter.most_common(25)]
    top_bigrams = [
        {"w1": a, "w2": b, "count": c} for (a, b), c in bigram_counter.most_common(20)
    ]

    content_token_total = int(sum(word_counter.values()))
    unique_content = int(len(word_counter))
    ttr = (unique_content / content_token_total) if content_token_total else 0.0

    bins = {"1–3": 0, "4–6": 0, "7–9": 0, "10+": 0}
    for wc in word_counts:
        if wc <= 3:
            bins["1–3"] += 1
        elif wc <= 6:
            bins["4–6"] += 1
        elif wc <= 9:
            bins["7–9"] += 1
        else:
            bins["10+"] += 1

    labels: dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
    for p in per:
        lab = str(p["label"])
        if lab in labels:
            labels[lab] += 1
    mean_compound = float(np.mean([p["compound"] for p in per])) if per else 0.0
    mean_pos = float(np.mean([p["pos"] for p in per])) if per else 0.0
    mean_neu = float(np.mean([p["neu"] for p in per])) if per else 0.0
    mean_neg = float(np.mean([p["neg"] for p in per])) if per else 0.0

    w_arr = np.array(word_counts, dtype=np.float64)
    c_arr = np.array(char_counts, dtype=np.float64)
    i_max = int(np.argmax(w_arr)) if n_captions else 0
    i_min = int(np.argmin(w_arr)) if n_captions else 0

    by_line: dict[str, list[str]] = defaultdict(list)
    for fn, text in usable:
        key = " ".join(text.lower().split())
        by_line[key].append(fn)
    dup_groups = [{"caption": k, "count": len(v), "files": v} for k, v in by_line.items() if len(v) > 1]
    n_dup_lines = len(dup_groups)
    n_dup_captions = sum(g["count"] for g in dup_groups)

    success_rate = ((n_total - n_failed) / n_total * 100.0) if n_total else 0.0
    insights: list[dict[str, str]] = []
    if n_failed:
        insights.append(
            {
                "type": "ingestion",
                "title": "Ingestion or decode failures",
                "text": f"{n_failed} row(s) have errors. For ML pipelines, filter or re-run failed assets so metrics are not biased.",
            }
        )
    if n_dup_lines and n_captions > 1:
        insights.append(
            {
                "type": "diversity",
                "title": "Exact duplicate captions in this batch",
                "text": f"{n_dup_lines} distinct caption(s) are repeated across files ({n_dup_captions} files involved). May indicate duplicate images or a narrow scene — re-balance the dataset for generalization.",
            }
        )
    if n_captions >= 5 and ttr < 0.25 and content_token_total >= 20:
        insights.append(
            {
                "type": "vocabulary",
                "title": "Narrow content vocabulary (type–token ratio)",
                "text": f"Type–token ratio is {ttr:.2f}. A low value suggests repetitive wording across captions; consider more diverse source images for robust training or contrastive learning.",
            }
        )
    if n_captions > 0:
        mx = max(labels.values())
        dom = [k for k, v in labels.items() if v == mx][0] if labels else "neutral"
        if mx / n_captions >= 0.75 and dom in ("positive", "negative"):
            insights.append(
                {
                    "type": "sentiment",
                    "title": "VADER text sentiment is skewed",
                    "text": f"About {100 * mx / n_captions:.0f}% of lines are {dom} in tone (word-level prior only, not image labels). If you need affect balance, downsample or augment for neutrality.",
                }
            )
    wc_std = float(np.std(w_arr)) if n_captions > 1 else 0.0
    if n_captions >= 3 and float(np.mean(w_arr)) < 4.0:
        insights.append(
            {
                "type": "length",
                "title": "Very short average captions",
                "text": "Average word count is low. For caption-to-image or retrieval, very short text may under-describe; consider re-captioning, beam search, or a larger decoder budget.",
            }
        )
    if n_captions >= 5 and wc_std > 4.0:
        insights.append(
            {
                "type": "variance",
                "title": "High length spread",
                "text": f"Word count standard deviation is {wc_std:.1f}. Mix of very short and long captions can help stress-test length limits and tokenization in production.",
            }
        )
    if not insights:
        insights.append(
            {
                "type": "ok",
                "title": "No major red flags from these heuristics",
                "text": "Use charts and per-row scores below for deeper checks before labeling, fine-tuning, or deployment.",
            }
        )

    return {
        "n_rows_total": n_total,
        "n_captions": n_captions,
        "n_failed": n_failed,
        "batch_quality": {
            "success_rate_pct": round(success_rate, 1),
            "failed_rows": n_failed,
            "successful_captions": n_captions,
        },
        "vocabulary": {
            "unique_content_words": unique_content,
            "total_content_tokens": content_token_total,
            "type_token_ratio": round(ttr, 4),
        },
        "duplicates": {
            "n_duplicate_lines": n_dup_lines,
            "n_files_in_duplicate_captions": n_dup_captions,
            "groups": sorted(dup_groups, key=lambda g: g["count"], reverse=True)[:20],
        },
        "outliers": {
            "longest": {
                "filename": per[i_max]["filename"],
                "caption": per[i_max]["caption"],
                "word_count": int(w_arr[i_max]),
            },
            "shortest": {
                "filename": per[i_min]["filename"],
                "caption": per[i_min]["caption"],
                "word_count": int(w_arr[i_min]),
            },
        },
        "word_count_stats": {
            "min": int(w_arr.min()),
            "max": int(w_arr.max()),
            "mean": round(float(w_arr.mean()), 2),
            "stdev": round(wc_std, 2) if n_captions > 1 else 0.0,
            "p25": round(float(np.percentile(w_arr, 25)), 1) if n_captions else 0.0,
            "p75": round(float(np.percentile(w_arr, 75)), 1) if n_captions else 0.0,
        },
        "char_count_stats": {
            "min": int(c_arr.min()),
            "max": int(c_arr.max()),
            "stdev": round(float(np.std(c_arr)), 2) if n_captions > 1 else 0.0,
        },
        "top_words": top,
        "top_bigrams": top_bigrams,
        "word_count_bins": bins,
        "caption_length": {
            "words_mean": round(float(np.mean(word_counts)), 2) if word_counts else 0.0,
            "words_median": float(np.median(word_counts)) if word_counts else 0.0,
            "chars_mean": round(float(np.mean(char_counts)), 1) if char_counts else 0.0,
        },
        "per_caption": per,
        "sentiment_summary": {
            "mean_compound": round(mean_compound, 4),
            "mean_pos": round(mean_pos, 4),
            "mean_neu": round(mean_neu, 4),
            "mean_neg": round(mean_neg, 4),
            "by_label": labels,
        },
        "insights": insights,
    }


class CaptionAnalyzeBody(BaseModel):
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


@app.post("/api/caption/analyze")
async def caption_analyze(body: CaptionAnalyzeBody) -> dict[str, Any]:
    """
    Data-science style stats on a batch: word frequencies, length distribution, VADER sentiment.
    """
    return analyze_caption_texts(body.rows)


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


app.mount(
    "/static",
    StaticFiles(directory=str(APP_ROOT / "static")),
    name="static",
)
