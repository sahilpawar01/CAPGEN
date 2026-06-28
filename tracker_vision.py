"""
YOLOv8 (objects) + FaceNet (MTCNN + InceptionResnetV1) for person-linked activity heuristics.
This is a practical MVP, not full DeepSORT multi-object tracking in crowds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

# COCO class names in YOLOv8 use these strings
PHONE = "cell phone"
LAPTOP = "laptop"
BOTTLE = "bottle"
CUP = "cup"
WINE = "wine glass"
BOOK = "book"
PERSON = "person"
# Extra COCO (80-class) — used for wider activity heuristics (objects must link to a person)
MOUSE = "mouse"
TV = "tv"
REMOTE = "remote"
# "Writing" is not a COCO class; laptop/mouse/keyboard-free typing = "at a computer" as best effort.
# "Talking" is not a COCO class; we only infer "with another person" if two people are nearby.

DRINK: frozenset[str] = frozenset({BOTTLE, CUP, WINE})

# Solid food / tableware (bowl = soup, cereal, etc.)
EAT: frozenset[str] = frozenset(
    {
        "bowl",
        "fork",
        "knife",
        "spoon",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
    }
)

SCREEN: frozenset[str] = frozenset({TV, REMOTE})
COMPUTER: frozenset[str] = frozenset({LAPTOP, MOUSE})

SPORTS: frozenset[str] = frozenset(
    {
        "sports ball",
        "frisbee",
        "tennis racket",
        "skateboard",
        "baseball bat",
        "baseball glove",
        "kite",
        "surfboard",
    }
)

_yolo: Any = None
_mtcnn: Any = None
_resnet: Any = None
_device: str = "cpu"

# Slightly looser for profile / off-angle when combined with multi-reference enrollment.
# Override: set env TRACKER_FACE_THRESHOLD=0.5 (lower = more matches, more false positives)
def get_face_match_threshold() -> float:
    return float(os.environ.get("TRACKER_FACE_THRESHOLD", "0.54"))


def init_models(device: str) -> str | None:
    """Load YOLO + face nets. Returns error message on failure, else None."""
    global _yolo, _mtcnn, _resnet, _device
    _device = device
    try:
        from facenet_pytorch import InceptionResnetV1, MTCNN
        from ultralytics import YOLO
    except ImportError as e:
        return (
            f"Tracker import failed: {e}. "
            "Install: pip install ultralytics facenet-pytorch"
        )
    try:
        _yolo = YOLO("yolov8n.pt")
        # Smaller min_face, margins, and looser MTCNN thresholds to catch more partial / side faces.
        _mtcnn = MTCNN(
            image_size=160,
            margin=12,
            min_face_size=16,
            thresholds=[0.5, 0.5, 0.5],
            factor=0.709,
            post_process=True,
            device=device,
        )
        _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    except Exception as e:
        return str(e)
    return None


def is_ready() -> bool:
    return _yolo is not None and _mtcnn is not None and _resnet is not None


def _face_embed(pil: Image.Image) -> np.ndarray | None:
    if _mtcnn is None or _resnet is None:
        return None
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    t = _mtcnn(pil)
    if t is None:
        return None
    if t.dim() == 3:
        t = t.unsqueeze(0)
    t = t.to(_device)
    with torch.inference_mode():
        emb = _resnet(t)
    e = emb.cpu().numpy().flatten()
    n = float(np.linalg.norm(e)) + 1e-6
    return (e / n).astype(np.float32)


def yolo_objects(pil: Image.Image) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """Return list of (class_name, conf, (x1,y1,x2,y2)) xyxy in pixel coords."""
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if _yolo is None:
        return []
    # Slightly higher input for small / distant cups, phones, and bottles (tunable via env).
    y_imgsz = int(os.environ.get("TRACKER_YOLO_IMGSZ", "800"))
    y_imgsz = max(512, min(1280, y_imgsz))
    res = _yolo(pil, verbose=False, imgsz=y_imgsz)[0]
    out: list[tuple[str, float, tuple[float, float, float, float]]] = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    nm = _yolo.names
    for b in res.boxes:
        cls = int(b.cls[0].item()) if b.cls is not None else 0
        if isinstance(nm, dict):
            name = str(nm.get(cls, f"class_{cls}"))
        else:
            name = f"class_{cls}"
        c = float(b.conf[0].item()) if b.conf is not None else 0.0
        xy = b.xyxy[0].cpu().numpy().tolist()
        box = (float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3]))
        out.append((str(name), c, box))
    return out


def _box_area(b: tuple[float, float, float, float]) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (_box_area(a) + _box_area(b) - inter + 1e-6)


def _center_in(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _expand_box(
    b: tuple[float, float, float, float], margin_frac: float = 0.12
) -> tuple[float, float, float, float]:
    """Grow box from center (helps when hand/object sits just outside a tight YOLO person box)."""
    w, h = b[2] - b[0], b[3] - b[1]
    mx, my = w * margin_frac, h * margin_frac
    return (b[0] - mx, b[1] - my, b[2] + mx, b[3] + my)


def _intersect_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _object_linked_to_person(
    pbox: tuple[float, float, float, float],
    obox: tuple[float, float, float, float],
    loose: bool = False,
) -> bool:
    """
    True if object box is "with" the person (cup, phone, book, etc.).
    ``loose=True`` is used for drinks / small hand objects that often sit
    on the border of a tight person box in webcam frames.
    """
    margin = 0.2 if loose else 0.12
    p_exp = _expand_box(pbox, margin_frac=margin)
    if _center_in(obox, p_exp) or _center_in(obox, pbox):
        return True
    iou = _iou(pbox, obox)
    th_iou = 0.0006 if loose else 0.0015
    if iou > th_iou:
        return True
    ia = _intersect_area(pbox, obox)
    obox_a = _box_area(obox)
    frac = 0.02 if loose else 0.04
    if obox_a > 1.0 and (ia / obox_a) > frac:
        return True
    return False


# Person box: avoid spurious "person" under low conf. Hand-held: keep fairly low for cups / bottles.
_CONF_PERSON = 0.24
_CONF_HAND_HELD = 0.14
_CONF_FOOD_DRINK = 0.10  # bottles/cups/food are small or low conf in low light
_CONF_OTHER = 0.2

_SET_HAND = {
    PHONE,
    LAPTOP,
    BOTTLE,
    CUP,
    WINE,
    BOOK,
    *DRINK,
    *EAT,
    *SPORTS,
    *SCREEN,
    *COMPUTER,
}
_HANDY_NAMES = _SET_HAND | {"wine glass"}


def _min_confidence_for_class(name: str) -> float:
    if name == PERSON:
        return _CONF_PERSON
    if name in EAT or name in DRINK or name in {BOTTLE, CUP, WINE, "bowl", "spoon", "fork", "knife"}:
        return _CONF_FOOD_DRINK
    if name in _SET_HAND or "phone" in name.lower():
        return _CONF_HAND_HELD
    return _CONF_OTHER


def _pair_of_people_close(
    pboxes: list[tuple[float, float, float, float]],
) -> bool:
    """Heuristic: two YOLO person boxes close enough to plausibly be in conversation (no audio)."""
    n = len(pboxes)
    if n < 2:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pboxes[i], pboxes[j]
            cax, cay = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
            cbx, cby = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
            dist = ((cax - cbx) ** 2 + (cay - cby) ** 2) ** 0.5
            w = (a[2] - a[0] + b[2] - b[0]) * 0.5
            w = max(w, 1.0)
            if dist < 2.6 * w:
                return True
    return False


def infer_activity(
    dets: list[tuple[str, float, tuple[float, float, float, float]]],
) -> str:
    """
    Rule-based activity from COCO (YOLO) classes linked to a person.
    COCO has no "drinking" or "talking" actions; we infer from objects and loose geometry.
    """
    by_name: dict[str, list[tuple[float, tuple[float, float, float, float]]]] = {}
    for n, c, b in dets:
        if c < _min_confidence_for_class(n):
            continue
        by_name.setdefault(n, []).append((c, b))

    persons = by_name.get(PERSON, [])
    person_boxes = [b for _, b in persons]

    def has_object(label: str, *, loose: bool = False) -> bool:
        keys = [label]
        if label == PHONE:
            keys.extend(
                k
                for k in by_name
                if "phone" in k.lower() and "microwave" not in k.lower()
            )
        for key in keys:
            if key not in by_name:
                continue
            for _, obox in by_name[key]:
                if not persons:
                    return True
                for _, pbox in persons:
                    if _object_linked_to_person(pbox, obox, loose=loose):
                        return True
        return False

    def has_any(
        members: set[str] | frozenset,
        *,
        loose: bool = False,
    ) -> bool:
        for m in members:
            if m not in by_name or not by_name[m]:
                continue
            if has_object(m, loose=loose):
                return True
        if not persons:
            return any(m in by_name and by_name[m] for m in members)
        return False

    if has_object(PHONE) or (PHONE in by_name and not persons):
        return "using phone"
    if has_object(LAPTOP) or (LAPTOP in by_name and not persons):
        return "using a laptop"
    if (has_object(BOOK, loose=True) and (has_object(LAPTOP) or has_object(MOUSE, loose=True))) or (
        (BOOK in by_name and LAPTOP in by_name) and (not persons)
    ):
        return "writing or studying (book and device in view)"
    if has_any(SCREEN, loose=True):
        return "watching a screen (TV or similar in view)"
    if not has_object(LAPTOP) and (has_object(MOUSE, loose=True) or (MOUSE in by_name and not persons)):
        return "at a computer (mouse; typing or on-screen work — pen/paper is rarely visible to YOLO)"
    seen_drink = has_any(DRINK, loose=True) or (not persons and any(k in by_name for k in DRINK))
    seen_eat = has_any(EAT, loose=True) or (not persons and any(k in by_name for k in EAT))
    if seen_drink and seen_eat:
        return "eating and drinking"
    if seen_drink:
        return "drinking (cup, bottle, or glass in view)"
    if seen_eat:
        return "eating (food, bowl, or utensils in view)"
    if has_object(BOOK) or (BOOK in by_name and not persons):
        return "reading"
    if has_any(SPORTS, loose=True):
        return "active (sports or exercise gear in view)"
    if len(person_boxes) >= 2:
        if _pair_of_people_close(person_boxes):
            return "with another person (often talking; not verified with audio)"
        return "with other people in the frame (often talking; not verified with audio)"
    if persons:
        return "in the frame (try brighter light or hold cups/bottles in front of the torso so YOLO can see them)"
    if dets:
        return "active in scene"
    return "no clear activity detected"


@dataclass
class AnalyzeResult:
    name: str
    face_match: bool
    face_score: float
    activity: str
    line: str
    detections: list[str]
    yolo_debug: str


def _format_time_safe() -> str:
    t = datetime.now()
    hour12 = t.hour % 12
    if hour12 == 0:
        hour12 = 12
    am_pm = "AM" if t.hour < 12 else "PM"
    return f"{hour12}:{t.minute:02d} {am_pm}"


def make_activity_line(name: str, face_match: bool, activity: str) -> str:
    t = _format_time_safe()
    display = (name or "Person").strip() or "Person"
    if not face_match:
        return f"{display} — not recognized in this frame at {t}"
    return f"{display} {activity} at {t}"


def enroll_from_pil(pil: Image.Image) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Compute reference embedding, or (None, error message) if no face."""
    e = _face_embed(pil)
    if e is None:
        return (None, "No face found in the reference image. Use a clear front-facing photo.")
    return (e, None)


def analyze_frame_pil(
    pil: Image.Image,
    ref_embed: np.ndarray,
    display_name: str,
    match_threshold: float | None = None,
) -> AnalyzeResult:
    dets = yolo_objects(pil)
    det_labels: list[str] = []
    for n, c, _ in dets:
        if c >= 0.25:
            det_labels.append(f"{n}")
    # de-dup preserve order
    seen = set()
    uniq: list[str] = []
    for x in det_labels:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    det_str = ", ".join(uniq[:12])
    th = match_threshold if match_threshold is not None else get_face_match_threshold()
    if ref_embed is None or getattr(ref_embed, "size", 0) == 0:
        m, sim = (False, 0.0)
    else:
        m, sim = match_face_pil(pil, ref_embed, th)
    if m:
        act = infer_activity(dets)
    else:
        act = "not recognized in frame"
    line = make_activity_line(display_name, m, act)
    return AnalyzeResult(
        name=display_name,
        face_match=m,
        face_score=sim,
        activity=act,
        line=line,
        detections=uniq[:20],
        yolo_debug=det_str,
    )


def match_face_pil(
    current: Image.Image, ref_embed: np.ndarray, threshold: float = 0.68
) -> tuple[bool, float]:
    """Match using max cosine vs one or more stored reference rows (N×D). Side profiles match a side ref."""
    e = _face_embed(current)
    if e is None or ref_embed is None:
        return (False, 0.0)
    r = np.atleast_2d(np.asarray(ref_embed, dtype=np.float32))
    if r.size == 0:
        return (False, 0.0)
    # rows are L2-normalized; e is too → dot = cosine
    sims = r @ e
    sim = float(np.max(sims))
    return (sim >= threshold, sim)


def enroll_many_embeddings(
    images: list[Image.Image],
) -> tuple[Optional[np.ndarray], list[str]]:
    """
    Stack embeddings from several photos (front + left + right, etc.).
    Returns (array shape N×D, per-image error strings; empty string if ok).
    """
    rows: list[np.ndarray] = []
    errors: list[str] = []
    for i, pil in enumerate(images):
        emb, err = enroll_from_pil(pil if pil.mode == "RGB" else pil.convert("RGB"))
        if emb is not None:
            rows.append(emb)
            errors.append("")
        else:
            errors.append(err or f"image {i + 1}: no face detected")
    if not rows:
        return (None, errors)
    return (np.stack(rows, axis=0), errors)
