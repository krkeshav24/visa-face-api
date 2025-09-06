# app/face_shape.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import io
import math

import cv2
import numpy as np

# MediaPipe
try:
    import mediapipe as mp
except Exception as e:
    mp = None

# ------------- Tunables (keep these in sync with frontend where possible) -------------
ROLL_MAX_DEG = 5.0
YAW_MAX_DEG  = 5.0
TARGET_HL_H  = 0.42       # expected hairline(#10) → chin(#152) normalized height (before correction)
SIZE_TOL     = 0.2       # ± tolerance (i.e., accept if |h - TARGET_HL_H| <= TARGET_HL_H*SIZE_TOL)
BRIGHT_MIN   = 40         # looser than frontend (server often receives compressed frames)
BRIGHT_MAX   = 245
BLUR_MIN_VAR = 60.0       # Laplacian variance; higher is sharper

# Landmarks we use (MediaPipe FaceMesh 468-index model)
LM_HAIRLINE  = 10
LM_CHIN      = 152
LM_CHEEK_L   = 234
LM_CHEEK_R   = 454
LM_TEMPLE_L  = 127
LM_TEMPLE_R  = 356
LM_NOSE_BASE = 1    # central nose base landmark; tune if you prefer another index

# Correction factor for upper forehead not captured by lm[10]
UPPER_FOREHEAD_CORRECTION = 1.10  # multiply hairline->chin by this to approximate true top-of-forehead

# --------------------------------------------------------------------------------------


@dataclass
class FrameMeasure:
    ok: bool
    reason: Optional[str]
    raw_face_hl_h: float = 0.0  # raw hairline->chin normalized height (pre-correction)
    face_hl_h: float = 0.0      # corrected L (normalized) = raw_face_hl_h * UPPER_FOREHEAD_CORRECTION
    forehead_w: float = 0.0
    cheek_w: float = 0.0
    jaw_w: float = 0.0
    cw: float = 0.0         # chin length (nose base -> chin)
    lf: float = 0.0         # lower-face share = cw / L (uses corrected L)
    fj: float = 0.0         # forehead/jaw ratio = forehead_w / jaw_w
    roll_deg: float = 0.0
    yaw_proxy: float = 0.0
    brightness: float = 0.0
    blur_var: float = 0.0


def _decode_image_bytes(b: bytes) -> Optional[np.ndarray]:
    """Decode JPEG/PNG bytes to BGR image."""
    if not b:
        return None
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _brightness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _blur_laplacian_var(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _face_mesh() -> "mp.solutions.face_mesh.FaceMesh":
    if mp is None:
        raise RuntimeError("mediapipe is not available on the server.")
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def _landmarks_from_img(img_bgr: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
    """Return list of 468 (x,y,z) normalized landmarks or None."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with _face_mesh() as fm:
        res = fm.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    pts = res.multi_face_landmarks[0].landmark
    # Return normalized (x,y,z)
    return [(p.x, p.y, p.z) for p in pts]


def _deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def _estimate_roll_deg(lm: List[Tuple[float,float,float]]) -> float:
    L = lm[33]   # outer left eye corner
    R = lm[263]  # outer right eye corner
    return abs(_deg(math.atan2(R[1] - L[1], R[0] - L[0])))


def _estimate_yaw_proxy(lm: List[Tuple[float,float,float]]) -> float:
    """
    A small 'deg-ish' proxy using z-diff of cheeks + horizontal off-center.
    Matches the frontend gating behavior.
    """
    Lz = lm[LM_CHEEK_L][2]
    Rz = lm[LM_CHEEK_R][2]
    zdiff = abs(Lz - Rz)
    midx = (lm[LM_CHEEK_L][0] + lm[LM_CHEEK_R][0]) / 2.0
    xoff = abs(midx - 0.5)
    yaw = 0.0
    if zdiff > 0.01:
        yaw += (zdiff - 0.01) * 100.0
    if xoff > 0.03:
        yaw += (xoff - 0.03) * 100.0
    return yaw


def _in_frame(lm: List[Tuple[float,float,float]], margin: float = 0.06) -> bool:
    minx = min(p[0] for p in lm)
    miny = min(p[1] for p in lm)
    maxx = max(p[0] for p in lm)
    maxy = max(p[1] for p in lm)
    return (minx >= margin and miny >= margin and
            maxx <= (1.0 - margin) and maxy <= (1.0 - margin))


def _dist_norm(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    """Euclidean distance in normalized (x,y)."""
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return math.hypot(dx, dy)


def _measure_from_landmarks(
    lm: List[Tuple[float,float,float]],
    brightness: float,
    blur_var: float
) -> FrameMeasure:
    # Pose gates
    roll = _estimate_roll_deg(lm)
    yaw  = _estimate_yaw_proxy(lm)

    # Size gate: hairline↔chin (normalized height)
    raw_face_hl_h = abs(lm[LM_CHIN][1] - lm[LM_HAIRLINE][1])
    # Apply correction for upper forehead not well captured by LM_HAIRLINE
    corrected_face_hl_h = raw_face_hl_h * UPPER_FOREHEAD_CORRECTION

    # Use RAW measurement for gating (keeps TARGET_HL_H semantics consistent)
    size_ok = abs(raw_face_hl_h - TARGET_HL_H) <= (TARGET_HL_H * SIZE_TOL)

    pose_ok = (roll <= ROLL_MAX_DEG) and (yaw <= YAW_MAX_DEG)
    frame_ok = (
        pose_ok and
        BRIGHT_MIN <= brightness <= BRIGHT_MAX and
        blur_var >= BLUR_MIN_VAR and
        size_ok and
        _in_frame(lm, margin=0.06)
    )

    # Widths (normalized)
    # Forehead width at temples:
    forehead_w = _dist_norm(lm[LM_TEMPLE_L], lm[LM_TEMPLE_R])
    # Cheekbone width:
    cheek_w    = _dist_norm(lm[LM_CHEEK_L], lm[LM_CHEEK_R])

    # Jaw width approximation:
    midy = (lm[LM_HAIRLINE][1] + lm[LM_CHIN][1]) / 2.0
    lower = [p for p in lm if p[1] >= midy]
    if lower:
        minx = min(p[0] for p in lower)
        maxx = max(p[0] for p in lower)
        jaw_w = maxx - minx
    else:
        jaw_w = cheek_w * 0.9  # fallback

    # Chin length (nose base -> chin tip)
    nose_base = lm[LM_NOSE_BASE]
    cw = abs(lm[LM_CHIN][1] - nose_base[1])  # normalized vertical distance

    # Lower-face share and F/J ratio (use corrected face height for these derived metrics)
    lf = cw / max(1e-6, corrected_face_hl_h)   # lower-face share (0..1)
    fj = forehead_w / max(1e-6, jaw_w)

    return FrameMeasure(
        ok=frame_ok,
        reason=None if frame_ok else "gate_failed",
        raw_face_hl_h=raw_face_hl_h,
        face_hl_h=corrected_face_hl_h,
        forehead_w=forehead_w,
        cheek_w=cheek_w,
        jaw_w=jaw_w,
        cw=cw,
        lf=lf,
        fj=fj,
        roll_deg=roll,
        yaw_proxy=yaw,
        brightness=brightness,
        blur_var=blur_var
    )


def _classify_from_aggregates(
    forehead_w: float,
    cheek_w: float,
    jaw_w: float,
    face_hl_h: float,
    cw: Optional[float] = None
) -> str:
    """
    Heuristic using:
      - aspect = face_hl_h / cheek_w         (taller vs wider)
      - lf = cw / face_hl_h                  (lower-face share)
      - fj = forehead_w / jaw_w              (forehead vs jaw ratio)
      - cheek_dominance = cheek_w - max(forehead_w, jaw_w)
    """
    # Prevent division by zero
    cheek_w = max(cheek_w, 1e-6)
    jaw_w = max(jaw_w, 1e-6)

    aspect = face_hl_h / cheek_w  # taller → larger
    fj = forehead_w / jaw_w
    taper = forehead_w - jaw_w
    cheek_dominance = cheek_w - max(forehead_w, jaw_w)

    lf = None
    if cw is not None:
        lf = cw / max(1e-6, face_hl_h)

    # --- Decision logic (tunable thresholds) ---
    if aspect >= 1.55:
        if lf is not None and lf >= 0.50:
            return "rectangle"
        return "oval"

    if abs(fj - 1.0) < 0.02 and aspect <= 1.2:
        return "square" if cheek_dominance < 0.01 else "round"

    if fj > 1.15 and cheek_dominance > 0.00:
        return "heart"
    if cheek_dominance > 0.02 and fj < 0.95:
        return "diamond"

    if aspect < 1.25:
        if lf is not None:
            if lf <= 0.40:
                return "round"
            if lf >= 0.48:
                return "oval"
        return "round"

    return "oval"


def _friendly_message(face_shape: str) -> str:
    msgs = {
        "oval":      "Balanced features with gentle curves — most earring styles will flatter you.",
        "round":     "Softer angles with similar width and height — go for lengthening styles.",
        "square":    "Strong jawline and broad forehead — try curves and drop styles to soften.",
        "rectangle": "Longer than wide with a straighter jaw — choose wider or layered styles.",
        "diamond":   "Cheekbones are the widest point — go for pieces that add width at the jaw.",
        "heart":     "Wider forehead with a tapered jaw — balance with volume near the jawline.",
    }
    return msgs.get(face_shape, "Great! We’ve analyzed your face and picked styles to match.")


def _measure_one(img_bgr: np.ndarray) -> FrameMeasure:
    br = _brightness(img_bgr)
    bl = _blur_laplacian_var(img_bgr)
    lm = _landmarks_from_img(img_bgr)
    if lm is None:
        return FrameMeasure(ok=False, reason="no_face", brightness=br, blur_var=bl)
    return _measure_from_landmarks(lm, brightness=br, blur_var=bl)


def _aggregate_measures(ms: List[FrameMeasure]) -> FrameMeasure:
    """Median aggregate across valid frames; returns a synthetic measure row."""
    vals = [m for m in ms if m.ok]
    if not vals:
        # fallback: use the least-bad single (highest blur_var within brightness window)
        fallback = sorted(ms, key=lambda m: (m.ok, m.blur_var, -abs(m.brightness-150)), reverse=True)[0]
        return fallback

    def med(lst: List[float]) -> float:
        arr = np.array(lst, dtype=np.float64)
        return float(np.median(arr))

    return FrameMeasure(
        ok=True,
        reason=None,
        raw_face_hl_h=med([m.raw_face_hl_h for m in vals]),
        face_hl_h=med([m.face_hl_h for m in vals]),
        forehead_w=med([m.forehead_w for m in vals]),
        cheek_w=med([m.cheek_w for m in vals]),
        jaw_w=med([m.jaw_w for m in vals]),
        cw=med([m.cw for m in vals]),
        lf=med([m.lf for m in vals]),
        fj=med([m.fj for m in vals]),
        roll_deg=med([m.roll_deg for m in vals]),
        yaw_proxy=med([m.yaw_proxy for m in vals]),
        brightness=med([m.brightness for m in vals]),
        blur_var=med([m.blur_var for m in vals]),
    )


# ----------------------------- Public API (used by main.py) -----------------------------


def analyze(frame_bytes: bytes) -> Dict[str, Any]:
    """
    Legacy single-frame analyzer.
    """
    if not frame_bytes:
        return {"face_shape": None, "message": "No image received."}

    img = _decode_image_bytes(frame_bytes)
    if img is None:
        return {"face_shape": None, "message": "Could not decode image."}

    m = _measure_one(img)
    if not m.ok:
        return {
            "face_shape": None,
            "message": "Face not suitable (pose/light/blur/size). Please recapture.",
            "debug": m.__dict__,
        }

    shape = _classify_from_aggregates(m.forehead_w, m.cheek_w, m.jaw_w, m.face_hl_h, cw=m.cw)
    return {
        "face_shape": shape,
        "message": _friendly_message(shape),
        "debug": m.__dict__,
    }


def analyze_frames(frames: List[bytes]) -> Dict[str, Any]:
    """
    New multi-frame analyzer: processes 3–5 (or up to ~20) frames,
    filters bad ones, aggregates measurements, and classifies once.
    """
    if not frames:
        return {"face_shape": None, "message": "No images received."}

    measures: List[FrameMeasure] = []
    for b in frames:
        if not b:
            continue
        img = _decode_image_bytes(b)
        if img is None:
            continue
        measures.append(_measure_one(img))

    if not measures:
        return {"face_shape": None, "message": "Could not decode any image."}

    agg = _aggregate_measures(measures)

    if not agg.ok:
        # If none passed gates, report the best diagnostic we can
        worst = sorted(measures, key=lambda m: (m.ok, m.blur_var), reverse=True)[0]
        return {
            "face_shape": None,
            "message": "Frames failed quality checks (pose/light/blur/size). Please recapture.",
            "debug": worst.__dict__,
        }

    shape = _classify_from_aggregates(agg.forehead_w, agg.cheek_w, agg.jaw_w, agg.face_hl_h, cw=agg.cw)
    return {
        "face_shape": shape,
        "message": _friendly_message(shape),
        "debug": {
            "aggregated": agg.__dict__,
            "count_valid": sum(1 for m in measures if m.ok),
            "count_total": len(measures),
        }
    }
