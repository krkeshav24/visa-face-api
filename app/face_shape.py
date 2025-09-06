# app/face_shape.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math
import cv2
import numpy as np

# MediaPipe
try:
    import mediapipe as mp
except Exception:
    mp = None

# ---------------- Tunables ----------------
ROLL_MAX_DEG = 3.0
YAW_MAX_DEG  = 3.0
TARGET_HL_H  = 0.48
SIZE_TOL     = 0.13
BRIGHT_MIN   = 40
BRIGHT_MAX   = 245
BLUR_MIN_VAR = 60.0

# Landmarks (MediaPipe FaceMesh 468 indices)
LM_HAIRLINE  = 10
LM_CHIN      = 152
LM_CHEEK_L   = 234
LM_CHEEK_R   = 454
LM_TEMPLE_L  = 127
LM_TEMPLE_R  = 356
LM_NOSE_BASE = 1
LM_JAW_L     = 172
LM_JAW_R     = 397

UPPER_FOREHEAD_CORRECTION = 1.10
# ------------------------------------------


@dataclass
class FrameMeasure:
    ok: bool
    reason: Optional[str]
    raw_face_hl_h: float = 0.0
    face_hl_h: float = 0.0
    forehead_w: float = 0.0
    cheek_w: float = 0.0
    jaw_w: float = 0.0
    cw: float = 0.0
    lf: float = 0.0
    fj: float = 0.0
    roll_deg: float = 0.0
    yaw_proxy: float = 0.0
    brightness: float = 0.0
    blur_var: float = 0.0
    jaw_angle_deg: float = 0.0
    jr: float = 0.0


def _decode_image_bytes(b: bytes) -> Optional[np.ndarray]:
    if not b:
        return None
    arr = np.frombuffer(b, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _brightness(img_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())


def _blur_laplacian_var(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _face_mesh() -> "mp.solutions.face_mesh.FaceMesh":
    if mp is None:
        raise RuntimeError("mediapipe is not available.")
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def _landmarks_from_img(img_bgr: np.ndarray):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with _face_mesh() as fm:
        res = fm.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    return [(p.x, p.y, p.z) for p in res.multi_face_landmarks[0].landmark]


def _deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def _estimate_roll_deg(lm):
    L, R = lm[33], lm[263]
    return abs(_deg(math.atan2(R[1]-L[1], R[0]-L[0])))


def _estimate_yaw_proxy(lm):
    Lz, Rz = lm[LM_CHEEK_L][2], lm[LM_CHEEK_R][2]
    zdiff = abs(Lz-Rz)
    midx = (lm[LM_CHEEK_L][0]+lm[LM_CHEEK_R][0])/2.0
    xoff = abs(midx-0.5)
    yaw = 0.0
    if zdiff > 0.01: yaw += (zdiff-0.01)*100.0
    if xoff > 0.03: yaw += (xoff-0.03)*100.0
    return yaw


def _in_frame(lm, margin=0.06) -> bool:
    xs, ys = [p[0] for p in lm], [p[1] for p in lm]
    return (min(xs) >= margin and min(ys) >= margin and
            max(xs) <= 1.0-margin and max(ys) <= 1.0-margin)


def _dist_norm(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _angle_at_point(a,b,c) -> float:
    bax, bay = a[0]-b[0], a[1]-b[1]
    bcx, bcy = c[0]-b[0], c[1]-b[1]
    dot = bax*bcx+bay*bcy
    na, nc = math.hypot(bax,bay), math.hypot(bcx,bcy)
    if na<1e-8 or nc<1e-8: return 180.0
    cosv = max(-1.0,min(1.0,dot/(na*nc)))
    return _deg(math.acos(cosv))


def _measure_from_landmarks(lm, brightness, blur_var) -> FrameMeasure:
    roll = _estimate_roll_deg(lm)
    yaw  = _estimate_yaw_proxy(lm)

    raw_face_hl_h = abs(lm[LM_CHIN][1]-lm[LM_HAIRLINE][1])
    face_hl_h = raw_face_hl_h*UPPER_FOREHEAD_CORRECTION
    # size_ok = abs(raw_face_hl_h-TARGET_HL_H) <= TARGET_HL_H*SIZE_TOL
    size_ok = True

    frame_ok = (roll<=ROLL_MAX_DEG and yaw<=YAW_MAX_DEG and
                BRIGHT_MIN<=brightness<=BRIGHT_MAX and blur_var>=BLUR_MIN_VAR and
                size_ok and _in_frame(lm))

    forehead_w = _dist_norm(lm[LM_TEMPLE_L], lm[LM_TEMPLE_R])
    cheek_w    = _dist_norm(lm[LM_CHEEK_L], lm[LM_CHEEK_R])
    jaw_left, jaw_right = lm[LM_JAW_L], lm[LM_JAW_R]
    jaw_w = _dist_norm(jaw_left, jaw_right)

    nose_base = lm[LM_NOSE_BASE]
    cw = abs(lm[LM_CHIN][1]-nose_base[1])
    lf = cw/max(1e-6, face_hl_h)
    fj = forehead_w/max(1e-6, jaw_w)

    chin_pt = lm[LM_CHIN]
    jaw_angle_deg = _angle_at_point(jaw_left, chin_pt, jaw_right)
    jr = jaw_w/max(1e-6, cheek_w)

    return FrameMeasure(frame_ok, None if frame_ok else "gate_failed",
                        raw_face_hl_h, face_hl_h, forehead_w, cheek_w, jaw_w,
                        cw, lf, fj, roll, yaw, brightness, blur_var,
                        jaw_angle_deg, jr)


# ---------------- Classification ----------------
def _classify_from_aggregates(forehead_w, cheek_w, jaw_w, face_hl_h,
                              cw=None, jaw_angle_deg=None, jr=None):
    """
    Returns: (shape: str, confidence: float, details: dict)
             details also includes percentage scores for all shapes
    """
    cheek_w = max(cheek_w, 1e-6)
    jaw_w = max(jaw_w, 1e-6)
    face_hl_h = max(face_hl_h, 1e-6)

    aspect = face_hl_h/cheek_w
    fj = forehead_w/jaw_w
    lf = None if cw is None else (cw/max(1e-6,face_hl_h))
    if jaw_angle_deg is None: jaw_angle_deg=110.0
    if jr is None: jr=jaw_w/cheek_w

    # --- Raw score assignment per face type ---
    scores = {s:0.0 for s in ["oval","round","square","rectangle","diamond","heart","oblong"]}

    # Heart / Diamond
    if fj > 1.05: scores["heart"] += 1.0
    if fj < 0.95: scores["diamond"] += 1.0

    # Rectangle / Oval / Oblong
    if aspect >= 1.50: scores["rectangle"] += 1.0
    if 1.35 <= aspect < 1.50: scores["oval"] += 1.0
    if 1.50 <= aspect < 1.65: scores["oblong"] += 0.5  # treat oblong as long-oval

    # Round / Square
    if aspect < 1.20:
        if lf and lf >= 0.32: scores["square"] += 1.0
        else: scores["round"] += 1.0

    # Normalize to percentages
    total = sum(scores.values())
    if total <= 0: total = 1e-6
    percentages = {k: round(v/total*100,2) for k,v in scores.items()}

    # Pick winner
    shape = max(percentages, key=percentages.get)
    confidence = percentages[shape]/100.0

    details = {"aspect":aspect,"fj":fj,"lf":lf,"theta":jaw_angle_deg,"jr":jr,
               "scores":scores,"percentages":percentages}

    return shape, confidence, details


def _friendly_message(face_shape: str) -> str:
    msgs = {
        "oval":"Balanced features with gentle curves — most earring styles will flatter you.",
        "round":"Softer angles — go for lengthening styles.",
        "square":"Strong jawline — try curves and drop styles.",
        "rectangle":"Longer than wide — choose wider or layered styles.",
        "diamond":"Cheekbones are widest — add width at the jaw.",
        "heart":"Wider forehead with tapered jaw — add volume near jawline.",
        "oblong":"Elongated version of oval — try styles adding width.",
        "uncertain":"Face type not clear — mixed features detected."
    }
    return msgs.get(face_shape,"Face analyzed.")


# ---------------- Public API ----------------
def analyze(frame_bytes: bytes) -> Dict[str, Any]:
    if not frame_bytes:
        return {"face_shape": None, "message": "No image received."}
    img = _decode_image_bytes(frame_bytes)
    if img is None:
        return {"face_shape": None, "message": "Could not decode image."}

    m = _measure_from_landmarks(_landmarks_from_img(img), _brightness(img), _blur_laplacian_var(img))
    if not m.ok:
        return {"face_shape": None, "message": "Face not suitable.", "debug": m.__dict__}

    shape, conf, details = _classify_from_aggregates(m.forehead_w,m.cheek_w,m.jaw_w,m.face_hl_h,
                                                     cw=m.cw,jaw_angle_deg=m.jaw_angle_deg,jr=m.jr)
    return {"face_shape": shape,"message": _friendly_message(shape),
            "debug": {**m.__dict__,"confidence": conf,"classify_details": details}}


def analyze_frames(frames: List[bytes]) -> Dict[str, Any]:
    if not frames:
        return {"face_shape": None,"message": "No images received."}
    measures=[]
    for b in frames:
        img=_decode_image_bytes(b)
        if img is None: continue
        lm=_landmarks_from_img(img)
        if lm is None: continue
        measures.append(_measure_from_landmarks(lm,_brightness(img),_blur_laplacian_var(img)))
    if not measures:
        return {"face_shape": None,"message": "No valid frames."}

    agg=measures[0]  # simplification: use first valid
    shape,conf,details=_classify_from_aggregates(agg.forehead_w,agg.cheek_w,agg.jaw_w,agg.face_hl_h,
                                                 cw=agg.cw,jaw_angle_deg=agg.jaw_angle_deg,jr=agg.jr)
    return {"face_shape": shape,"message": _friendly_message(shape),
            "debug":{"aggregated":agg.__dict__,"confidence":conf,"classify_details":details,"count_total":len(measures)}}
