# app/face_shape.py
import math
from typing import Dict, Tuple
import numpy as np
from PIL import Image
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

IDX = {
    "CHIN": 152, "TOP": 10,
    "LC": 234, "RC": 454,            # cheek width
    "LJ": 172, "RJ": 397,            # jaw corners
    "LEO": 33,  "LEI": 133,          # left eye outer/inner
    "REI": 362, "REO": 263,          # right eye inner/outer
}

def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _xy(lm, w: int, h: int, i: int) -> Tuple[float, float]:
    p = lm[i]; return (p.x * w, p.y * h)

def compute_metrics(pil_img: Image.Image) -> Dict[str, float]:
    """Returns H, FW, CW, JW, IPD (+ simple pose proxies) from a single image."""
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
        res = fm.process(img)
        if not res.multi_face_landmarks:
            raise ValueError("No face detected")
        lm = res.multi_face_landmarks[0].landmark

        chin, top = _xy(lm, w, h, IDX["CHIN"]), _xy(lm, w, h, IDX["TOP"])
        lc, rc   = _xy(lm, w, h, IDX["LC"]),   _xy(lm, w, h, IDX["RC"])
        lj, rj   = _xy(lm, w, h, IDX["LJ"]),   _xy(lm, w, h, IDX["RJ"])

        leo, lei = _xy(lm, w, h, IDX["LEO"]), _xy(lm, w, h, IDX["LEI"])
        reo, rei = _xy(lm, w, h, IDX["REO"]), _xy(lm, w, h, IDX["REI"])
        left_eye  = ((leo[0]+lei[0])/2.0, (leo[1]+lei[1])/2.0)
        right_eye = ((reo[0]+rei[0])/2.0, (reo[1]+rei[1])/2.0)

        # hairline heuristic (extend to hairline)
        H  = _euclid(chin, top) * 1.16
        CW = _euclid(lc, rc)
        JW = _euclid(lj, rj)
        FW = CW
        IPD = _euclid(left_eye, right_eye)

        # roll from eye line (deg)
        roll_rad = math.atan2(reo[1] - leo[1], reo[0] - leo[0])
        roll_deg = abs(roll_rad * 180.0 / math.pi)

        # crude yaw proxy from z-depth + off-center
        Lz, Rz = lm[IDX["LC"]].z, lm[IDX["RC"]].z
        zdiff = abs(Lz - Rz)
        midx = (lm[IDX["LC"]].x + lm[IDX["RC"]].x) / 2.0
        xoff = abs(midx - 0.5)
        yaw_proxy = 0.0
        if zdiff > 0.01: yaw_proxy += (zdiff - 0.01) * 100.0
        if xoff  > 0.03: yaw_proxy += (xoff  - 0.03) * 100.0

        return {
            "H": float(H), "FW": float(FW), "CW": float(CW), "JW": float(JW), "IPD": float(IPD),
            "roll_deg": float(roll_deg), "yaw_proxy": float(yaw_proxy)
        }

def classify_face_shape(m: Dict[str, float]) -> str:
    """Heuristic mapping to: oval, round, square, rectangle, diamond, heart."""
    H, FW, CW, JW = m["H"], m["FW"], m["CW"], m["JW"]
    aspect = H / max(FW, 1e-6)
    diff_fw_jw = abs(FW - JW) / max(FW, JW, 1e-6)
    cw_top = (CW >= FW and CW >= JW)
    fw_top = (FW >= CW and FW >= JW)
    jw_small = JW / max(CW, FW, 1e-6) < 0.8

    if aspect >= 1.5 and diff_fw_jw < 0.12:
        return "rectangle"
    if 1.0 <= aspect <= 1.25 and diff_fw_jw <= 0.10 and fw_top:
        return "square"
    if 0.95 <= aspect <= 1.20 and diff_fw_jw <= 0.06 and CW >= FW * 0.96:
        return "round"
    if cw_top and jw_small and aspect >= 1.25:
        return "heart"
    if cw_top and FW / max(CW,1e-6) < 0.92 and JW / max(CW,1e-6) < 0.92 and aspect >= 1.2:
        return "diamond"
    return "oval"
