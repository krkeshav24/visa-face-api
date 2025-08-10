# app/detect.py
import cv2, numpy as np, mediapipe as mp
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from .suggester import suggest_for_face

router = APIRouter()
mp_face = mp.solutions.face_mesh

# FaceMesh landmark indices
TOP, CHIN = 10, 152
L_TEMPLE, R_TEMPLE = 127, 356
L_CHEEK,  R_CHEEK  = 234, 454
L_JAW,    R_JAW    = 227, 447

class DetectResponse(BaseModel):
  shape: str
  ratios: dict
  suggestions: list

def d(a,b): return float(np.hypot(a.x-b.x, a.y-b.y))

def classify(h, fw, cw, jw):
  ipd = fw or 1.0
  H, FW, CW, JW = h/ipd, fw/ipd, cw/ipd, jw/ipd
  if abs(FW-JW) < 0.05 and H > 1.5 and FW < 0.9: s="oblong"
  elif JW > FW*0.95 and CW > FW and H < 1.45:    s="square"
  elif FW > JW*1.08 and CW > FW and H < 1.5:     s="heart"
  elif abs(FW-JW) < 0.08 and abs(CW-FW) < 0.08 and 1.3 <= H <= 1.55: s="round"
  else: s="oval"
  return s, {"H":H,"FW":FW,"CW":CW,"JW":JW}

@router.post("/detect", response_model=DetectResponse)
async def detect(image: UploadFile = File(...)):
  try:
    arr = np.frombuffer(await image.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "Bad image")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
      res = fm.process(rgb)
    if not res.multi_face_landmarks: raise HTTPException(422, "No face found")
    lm = res.multi_face_landmarks[0].landmark
    H  = d(lm[CHIN], lm[TOP]) * 1.16
    FW = d(lm[L_TEMPLE], lm[R_TEMPLE])
    CW = d(lm[L_CHEEK],  lm[R_CHEEK])
    JW = d(lm[L_JAW],    lm[R_JAW])
    shape, ratios = classify(H, FW, CW, JW)
    suggestions = [s.model_dump() for s in suggest_for_face(shape)]
    return DetectResponse(shape=shape, ratios=ratios, suggestions=suggestions)
  except HTTPException: raise
  except Exception as e: raise HTTPException(500, f"Detect error: {e}")
