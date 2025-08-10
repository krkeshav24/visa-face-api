from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np
import cv2
import io, json, os

import mediapipe as mp
from .face import compute_metrics, classify_face_shape

SUGGESTIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "suggestions.json")

app = FastAPI(title="VISA Face API", version="0.2.0")

# --- CORS ---
allowed_origins = [
    "https://q51zdw-if.myshopify.com",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VersionInfo(BaseModel):
    name: str = "visa-face-api"
    version: str = "0.2.0"
    description: str = "Face shape detection + earrings suggestions"
    cors: List[str] = allowed_origins

class DetectionResponse(BaseModel):
    shape: str
    confidence: float
    metrics: Dict[str, float]
    ratios: Dict[str, float]

class Suggestion(BaseModel):
    title: str
    image_url: str
    product_url: str

class SuggestionsResponse(BaseModel):
    shape: str
    count: int
    items: List[Suggestion]

@app.get("/", response_model=VersionInfo)
def root():
    return VersionInfo()

@app.get("/health")
def health():
    return {"status": "ok"}

def _load_image_to_rgb(img_bytes: bytes) -> np.ndarray:
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    return np.array(pil)

@app.post("/detect-face-shape", response_model=DetectionResponse)
async def detect_face_shape(file: UploadFile = File(...)):
    content = await file.read()
    rgb = _load_image_to_rgb(content)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        result = face_mesh.process(rgb)  # RGB expected
        if not result.multi_face_landmarks:
            raise HTTPException(status_code=422, detail="No face detected")

        # Take first face
        lm = result.multi_face_landmarks[0].landmark
        # Convert to normalized (x,y) array
        pts = np.array([[p.x, p.y] for p in lm], dtype=np.float32)

        metrics = compute_metrics(pts)
        shape, conf = classify_face_shape(metrics)

        return DetectionResponse(
            shape=shape,
            confidence=round(conf, 3),
            metrics={
                "H": round(metrics.H, 4),
                "FW": round(metrics.FW, 4),
                "CW": round(metrics.CW, 4),
                "JW": round(metrics.JW, 4),
                "IPD": round(metrics.IPD, 4),
            },
            ratios={k: round(v, 4) for k, v in metrics.ratios.items()}
        )

@app.get("/suggest/earrings", response_model=SuggestionsResponse)
def suggest_earrings(shape: str = Query(..., description="face shape label")):
    shape = shape.lower().strip()
    if not os.path.exists(SUGGESTIONS_PATH):
        raise HTTPException(status_code=500, detail="Suggestions file missing")
    try:
        with open(SUGGESTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read suggestions")

    items = data.get(shape) or data.get("default") or []
    # Validate items
    cleaned = []
    for it in items[:5]:
        if all(k in it for k in ("title", "image_url", "product_url")):
            cleaned.append(it)
    return SuggestionsResponse(shape=shape, count=len(cleaned), items=cleaned)
