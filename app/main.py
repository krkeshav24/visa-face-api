import os, json
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from .face import classify_face_shape_from_image

VERSION = "0.1.0"

def get_env_list(key: str) -> List[str]:
    raw = os.getenv(key, "")
    # split by comma and strip whitespace
    return [x.strip() for x in raw.split(",") if x.strip()]

ALLOWED_ORIGINS = get_env_list("ALLOWED_ORIGINS")
API_KEY = os.getenv("API_KEY", "")

app = FastAPI(title="Face API", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load suggestions once at startup
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SUGGESTIONS_PATH = os.path.join(BASE_DIR, "data", "suggestions.json")
if not os.path.exists(SUGGESTIONS_PATH):
    # Also look relative to CWD for Docker/local edge cases
    alt = os.path.join(os.getcwd(), "data", "suggestions.json")
    SUGGESTIONS_PATH = alt if os.path.exists(alt) else SUGGESTIONS_PATH

def load_suggestions() -> Dict[str, List[dict]]:
    try:
        with open(SUGGESTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

SUGGESTIONS = load_suggestions()

def require_api_key(x_api_key: str = Header(default="")):
    if not API_KEY:
        # No API_KEY configured → allow all (useful for local dev)
        return True
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": VERSION}

@app.post("/detect-face-shape")
async def detect_face_shape(file: UploadFile = File(...), auth_ok: bool = require_api_key) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB safety limit
        raise HTTPException(413, "File too large (max 10MB)")
    shape, confidence = classify_face_shape_from_image(content)
    return {"shape": shape, "confidence": confidence}

@app.get("/suggest/{face_shape}")
def suggest(face_shape: str, auth_ok: bool = require_api_key):
    key = face_shape.lower()
    items = SUGGESTIONS.get(key, [])
    return {"shape": key, "count": len(items), "items": items}

@app.get("/")
def root():
    return JSONResponse({"ok": True, "service": "Face API", "version": VERSION})
