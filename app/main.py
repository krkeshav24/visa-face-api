# app/main.py
import io, os, json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv

from .face_shape import compute_metrics, classify_face_shape

load_dotenv()

app = FastAPI(title="VISA Face API", version="1.0.0")

ALLOWED = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MSG_PATH = os.path.join(DATA_DIR, "messages.json")

NORM = {
    "oval":"oval","round":"round","square":"square","diamond":"diamond","heart":"heart",
    "rectangle":"rectangle","oblong":"rectangle","long":"rectangle"
}

def _load_messages() -> dict:
    if not os.path.exists(MSG_PATH):
        return {}
    with open(MSG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        key = NORM.get(str(k).strip().lower(), str(k).strip().lower())
        out[key] = {
            "message": (v.get("message") or "").strip(),
            "rasa": v.get("rasa"),
            "celestial": v.get("celestial"),
            "code": v.get("code"),
        }
    return out

MESSAGES = _load_messages()

@app.get("/health")
def health():
    return {"ok": True, "messages_loaded": bool(MESSAGES), "allowed_origins": ALLOWED}

@app.post("/detect/face-shape")
async def detect_face_shape(file: UploadFile = File(...)):
    """
    Expects: multipart/form-data with field 'file' (JPEG/PNG).
    Returns: face_shape, message, rasa, celestial, code.
    """
    try:
        content = await file.read()
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    try:
        mets = compute_metrics(pil)
        shape = classify_face_shape(mets).lower()
        shape = NORM.get(shape, shape)
    except Exception as e:
        raise HTTPException(422, f"Detection failed: {e}")

    meta = MESSAGES.get(shape, {})
    return {
        "face_shape": shape,
        "message": meta.get("message", ""),
        "rasa": meta.get("rasa"),
        "celestial": meta.get("celestial"),
        "code": meta.get("code"),
    }
