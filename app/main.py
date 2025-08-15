# app/main.py

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import uvicorn

# Your analyzer module
from app import face_shape  # adjust if your import path differs

app = FastAPI(title="VISA Face API", version="2.0")

# CORS — keep permissive, or tighten to your shop domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # e.g. ["https://yourshop.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "version": app.version}

def _read_upload(file: UploadFile) -> bytes:
    if file is None:
        return b""
    data = file.file.read()
    file.file.close()
    return data

@app.post("/detect/face-shape")
async def detect_face_shape(
    files: Optional[List[UploadFile]] = File(default=None, description="3–5 best frames"),
    file: Optional[UploadFile] = File(default=None, description="legacy single frame")
) -> JSONResponse:
    """
    Accepts either:
      - multiple frames via 'files' (preferred), or
      - a single frame via 'file' (backward compatibility).
    """
    # Collect bytes in order (small list, we keep in memory)
    frames: List[bytes] = []

    if files:
        for f in files:
            if f and f.filename:
                frames.append(_read_upload(f))

    # Fallback to single-file if no 'files' provided
    if not frames and file:
        frames.append(_read_upload(file))

    if not frames:
        raise HTTPException(status_code=400, detail="No image(s) provided. Send 'files' or 'file'.")

    # Call analyzer: prefer analyze_frames(frames), else legacy analyze(frame)
    try:
        if hasattr(face_shape, "analyze_frames"):
            # New multi-frame API: expects List[bytes] (or adjust inside face_shape)
            result = face_shape.analyze_frames(frames)
        elif hasattr(face_shape, "analyze"):
            # Legacy single-frame analyzer: use first frame only
            result = face_shape.analyze(frames[0])
        else:
            raise RuntimeError("face_shape module has neither 'analyze_frames' nor 'analyze'.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyzer error: {e}")

    # Ensure a consistent response shape
    # Expected keys: face_shape, message, and any extras you add (e.g., debug, ratios, etc.)
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Analyzer returned unexpected result type.")

    # Minimal contract
    payload = {
        "face_shape": result.get("face_shape"),
        "message": result.get("message"),
        # pass through any extra keys your analyzer adds
        **{k: v for k, v in result.items() if k not in {"face_shape", "message"}}
    }

    return JSONResponse(payload)

if __name__ == "__main__":
    # Bind to 0.0.0.0 for Docker
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
