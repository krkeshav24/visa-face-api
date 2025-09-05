# app/main.py

from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

# Your analyzer module (adjust import path if needed)
from app import face_shape

logger = logging.getLogger("visa_face_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="VISA Face API", version="2.0")

# CORS — keep permissive for development; lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "version": app.version}

async def _read_upload(file: UploadFile) -> bytes:
    """Read bytes from UploadFile and close it.

    Using UploadFile.read() is async-friendly and avoids loading extra
    file descriptor handles.
    """
    if file is None:
        return b""
    data = await file.read()
    try:
        await file.close()
    except Exception:
        # UploadFile.close may be sync depending on starlette version
        try:
            file.file.close()
        except Exception:
            pass
    return data

@app.post("/detect/face-shape")
async def detect_face_shape(
    file: UploadFile = File(..., description="Single image file (jpg/png)")
) -> JSONResponse:
    """
    Accepts only a single uploaded image.

    Expects your analyzer (app.face_shape) to expose a function `analyze(frame_bytes)`
    that takes raw bytes and returns a dict with at least `face_shape` and `message`
    (additional keys are passed through).
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No image provided. Send a 'file'.")

    # Optional: check content type and reasonable file size
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        logger.warning("Incoming file has content_type=%s", file.content_type)
        # do not reject strictly if you want flexibility; uncomment to enforce
        # raise HTTPException(status_code=415, detail="Unsupported media type")

    try:
        frame = await _read_upload(file)
    except Exception as e:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    if not frame:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Call analyzer (legacy single-frame API expected)
    try:
        if hasattr(face_shape, "analyze"):
            result = face_shape.analyze(frame)
        else:
            raise RuntimeError("face_shape module has no 'analyze' function.")
    except HTTPException:
        # let FastAPI HTTPExceptions bubble up unchanged
        raise
    except Exception as e:
        logger.exception("Analyzer error")
        raise HTTPException(status_code=500, detail=f"Analyzer error: {e}")

    # Validate contract
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Analyzer returned unexpected result type.")

    payload = {
        "face_shape": result.get("face_shape"),
        "message": result.get("message"),
        # pass through any extras
        **{k: v for k, v in result.items() if k not in {"face_shape", "message"}}
    }

    return JSONResponse(payload)

if __name__ == "__main__":
    # Run with: python -m app.main OR in Docker set entrypoint/command to this file
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
