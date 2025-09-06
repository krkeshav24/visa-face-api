# app/main.py

import os
import time
import uuid
import json
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

# ----------------- Image saving setup -----------------
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Images")
os.makedirs(SAVE_DIR, exist_ok=True)

def _unique_filename(orig_name: str) -> str:
    ext = os.path.splitext(orig_name)[1].lower() or ".jpg"
    return f"{int(time.time())}_{uuid.uuid4().hex}{ext}"

def _save_bytes_as_file(data: bytes, orig_filename: str) -> str:
    fname = _unique_filename(orig_filename)
    fpath = os.path.join(SAVE_DIR, fname)
    with open(fpath, "wb") as fh:
        fh.write(data)
    print(f"Saved image: {fpath}")    
    return fname

def _save_json_for_image(filename: str, payload: Dict[str, Any]) -> None:
    """
    Save analyzer payload (dict) as a JSON file next to the image.
    The JSON filename will be <filename> + '.json' (same basename).
    """
    try:
        base = os.path.splitext(filename)[0]
        jname = base + ".json"
        jpath = os.path.join(SAVE_DIR, jname)
        # Ensure JSON is serializable; use default=str for any odd objects
        with open(jpath, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2, default=str)
    except Exception:
        # Do not raise — logging would be preferable; we silently continue to avoid breaking requests
        pass
# -----------------------------------------------------


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
    Saves uploaded image(s) to disk (Images/) and includes saved filenames in response.
    Also writes a .json file per saved image with the analyzer result for later debugging.
    """
    # Collect bytes in order (small list, we keep in memory)
    frames: List[bytes] = []
    saved_filenames: List[Optional[str]] = []

    if files:
        for f in files:
            if f and f.filename:
                # read bytes (synchronous, same as previous behavior)
                data = _read_upload(f)
                frames.append(data)
                try:
                    saved_name = _save_bytes_as_file(data, f.filename)
                    saved_filenames.append(saved_name)
                except Exception:
                    saved_filenames.append(None)

    # Fallback to single-file if no 'files' provided
    if not frames and file:
        data = _read_upload(file)
        frames.append(data)
        try:
            saved_name = _save_bytes_as_file(data, file.filename)
            saved_filenames.append(saved_name)
        except Exception:
            saved_filenames.append(None)

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
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Analyzer returned unexpected result type.")

    # Build payload similar to before but include saved filenames
    payload = {
        "face_shape": result.get("face_shape"),
        "message": result.get("message"),
        # pass through any extra keys your analyzer adds
        **{k: v for k, v in result.items() if k not in {"face_shape", "message"}},
        "saved_images": saved_filenames
    }

    # Save analyzer JSON per saved image so you can debug later.
    # We write the same analyzer `payload` into each image's .json file for easy correlation.
    for fname in saved_filenames:
        if fname:
            try:
                _save_json_for_image(fname, payload)
            except Exception:
                # intentionally ignore failures to avoid breaking the response
                pass

    return JSONResponse(payload)


if __name__ == "__main__":
    # Bind to 0.0.0.0 for Docker
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
