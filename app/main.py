import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .schemas import EarringsRequest, EarringsResponse
from .suggester import suggest_for_face

load_dotenv()
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

app = FastAPI(title="VISA Earrings Suggestion API", version="0.1.0")

allow_origins = ["*"] if CORS_ORIGINS == "*" else [o.strip() for o in CORS_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/suggest/earrings", response_model=EarringsResponse)
def suggest_earrings(body: EarringsRequest):
    suggestions = suggest_for_face(body.face_shape)
    return EarringsResponse(face_shape=body.face_shape, suggestions=suggestions)