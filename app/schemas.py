from pydantic import BaseModel, Field, constr
from typing import List

FaceShapeStr = constr(strip_whitespace=True, to_lower=True)

class EarringsRequest(BaseModel):
    face_shape: FaceShapeStr = Field(..., description="Detected face shape.")

class Suggestion(BaseModel):
    id: str
    title: str
    product_url: str
    image_url: str
    reason: str

class EarringsResponse(BaseModel):
    face_shape: str
    suggestions: List[Suggestion]