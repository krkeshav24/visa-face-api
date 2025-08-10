from typing import Tuple
from io import BytesIO
try:
    from PIL import Image
except Exception:
    Image = None

# Placeholder heuristic. Replace with MediaPipe/OpenCV.
def classify_face_shape_from_image(image_bytes: bytes) -> Tuple[str, float]:
    if Image is None:
        return "oval", 0.5
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        if h / max(1, w) > 1.3:
            return "oblong", 0.55
        if w / max(1, h) > 1.2:
            return "square", 0.55
        if 0.95 <= h / max(1, w) <= 1.05:
            return "round", 0.55
        return "oval", 0.55
    except Exception:
        return "oval", 0.5
