import os
from typing import List, Dict
from .schemas import Suggestion

SHOPIFY_DOMAIN = os.getenv("SHOPIFY_DOMAIN", "https://q51zdw-if.myshopify.com/")

STYLE_RULES: Dict[str, List[Dict]] = {
    "oval": [
        {"id": "oval-drop-1", "title": "Elongated Teardrop", "slug": "elongated-teardrop-earrings", "reason": "Drops accentuate natural symmetry."},
        {"id": "oval-hoop-1", "title": "Medium Open Hoops", "slug": "medium-open-hoops", "reason": "Open hoops keep balance and add lift."},
        {"id": "oval-stud-1", "title": "Facet Studs", "slug": "facet-studs", "reason": "Compact sparkle keeps proportions clean."},
        {"id": "oval-dangle-1", "title": "Tapered Dangles", "slug": "tapered-dangles", "reason": "Taper echoes cheek contour."},
        {"id": "oval-geo-1", "title": "Soft-Edge Geometric", "slug": "soft-geo", "reason": "Gentle geometry adds interest."}
    ]
}

def to_url(slug: str) -> str:
    return SHOPIFY_DOMAIN.rstrip("/") + "/products/" + slug.strip("/")

def to_image(slug: str) -> str:
    return SHOPIFY_DOMAIN.rstrip("/") + "/cdn/shop/files/" + slug.replace('-', '_') + ".jpg"

def suggest_for_face(face_shape: str) -> List[Suggestion]:
    key = face_shape.lower().strip()
    rules = STYLE_RULES.get(key) or STYLE_RULES["oval"]
    out: List[Suggestion] = []
    for r in rules[:5]:
        out.append(Suggestion(
            id=r["id"],
            title=r["title"],
            product_url=to_url(r["slug"]),
            image_url=to_image(r["slug"]),
            reason=r["reason"]
        ))
    return out