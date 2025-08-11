# VISA Face API — Clean Backend

FastAPI backend that:
- accepts an **image** upload (`file`) at `POST /detect/face-shape`
- computes face metrics with MediaPipe and classifies into one of:
  `oval | round | square | rectangle | diamond | heart`
- returns only: `face_shape`, `message`, `rasa`, `celestial`, `code`
- **No product data** — the frontend fetches Shopify collection items itself.

## Endpoints
- `GET /health` → status, messages loaded
- `POST /detect/face-shape` → upload JPEG/PNG via multipart field **file**

### Response example
```json
{
  "face_shape": "oval",
  "message": "Your Style Signature is Grace inspired by Lyra — ...",
  "rasa": "Grace",
  "celestial": "Lyra",
  "code": "VISA-OVAL-01"
}
```

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional: set allowed origins
echo 'CORS_ALLOW_ORIGINS=https://YOUR-SHOP.myshopify.com' > .env

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Test:
```bash
curl -F "file=@/path/to/selfie.jpg" http://localhost:8000/detect/face-shape
```

## Docker

```bash
docker compose up -d --build
```

## Configure messages
Edit `app/data/messages.json` with your **After Detection Message**, **rasa**, **celestial body**, and **internal code** for each face type.
