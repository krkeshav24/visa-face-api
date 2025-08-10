# VISA Face API — Full Server (Face Shape + Earring Suggestions)

This is a production-ready FastAPI backend skeleton for your VISA project.
It includes:
- `/detect-face-shape` — accepts an uploaded image, runs MediaPipe FaceMesh, computes H/FW/CW/JW + ratios, and classifies one of 6 shapes.
- `/suggest/earrings?shape=oval` — returns 5 earring suggestions for a given face shape (editable in `data/suggestions.json`).
- CORS restricted to `https://q51zdw-if.myshopify.com` plus localhost for development.

## Quick start (local)
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# then open http://127.0.0.1:8000/docs
```

## With Docker
```bash
docker build -t visa-face-api .
docker run -p 8000:8000 visa-face-api
```

## Deploy (DigitalOcean App Platform)
- Push this repo to GitHub.
- Create App from repo. It will detect the Dockerfile automatically.
- Set HTTP port: 8000 (if prompted).
- Deploy.

## Notes
- The classification is heuristic (rules on measured ratios). Tune thresholds in `app/face.py` to match your VISA model.
- `suggestions.json` is a simple editable mapping; replace image and product URLs with real ones.
