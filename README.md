# VISA Earrings Suggestion API (FastAPI + Docker)

This is a minimal, production-ready API that exposes **POST /suggest/earrings**.
It returns 5 earring suggestions based on a provided **face_shape**.

## Endpoints
- `GET /health` → returns `{ "status": "ok" }`
- `POST /suggest/earrings` → body: `{ "face_shape": "oval" }`

## Quick Start (on your Droplet)
```bash
# 1) Upload and unzip this folder on your server (e.g., /opt/visa-earring-api)
unzip visa-earring-api.zip -d /opt/
cd /opt/visa-earring-api

# 2) Create your .env
cp .env.example .env
# then edit .env to set SHOPIFY_DOMAIN and CORS_ORIGINS

# 3) Start with Docker
docker compose up -d --build

# 4) Test
curl http://YOUR_DROPLET_IP:8000/health
curl -X POST http://YOUR_DROPLET_IP:8000/suggest/earrings -H "Content-Type: application/json" -d '{"face_shape":"oval"}'
```

## Environment Variables
Create `.env` in the project root:
```
SHOPIFY_DOMAIN=https://q51zdw-if.myshopify.com/
CORS_ORIGINS=https://q51zdw-if.myshopify.com,http://localhost:3000
PORT=8000
```