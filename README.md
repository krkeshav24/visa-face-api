# VISA Earrings Suggestion API (FastAPI + Docker)

This is a production-ready API for **VISA** (Visual Identity & Style Alignment).  
It provides two main features:

1. **Face Shape Detection** using Google Face Mesh (server-side, hidden logic)  
2. **Earring Suggestions** based on the detected face shape

---

## Endpoints

### 1. Health Check
`GET /health`  
Returns:
```json
{ "status": "ok" }
