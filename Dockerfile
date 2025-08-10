# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional, kept slim)
RUN apt-get update && apt-get install -y --no-install-recommends     && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app ./app
COPY .env.example ./
COPY README.md ./

# Expose port
EXPOSE 8000

# Start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
