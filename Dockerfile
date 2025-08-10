FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install gunicorn uvicorn[standard]
COPY app ./app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["gunicorn","-k","uvicorn.workers.UvicornWorker","app.main:app","--bind","0.0.0.0:8000","--workers","2","--timeout","120"]
