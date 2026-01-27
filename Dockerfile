FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal; expand if pip builds fail)
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install backend deps first for better layer caching (keeps Cloud Run image reasonable)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the full repo (backend imports project modules from repo root)
COPY . /app

ENV PORT=8080
EXPOSE 8080

CMD ["bash", "-lc", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
