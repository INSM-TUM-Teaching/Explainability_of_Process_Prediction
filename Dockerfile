FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal; expand if pip builds fail)
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential libgomp1 tini \
  && rm -rf /var/lib/apt/lists/*

# Install backend deps first for better layer caching (keeps Cloud Run image reasonable)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy only the runtime code needed by the backend (keeps build context small and cache stable)
COPY backend /app/backend
COPY conv_and_viz /app/conv_and_viz
COPY explainability /app/explainability
COPY gnns /app/gnns
COPY transformers /app/transformers
COPY utils /app/utils
COPY ppm_pipeline.py /app/ppm_pipeline.py

ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["tini", "--"]
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
