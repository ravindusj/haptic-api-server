# ── Build stage ──────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps: FFmpeg + audio/video codecs, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/haptic-jobs/uploads /tmp/haptic-jobs/results

# ── Pre-download YAMNet model (~18 MB, cached in image layer) ─
RUN python -c "import tensorflow_hub as hub; model = hub.load('https://tfhub.dev/google/yamnet/1'); print('YAMNet model cached successfully')"

# ── Pre-download faster-whisper tiny model (~75 MB) ──────
RUN python -c "from faster_whisper import WhisperModel; model = WhisperModel('tiny', device='cpu', compute_type='int8'); print('faster-whisper tiny model cached successfully')"

EXPOSE 8000

# Default: run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
