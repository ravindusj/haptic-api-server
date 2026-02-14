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

# ── Pre-download PANNs CNN14 model weights ───────────────
# This ~300MB download is cached in the Docker image layer
RUN echo "try:" > /tmp/download_panns.py && \
    echo "    from panns_inference import AudioTagging" >> /tmp/download_panns.py && \
    echo "    at = AudioTagging(device='cpu')" >> /tmp/download_panns.py && \
    echo "    print('PANNs model downloaded successfully')" >> /tmp/download_panns.py && \
    echo "except Exception as e:" >> /tmp/download_panns.py && \
    echo "    print(f'PANNs download skipped: {e}')" >> /tmp/download_panns.py && \
    python /tmp/download_panns.py || true

EXPOSE 8000

# Default: run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
