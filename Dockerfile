# ── Build stage ──────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps: FFmpeg + audio/video codecs, build tools, OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
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

# ── Pre-download MoViNet-A0 action recognition model (~20 MB) ─
RUN python -c "import tensorflow_hub as hub; model = hub.load('https://www.kaggle.com/models/google/movinet/TensorFlow2/a0-base-kinetics-600-classification/3'); print('MoViNet-A0 model cached successfully')" || echo 'MoViNet cache skipped (will download on first use)'

# ── Pre-download Kinetics-600 labels ───────────────────
RUN mkdir -p /app/app/data && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt', '/app/app/data/kinetics_600_labels.txt'); print('Kinetics-600 labels cached')" \
    || echo 'Labels cache skipped'

EXPOSE 8000

# Default: run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
