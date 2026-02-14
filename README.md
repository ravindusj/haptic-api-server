# Haptic Video Analyzer – API Server

> **Intelligent haptic pattern generator for videos.**
> Upload a video → AI + DSP analysis → AHAP haptic file.
> Detects bass, impacts, explosions, beats and **suppresses dialogue and ambient noise** for immersive vibration feedback.

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────────────┐
│  Mobile App  │────▶│  FastAPI Server (POST /api/v1/analyze)           │
│  (iOS)       │     │                                                  │
│              │◀────│  GET /api/v1/result/{job_id}  → .ahap download   │
└──────────────┘     └──────────┬──────────────────────────────────────-┘
                                │ Celery Task Queue
                                ▼
                     ┌──────────────────────────────────────┐
                     │         Analysis Pipeline            │
                     │                                      │
                     │  1. FFmpeg: Video → Audio (WAV)      │
                     │  2. librosa: DSP Feature Extraction  │
                     │     • RMS Energy (loudness)          │
                     │     • Onset Strength (transients)    │
                     │     • Low-Freq Energy (bass/rumble)  │
                     │     • Spectral Centroid (sharpness)  │
                     │     • Beat Tracking                  │
                     │  3. PANNs CNN14: AI Classification   │
                     │     • 527 AudioSet sound classes     │
                     │     • Speech detection & suppression │
                     │     • Impact/explosion detection     │
                     │  4. Score Fusion (DSP + AI)          │
                     │     • Dialogue suppression gate      │
                     │     • Silence gate                   │
                     │     • Sensitivity-based threshold    │
                     │  5. AHAP Generation                  │
                     │     • HapticTransient (taps)         │
                     │     • HapticContinuous (rumbles)     │
                     │     • ParameterCurves (smooth)       │
                     │     • Chunked for long content       │
                     └──────────────────────────────────────┘
```

## Key Innovation vs Sony DVS

Sony's Dynamic Vibration System vibrates to **all** audio, including dialogue. This system:

- **Suppresses dialogue** – AI detects speech and gates haptic output to near-zero
- **Suppresses ambient noise** – silence, wind, rain → no vibration
- **Amplifies impacts** – explosions, bass drops, gunshots → strong haptics
- **Beat-aware** – rhythmic taps aligned to musical beats
- **Immersive mapping** – bass → dull rumble, treble → sharp tap

## Quick Start

### Prerequisites

- Docker & Docker Compose
- (Or) Python 3.11+, FFmpeg, Redis

### Run with Docker

```bash
# 1. Copy env
cp .env.example .env

# 2. Build & start (API + Worker + Redis)
docker-compose up --build

# 3. Open Swagger docs
open http://localhost:8000/docs
```

### Run Locally (Development)

```bash
# 1. Create venv
python3 -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Start Redis
redis-server &

# 4. Start Celery worker
celery -A app.core.celery_app:celery_app worker --loglevel=info &

# 5. Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/analyze` | Upload video → returns `job_id` |
| `GET` | `/api/v1/status/{job_id}` | Poll processing status & progress |
| `GET` | `/api/v1/result/{job_id}` | Download `.ahap` file |
| `GET` | `/api/v1/result/{job_id}/info` | AHAP metadata (events, chunks, size) |
| `GET` | `/api/v1/preview/{job_id}` | JSON timeline preview |
| `GET` | `/api/v1/health` | Health check |

### Upload Example

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@my_video.mp4" \
  -F "sensitivity=0.6" \
  -F "style=cinematic" \
  -F "bass_boost=1.3"
```

Response:
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "queued",
  "message": "Job 'a1b2c3d4e5f6' queued. Poll GET /status/a1b2c3d4e5f6 for progress."
}
```

### Poll Status

```bash
curl http://localhost:8000/api/v1/status/a1b2c3d4e5f6
```

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "analyzing_dsp",
  "progress": 35.0,
  "file_name": "my_video.mp4",
  "duration_seconds": 120.5
}
```

### Download AHAP

```bash
curl -o haptics.ahap http://localhost:8000/api/v1/result/a1b2c3d4e5f6
```

## AHAP Output Format

The generated `.ahap` file follows Apple's Core Haptics JSON schema:

```json
{
  "Version": 1.0,
  "Pattern": [
    {
      "Event": {
        "Time": 2.345,
        "EventType": "HapticTransient",
        "EventParameters": [
          { "ParameterID": "HapticIntensity", "ParameterValue": 0.85 },
          { "ParameterID": "HapticSharpness", "ParameterValue": 0.15 }
        ]
      }
    },
    {
      "Event": {
        "Time": 5.120,
        "EventType": "HapticContinuous",
        "EventDuration": 1.5,
        "EventParameters": [
          { "ParameterID": "HapticIntensity", "ParameterValue": 0.70 },
          { "ParameterID": "HapticSharpness", "ParameterValue": 0.10 }
        ]
      }
    }
  ]
}
```

- **HapticTransient** → sharp tap (impacts, beats, gunshots)
- **HapticContinuous** → sustained rumble (bass, explosions, engines)
- **Low sharpness** (0.1) → deep, dull vibration (bass)
- **High sharpness** (0.9) → sharp, crisp tap (snare, click)

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `SENSITIVITY` | `0.5` | Global threshold (0=selective, 1=permissive) |
| `BASS_BOOST` | `1.0` | Low-freq energy multiplier |
| `SPEECH_SUPPRESSION_FACTOR` | `0.05` | Haptic output during speech (near-zero) |
| `MAX_UPLOAD_SIZE_MB` | `500` | Max video file size |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |

## Project Structure

```
haptic-api-server/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── api/
│   │   └── routes.py              # API endpoints
│   ├── core/
│   │   ├── config.py              # Settings (pydantic-settings)
│   │   └── celery_app.py          # Celery worker + pipeline task
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   └── services/
│       ├── audio_extractor.py     # FFmpeg video → WAV
│       ├── dsp_analyzer.py        # librosa DSP features
│       ├── ai_classifier.py       # PANNs CNN14 sound classification
│       ├── haptic_scorer.py       # DSP + AI fusion → haptic scores
│       └── ahap_generator.py      # Score timeline → AHAP JSON
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## AWS Deployment (EC2)

```bash
# 1. Launch EC2 instance (t3.large minimum: 2 vCPU, 8 GB RAM)
# 2. Install Docker & Docker Compose
# 3. Clone repo & configure .env
# 4. Run
docker-compose up -d --build

# 5. (Optional) Put behind ALB with HTTPS
```

## Technologies

| Component | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Task Queue | Celery + Redis |
| DSP Analysis | librosa, scipy |
| AI Classification | PANNs CNN14 (PyTorch) |
| Audio Extraction | FFmpeg |
| Containerisation | Docker + Docker Compose |
| Haptic Format | Apple AHAP (Core Haptics) |

## License

This project is part of a final-year research project at NSBM Green University.
