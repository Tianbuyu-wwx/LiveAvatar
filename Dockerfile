# LiveAvatar inference + serving image (CUDA).
# Licensed under AGPL-3.0-or-later; commercial use requires a separate
# written license from LiveAvatar Contributors (see SECURITY.md and LICENSE).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps: torch CUDA first, then the package with extras.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

COPY pyproject.toml ./
COPY src/ ./src/
# transformers/einops below are for the vendored GPT-SoVITS TTS engine
# (the MuseTalk inference path itself needs neither).
RUN pip install --no-deps . \
    && pip install "fastapi>=0.110" "uvicorn>=0.29" \
        "opencv-python" "transformers" "einops" \
        "numpy" "pydantic>=2" "pydantic-settings>=2" "huggingface_hub" \
        "torchaudio" "librosa" "fast_langdetect"

# Demo page + scripts.
COPY web/ ./web/
COPY scripts/ ./scripts/

EXPOSE 8000
CMD ["uvicorn", "liveavatar.publish:app", "--host", "0.0.0.0", "--port", "8000"]
