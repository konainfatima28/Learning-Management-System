# Dockerfile — robust for streamlit-webrtc + MediaPipe + ffmpeg (stable Debian)
FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- Install system deps (with retry and safer package list) ----
# - use --allow-releaseinfo-change fallback for apt metadata changes
# - removed libavresample-dev (often missing) and libavdevice-dev (not always needed)
RUN set -eux; \
    apt-get update || (apt-get update --allow-releaseinfo-change -y); \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        gnupg \
        build-essential \
        pkg-config \
        git \
        cmake \
        libffi-dev \
        libssl-dev \
        python3-dev \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libavfilter-dev \
        libxrender1 \
        libsm6 \
        libx11-6 \
        libxcb1 \
    ; \
    # cleanup apt cache
    rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip & core python build tools ----
RUN pip install --upgrade pip setuptools wheel

# ---- Copy requirements and install python deps ----
COPY requirements.txt /app/requirements.txt
# Recommend pinning versions in requirements.txt; these are example tolerant options.
RUN pip install -r /app/requirements.txt

# ---- Copy app code ----
COPY . /app

# Create a non-root user for safety (optional but recommended)
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Start Streamlit (headless)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false"]
