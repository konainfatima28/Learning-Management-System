# Dockerfile — adapted for streamlit-webrtc + MediaPipe + ffmpeg
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps for ffmpeg, mediapipe and building pyav
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    git \
    wget \
    ca-certificates \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libavresample-dev \
    libxrender1 \
    libsm6 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache)
COPY requirements.txt /app/requirements.txt

# Upgrade pip & friends and preinstall build tools that pyav needs:
RUN pip install --upgrade pip setuptools wheel

# Install a Cython version compatible with building pyav from source
# (prevents the 'noexcept' Cython compile errors).
# If you prefer a stable release, change version to a pinned stable you tested.
RUN pip install "Cython>=3.0.0"

# Optionally disable av logging callback code during build to avoid some compile paths:
ENV PYAV_LOGGING=off

# Now install Python deps (this will build av if no wheel available)
RUN pip install -r /app/requirements.txt

# Copy project files
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false"]
