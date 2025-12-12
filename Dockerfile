# Dockerfile — adapted for streamlit-webrtc + MediaPipe + ffmpeg
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system deps for ffmpeg, mediapipe and building pyav
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

# Copy files
COPY requirements.txt /app/requirements.txt
# Upgrade pip then install Python deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# Copy code
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Use a safe start command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false"]
