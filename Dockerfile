# --- Builder Stage ---
FROM python:3.10-slim-bullseye AS builder

# Install comprehensive dependencies for document processing
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libfontconfig1-dev \
    libfreetype6-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libleptonica-dev \
    libopencv-dev \
    python3-opencv \ 
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and use a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --- Final Runtime Stage ---
FROM python:3.10-slim-bullseye

WORKDIR /app

# Install ALL runtime dependencies for document processing
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    libfontconfig1 \
    libfreetype6 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    libwebp6 \
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    libopencv-imgcodecs4.5 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set up model cache directories - use standard HuggingFace locations
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets
ENV HF_HUB_OFFLINE=0

# Create cache directories with proper permissions
RUN mkdir -p $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE $HF_DATASETS_CACHE && \
    chmod -R 755 /app/.cache

# Download models during build (with HF_TOKEN if provided)
ARG HF_TOKEN_BUILD_ARG
ENV HF_TOKEN=${HF_TOKEN_BUILD_ARG}
ENV HF_HUB_OFFLINE=0

# Pre-download docling models to the cache
RUN python -c "\
import os; \
from huggingface_hub import snapshot_download; \
from pathlib import Path; \
print('Starting model download...'); \
print(f'HF_HOME: {os.environ.get(\"HF_HOME\")}'); \
print(f'HF_HUB_CACHE: {os.environ.get(\"HF_HUB_CACHE\")}'); \
print(f'HF_HUB_OFFLINE: {os.environ.get(\"HF_HUB_OFFLINE\")}'); \
download_path = snapshot_download( \
    repo_id='ds4sd/docling-models', \
    revision='v2.2.0', \
    cache_dir=os.environ.get('HF_HUB_CACHE'), \
    local_files_only=False, \
    token=os.environ.get('HF_TOKEN'), \
    force_download=False, \
    resume_download=True \
); \
print(f'Successfully downloaded docling models to: {download_path}'); \
cache_path = Path(os.environ.get('HF_HUB_CACHE')); \
file_count = len([f for f in cache_path.rglob('*') if f.is_file()]) if cache_path.exists() else 0; \
print(f'Total files in cache: {file_count}'); \
"

# Copy application code
COPY main.py .

# Create non-root user for security
RUN useradd -m appuser && \
    mkdir -p /tmp && \
    chown -R appuser:appuser /tmp /app
USER appuser

# Critical: Enable offline mode for runtime to use cached models
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCLING_DEVICE=cpu \
    HF_HUB_OFFLINE=1 \
    OFFLINE_MODE=1 \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    OPENCV_NUM_THREADS=8

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]