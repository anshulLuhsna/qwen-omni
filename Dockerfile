# GPU-ready base image with CUDA 12.1 and PyTorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Build arguments for optional features
ARG ENABLE_FLASH_ATTN=1
ARG HF_TOKEN=""

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_TOKEN=${HF_TOKEN}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    ninja-build \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Conditional Flash Attention installation
RUN if [ "$ENABLE_FLASH_ATTN" = "1" ]; then \
        echo "Attempting Flash Attention 2 installation..." && \
        pip install --no-cache-dir packaging wheel && \
        pip install --no-cache-dir flash-attn --no-build-isolation || \
        echo "Flash Attention installation failed, continuing without it"; \
    else \
        echo "Flash Attention disabled by build arg"; \
    fi

# Pre-download model (optional, comment out to download on first use)
# RUN python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('Qwen/Qwen2.5-Omni-7B-AWQ', cache_dir='/root/.cache/huggingface')"

# Copy application files
COPY app.py .
COPY frontend/ ./frontend/

# Create directories for audio processing
RUN mkdir -p /tmp/audio_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]