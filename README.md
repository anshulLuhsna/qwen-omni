# Qwen2.5-Omni-7B AWQ RunPod Deployment Guide

## Prerequisites
- RunPod account with credits
- Local terminal with curl installed

## Step 1: Create RunPod Pod

1. Go to RunPod Console → Community Cloud
2. Select GPU: **RTX A5000 24GB** (or L4 24GB as fallback)
3. Configure:
   - Container Image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
   - Disk Size: 50-100GB
   - Expose HTTP Port: 8000
4. Deploy Pod and wait for it to start

## Step 2: Initial Setup (Web Terminal)

```bash
# Verify GPU
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install system dependencies
apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    ninja-build \
    wget \
    curl

# Set environment variables
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Add to .bashrc for persistence
echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" >> ~/.bashrc
source ~/.bashrc
```

## Step 3: Install Python Dependencies

### Option A: With Flash Attention (Recommended)

```bash
# Install base packages
pip install --upgrade pip
pip install transformers==4.52.3 \
    autoawq==0.2.9 \
    accelerate \
    qwen-omni-utils \
    fastapi \
    uvicorn \
    "pydantic<2" \
    soundfile \
    librosa \
    torch \
    huggingface-hub \
    hf-transfer

# Attempt Flash Attention 2 installation
pip install packaging wheel
pip install flash-attn --no-build-isolation

# If flash-attn fails, continue without it (see Option B)
```

### Option B: Fallback (No Flash Attention)

If Flash Attention installation fails with build errors:

```bash
# Skip flash-attn, use standard attention
echo "Flash Attention build failed, using standard attention"
# All other packages already installed from Option A
```

## Step 4: Download Model (Optional Pre-download)

```bash
# Pre-download AWQ model to cache (optional, will auto-download on first use)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-Omni-7B-AWQ', 
                   cache_dir='/root/.cache/huggingface')
print('Model pre-downloaded successfully')
"
```

## Step 5: Create Application Files

```bash
# Create app directory
mkdir -p /workspace/qwen-app
cd /workspace/qwen-app

# Create app.py (copy from deliverables)
cat > app.py << 'EOF'
# [Insert app.py content here]
EOF

# Create frontend directory
mkdir -p frontend
# Create frontend/index.html and frontend/main.js (copy from deliverables)
```

## Step 6: Start Server

```bash
# Start FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

## Step 7: Get RunPod Proxy URL

Your RunPod proxy URL format:
```
https://<POD_ID>-8000.proxy.runpod.net
```

Find your POD_ID in RunPod dashboard under "Pod Details" → "Pod ID"

Example: If POD_ID is `abc123def456`, your URL is:
```
https://abc123def456-8000.proxy.runpod.net
```

## Step 8: Test Endpoints

```bash
# Set your proxy URL
export RUNPOD_URL="https://<POD_ID>-8000.proxy.runpod.net"

# Test health endpoint
curl $RUNPOD_URL/healthz

# Test chat endpoint (text only)
curl -X POST $RUNPOD_URL/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?", "voice": "Cherry"}' \
  -o response.json

# Extract and save audio
python -c "
import json, base64
with open('response.json') as f:
    data = json.load(f)
    audio_bytes = base64.b64decode(data['audio_wav_base64'])
    with open('output.wav', 'wb') as out:
        out.write(audio_bytes)
print('Audio saved to output.wav')
"

# Play audio (if available)
ffplay output.wav
```

## Step 9: Access Frontend

Open in browser:
```
https://<POD_ID>-8000.proxy.runpod.net/frontend/
```

## Troubleshooting

### Out of Memory (OOM)
- Ensure using AWQ quantized model (not full precision)
- Clear CUDA cache: `torch.cuda.empty_cache()`
- Reduce max_new_tokens in generation config
- Restart pod if persistent

### Flash Attention Build Fails
- Use fallback option (standard attention)
- Model will work without flash-attn, just slightly slower
- No code changes needed, auto-detects availability

### CORS Issues
- Server has CORS enabled with `allow_origins=["*"]`
- If issues persist, check browser console for specific errors
- Try incognito mode to avoid cached headers

### Large Audio Inputs
- Maximum audio length: 30 seconds (configurable)
- Chunking for longer audio not implemented in base version
- Consider preprocessing audio to 24kHz mono before sending

### Voice Names
- Valid voices: "Cherry" (default), "Ethan", "Serena", "Chelsie"
- Case-sensitive in current implementation
- Invalid voice falls back to "Cherry"

### RunPod Proxy Issues
- Ensure port 8000 is exposed in pod settings
- Check pod is running (green status)
- Verify URL format: `https://<POD_ID>-8000.proxy.runpod.net`
- No trailing slash after port number

## Performance Optimization

### Memory Management
- AWQ INT4 uses ~8GB VRAM for model weights
- Keep context length under 2048 tokens for stability
- Monitor with `nvidia-smi` during inference

### Latency Optimization
- First inference is slow (model loading)
- Subsequent calls should have TTFT < 1 second
- Keep model loaded between requests (FastAPI handles this)

### Scaling to Other GPUs
- **L4 24GB**: Similar performance, may need reduced batch size
- **RTX 4090 24GB**: Consumer card, excellent performance
- **L40S 48GB**: Can run larger batches or FP16 version

## Advanced: WebSocket Streaming (Optional)

For real-time bidirectional audio streaming, add WebSocket endpoint:

```python
# Add to app.py
@app.websocket("/ws_chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Implementation for streaming audio chunks
```

## Environment Variables

Create `.env` file:
```bash
HF_TOKEN=<your_huggingface_token>  # Optional, for gated models
HF_HUB_ENABLE_HF_TRANSFER=1
CUDA_VISIBLE_DEVICES=0
```

Load in app:
```bash
export $(cat .env | xargs)
```

## Quick Commands Reference

```bash
# Check GPU memory
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi

# Check model cache
du -sh ~/.cache/huggingface/

# Clear Python cache
pip cache purge

# Restart server
pkill uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```