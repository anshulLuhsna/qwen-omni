# Makefile for Qwen2.5-Omni-7B AWQ Deployment
# Usage: make [target]

.PHONY: help setup serve test smoke docker-build docker-run clean

# Default target
help:
	@echo "Qwen2.5-Omni-7B AWQ Deployment Makefile"
	@echo "======================================="
	@echo "Available targets:"
	@echo "  make setup        - Install Python dependencies"
	@echo "  make serve        - Start FastAPI server"
	@echo "  make test         - Run API tests"
	@echo "  make smoke        - Run smoke test (model loading)"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make clean        - Clean cache and temp files"
	@echo ""
	@echo "Quick start:"
	@echo "  make setup && make serve"

# Install dependencies
setup:
	@echo "Installing Python dependencies..."
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
		hf-transfer \
		python-multipart \
		aiofiles \
		websockets
	@echo "Attempting Flash Attention installation..."
	-pip install packaging wheel
	-pip install flash-attn --no-build-isolation
	@echo "Setup complete!"

# Start server
serve:
	@echo "Starting Qwen2.5-Omni-7B AWQ server..."
	@echo "Server will be available at http://localhost:8000"
	@echo "Frontend at http://localhost:8000/frontend/"
	@echo "Press Ctrl+C to stop"
	uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

# Run tests
test:
	@echo "Running API tests..."
	@chmod +x run_local_tests.sh
	./run_local_tests.sh

# Run smoke test
smoke:
	@echo "Running smoke test..."
	python smoke_test.py

# Docker build
docker-build:
	@echo "Building Docker image..."
	docker build -t qwen-omni-awq:latest .
	@echo "Docker image built: qwen-omni-awq:latest"

# Docker run
docker-run:
	@echo "Running Docker container..."
	@echo "Server will be available at http://localhost:8000"
	docker run --gpus all \
		-p 8000:8000 \
		-v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
		--env HF_HUB_ENABLE_HF_TRANSFER=1 \
		qwen-omni-awq:latest

# Clean cache and temp files
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f qwen_out.wav qwen_ethan.wav test_response.json test_ethan.json
	@echo "Cleanup complete!"

# Advanced targets
.PHONY: dev gpu-check pre-download

# Development mode with auto-reload
dev:
	@echo "Starting in development mode with auto-reload..."
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Check GPU status
gpu-check:
	@echo "Checking GPU status..."
	@nvidia-smi || echo "nvidia-smi not found. Is NVIDIA driver installed?"
	@python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')" || echo "PyTorch not installed"

# Pre-download model to cache
pre-download:
	@echo "Pre-downloading model to cache..."
	@python -c "from huggingface_hub import snapshot_download; \
		snapshot_download('Qwen/Qwen2.5-Omni-7B-AWQ'); \
		print('Model downloaded successfully')"