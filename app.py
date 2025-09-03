#!/usr/bin/env python3
"""
Qwen2.5-Omni-7B FastAPI Server
Serves text and audio chat with TTS capabilities
"""

import os

# Force HF cache inside your mounted volume
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/transformers"

import base64
import logging
import traceback
from typing import Optional
import torch
import numpy as np
import soundfile as sf
from io import BytesIO

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CUDA check
if not torch.cuda.is_available():
    logger.warning("CUDA not available! Running on CPU will be very slow.")
else:
    logger.info(f"CUDA available. Device: {torch.cuda.get_device_name(0)}")

# Model configuration
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"   # ✅ Full precision model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
SAMPLE_RATE = 24000  # 24 kHz audio

# Globals
model = None
processor = None
tokenizer = None

# API request/response models
class ChatRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text input")
    audio_wav_base64: Optional[str] = Field(None, description="Base64 encoded WAV audio")
    voice: str = Field("Cherry", description="Voice for TTS: Cherry, Ethan, Serena, Chelsie")

class ChatResponse(BaseModel):
    text: str
    audio_wav_base64: str

# FastAPI app
app = FastAPI(
    title="Qwen2.5-Omni-7B API",
    description="Text and audio chat with TTS",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional static frontend
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# -------------------------
# Model loader
# -------------------------
def load_model():
    """Load Qwen2.5-Omni-7B FP16"""
    global model, processor, tokenizer
    logger.info(f"Loading model: {MODEL_ID}")

    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,   # ✅ FP16
            attn_implementation="sdpa",  # use PyTorch SDPA kernels
            trust_remote_code=True
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
        tokenizer = processor.tokenizer

        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        raise

# -------------------------
# Helper: TTS system prompt
# -------------------------
def get_tts_system_message(voice: str = "Cherry") -> str:
    voices = {"Cherry": "Cherry", "Ethan": "Ethan", "Serena": "Serena", "Chelsie": "Chelsie"}
    selected = voices.get(voice, "Cherry")
    return f"You are a helpful assistant with voice capabilities. Use the {selected} voice style."

# -------------------------
# Startup hook
# -------------------------
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        logger.error(f"Startup model load failed: {e}")

@app.get("/healthz")
async def health_check():
    return {"ok": True}

# -------------------------
# Main chat endpoint
# -------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global model, processor, tokenizer
    if model is None:
        load_model()

    try:
        text_input = request.text or ""
        audio_input = None

        # Handle audio input
        if request.audio_wav_base64:
            try:
                audio_bytes = base64.b64decode(request.audio_wav_base64)
                audio_input = BytesIO(audio_bytes)
                audio_data, sr = sf.read(audio_input)
                if sr != SAMPLE_RATE:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            except Exception as e:
                logger.error(f"Audio error: {e}")
                raise HTTPException(status_code=400, detail="Invalid audio data")

        # Conversation prompt
        system_msg = get_tts_system_message(request.voice)
        messages = [{"role": "system", "content": system_msg}]
        if text_input:
            messages.append({"role": "user", "content": text_input})
        elif audio_input is not None:
            messages.append({"role": "user", "content": "Process audio input"})
        else:
            raise HTTPException(status_code=400, detail="No input provided")

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Inputs
        if audio_input is not None:
            inputs = processor(text=text, audios=[audio_data], sampling_rate=SAMPLE_RATE,
                               return_tensors="pt").to(DEVICE)
        else:
            inputs = processor(text=text, return_tensors="pt").to(DEVICE)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        text_response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Generate audio (placeholder sine wave for now)
        audio_response = generate_audio_from_text(text_response, request.voice)
        buf = BytesIO()
        sf.write(buf, audio_response, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        audio_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ChatResponse(text=text_response, audio_wav_base64=audio_base64)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Dummy TTS generator
# -------------------------
def generate_audio_from_text(text: str, voice: str = "Cherry") -> np.ndarray:
    try:
        duration = min(len(text.split()) * 0.5, 10)
        samples = int(duration * SAMPLE_RATE)
        t = np.linspace(0, duration, samples)
        freq = 440
        audio = 0.3 * np.sin(2 * np.pi * freq * t)

        for i, char in enumerate(text[:20]):
            freq_mod = 440 + (ord(char) % 12) * 20
            audio += 0.1 * np.sin(2 * np.pi * freq_mod * t)

        audio = audio / np.max(np.abs(audio))
        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return np.zeros(SAMPLE_RATE, dtype=np.float32)

# -------------------------
# WebSocket endpoint
# -------------------------
@app.websocket("/ws_chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            request = ChatRequest(**data)
            response = await chat(request)
            await websocket.send_json({"text": response.text, "audio_wav_base64": response.audio_wav_base64})
    except Exception as e:
        logger.error(f"WS error: {e}")
        await websocket.close()

@app.get("/")
async def root():
    return {
        "name": "Qwen2.5-Omni-7B API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/healthz",
            "chat": "/chat",
            "frontend": "/frontend/",
            "websocket": "/ws_chat",
        },
        "model": MODEL_ID,
        "device": DEVICE,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
