#!/usr/bin/env python3
"""
Qwen2.5-Omni-7B AWQ FastAPI Server
Serves text and audio chat with TTS capabilities
"""

import sys
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

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app")

# -----------------------------------------------------------------------------
# CUDA / Torch prefs
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    logger.info(f"CUDA available. Device: {torch.cuda.get_device_name(0)}")
    # allow TF32 for perf on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    logger.warning("CUDA not available! Running on CPU will be very slow.")

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
MODEL_ID_AWQ = "Qwen/Qwen2.5-Omni-7B-AWQ"   # preferred (quantized)
MODEL_ID_FP16 = "Qwen/Qwen2.5-Omni-7B"      # fallback (full precision)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
SAMPLE_RATE = 24000  # 24kHz audio

# Globals
model = None
processor = None

# -----------------------------------------------------------------------------
# API schemas
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text input")
    audio_wav_base64: Optional[str] = Field(None, description="Base64 encoded WAV audio")
    voice: str = Field("Cherry", description="Voice for TTS: Cherry, Ethan, Serena, or Chelsie")

class ChatResponse(BaseModel):
    text: str
    audio_wav_base64: str

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Qwen2.5-Omni-7B AWQ API",
    description="Text and audio chat with TTS",
    version="1.0.0"
)

# Serve simple frontend if present
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Model loading with AWQ -> FP16 fallback
# -----------------------------------------------------------------------------
def _load_qwen(model_id: str):
    """Internal: load a Qwen Omni model by id with sane defaults."""
    from transformers import Qwen2_5OmniForConditionalGeneration
    logger.info(f"Loading model: {model_id}")

    # NOTE:
    # - Use float16 for both AWQ and FP16 paths (AWQ kernels do not support bfloat16).
    # - Do NOT pass custom AwqConfig; let the repo’s own quantization_config (if any) apply.
    return Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

def load_model():
    """Load Qwen2.5-Omni model; try AWQ first, then fall back to FP16 if AWQ fails."""
    global model, processor
    try:
        # Try AWQ repo first
        model = _load_qwen(MODEL_ID_AWQ)
        logger.info("Model loaded successfully (AWQ).")
    except AssertionError as e:
        # Canonical AWQ replacement failure (group_size / in_features assertion)
        logger.warning(f"AWQ assertion during load ({e}). Falling back to FP16 model.")
        model = _load_qwen(MODEL_ID_FP16)
        logger.info("Model loaded successfully (FP16 fallback).")
    except Exception as e:
        # Any other failure -> try FP16 as a best-effort fallback
        logger.warning(f"AWQ load failed ({type(e).__name__}: {e}). Trying FP16 fallback...")
        try:
            model = _load_qwen(MODEL_ID_FP16)
            logger.info("Model loaded successfully (FP16 fallback).")
        except Exception:
            logger.error("FP16 fallback also failed.")
            raise

    # Processor is shared for both
    from transformers import Qwen2_5OmniProcessor
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_tts_system_message(voice: str = "Cherry") -> str:
    voice_map = {
        "Cherry": "Cherry",
        "Ethan": "Ethan",
        "Serena": "Serena",
        "Chelsie": "Chelsie",
    }
    selected_voice = voice_map.get(voice, "Cherry")
    return (
        "You are a helpful assistant with voice capabilities. "
        f"When generating speech, use the {selected_voice} voice style. "
        "Respond naturally and conversationally."
    )

def _placeholder_tts(text: str) -> np.ndarray:
    """Generate a short beep so the endpoint returns WAV without depending on model audio head."""
    duration = min(max(len(text.split()) * 0.25, 0.5), 5.0)
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return y

# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        traceback.print_exc()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/healthz")
def health_check():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global model, processor

    # Ensure model
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    try:
        text_input = request.text or ""
        audio_input = None
        audio_data = None

        if request.audio_wav_base64:
            try:
                audio_bytes = base64.b64decode(request.audio_wav_base64)
                audio_input = BytesIO(audio_bytes)
                audio_data, sr = sf.read(audio_input)
                if sr != SAMPLE_RATE:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                raise HTTPException(status_code=400, detail="Invalid audio data")

        # System message
        system_msg = get_tts_system_message(request.voice)

        # Messages -> chat template
        messages = [{"role": "system", "content": system_msg}]
        if text_input:
            messages.append({"role": "user", "content": text_input})
        elif audio_data is not None:
            messages.append({"role": "user", "content": "Process audio input"})
        else:
            raise HTTPException(status_code=400, detail="No input provided")

        text_prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Build inputs
        if audio_data is not None:
            inputs = processor(
                text=text_prompt,
                audios=[audio_data],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).to(DEVICE)
        else:
            inputs = processor(
                text=text_prompt,
                return_tensors="pt"
            ).to(DEVICE)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # Decode text
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        text_response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Placeholder audio (replace with model audio output later if desired)
        audio_response = _placeholder_tts(text_response)

        # Encode audio
        audio_buf = BytesIO()
        sf.write(audio_buf, audio_response, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        audio_base64 = base64.b64encode(audio_buf.getvalue()).decode("utf-8")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ChatResponse(text=text_response, audio_wav_base64=audio_base64)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws_chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            request = ChatRequest(**data)
            # Reuse the same chat logic via direct call
            resp = chat(request)
            await websocket.send_json(resp.dict())
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/")
def root():
    return {
        "name": "Qwen2.5-Omni-7B AWQ API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/healthz",
            "chat": "/chat",
            "frontend": "/frontend/",
            "websocket": "/ws_chat",
        },
        "model_pref": MODEL_ID_AWQ,
        "fallback": MODEL_ID_FP16,
        "device": DEVICE,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
