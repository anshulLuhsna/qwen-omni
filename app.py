#!/usr/bin/env python3
"""
Qwen2.5-Omni-7B FastAPI Server (non-quantized)
- Text in -> Text + real speech (24 kHz) out
- Uses official Qwen2.5-Omni Transformers API
"""

import os
import base64
import logging
import traceback
from io import BytesIO
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("app")

# -----------------------------------------------------------------------------
# Hugging Face cache on the mounted volume (so weights persist)
# -----------------------------------------------------------------------------
HF_HOME = os.environ.get("HF_HOME", "/workspace/.cache/huggingface")
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

# -----------------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"   # non-quantized
SAMPLE_RATE = 24_000                # model outputs 24 kHz audio (per model card)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9

logger.info("CUDA available? %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("GPU: %s", torch.cuda.get_device_name(0))

# Globals
model = None
processor = None

# -----------------------------------------------------------------------------
# API types
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text input")
    # optional: accept a base64 WAV from client in the future
    audio_wav_base64: Optional[str] = Field(None, description="Optional user audio (WAV, base64)")
    voice: str = Field("Cherry", description="Voice style hint (prompted via system message)")

class ChatResponse(BaseModel):
    text: str
    audio_wav_base64: str  # 24 kHz mono WAV (PCM 16) in base64

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Qwen2.5-Omni-7B API",
              description="Text → Text + Speech (24 kHz)",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# -----------------------------------------------------------------------------
# Model loader (official Transformers path)
# -----------------------------------------------------------------------------
def load_model():
    """
    Load non-quantized Qwen2.5-Omni-7B with the official processor.
    Follows the model card pattern: Processor.apply_chat_template -> process_mm_info -> model.generate -> (text_ids, audio)
    """
    global model, processor

    if model is not None and processor is not None:
        return

    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info

        logger.info("Loading %s ...", MODEL_ID)

        # Prefer FlashAttention2 if present, otherwise SDPA is fine.
        attn_impl = None
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            logger.info("flash-attn detected: using flash_attention_2")
        except Exception:
            logger.info("flash-attn not available: using SDPA")

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attn_impl,   # None -> SDPA; else FA2
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error("Failed to load model: %s", e)
        traceback.print_exc()
        raise

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _tts_system_message(voice: str) -> str:
    # The model card doesn’t define a hard parameter for voice; we nudge via system prompt.
    return (f"You are a helpful assistant with voice capabilities. "
            f"When generating speech, use a natural '{voice}' voice style. "
            f"Respond clearly and conversationally.")

def _build_inputs(text_prompt: str):
    """Builds inputs for text-only chat but via the official multimodal path."""
    from qwen_omni_utils import process_mm_info  # local import to avoid top-level dep if unused
    from transformers import Qwen2_5OmniProcessor

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": text_prompt}],
        },
        # Actual user text will be appended by caller as role=user
    ]

    # Convert to model text & (empty) multimodal structures
    chat_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

    # Build tensor inputs; even for pure text we pass the multimodal kwargs (they’ll be empty lists)
    inputs = processor(
        text=chat_text,
        audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False
    )
    return inputs.to(model.device).to(model.dtype)

def _gen_speech_from_messages(messages: list):
    """
    messages: chat turns as dicts with 'role' and 'content' (list of {'type','text'} etc.)
    Returns: (text_out, audio_numpy)
    """
    from qwen_omni_utils import process_mm_info

    # Template → tensors (official path)
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False
    ).to(model.device).to(model.dtype)

    # The official API returns (text_ids, audio)
    # See model card “Transformers Usage”. :contentReference[oaicite:1]{index=1}
    text_ids, audio = model.generate(**inputs)

    # Decode text
    text_out = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Audio → numpy (24 kHz per model card) :contentReference[oaicite:2]{index=2}
    audio_np = audio.reshape(-1).detach().cpu().numpy()
    return text_out, audio_np

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def _startup():
    try:
        load_model()
    except Exception as e:
        logger.error("Startup load failed: %s", e)

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Text → Text + Speech (WAV@24kHz base64).
    """
    global model, processor
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    if not req.text and not req.audio_wav_base64:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'audio_wav_base64'.")

    try:
        # Build conversation
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _tts_system_message(req.voice)}]},
        ]
        if req.text:
            messages.append({"role": "user", "content": [{"type": "text", "text": req.text}]})
        else:
            # (Optional) handle user audio in future — current path is text→speech
            messages.append({"role": "user", "content": [{"type": "text", "text": "(User sent audio)"}]})

        text_out, audio_np = _gen_speech_from_messages(messages)

        # Encode WAV (PCM_16) at 24kHz
        buf = BytesIO()
        sf.write(buf, audio_np, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ChatResponse(text=text_out, audio_wav_base64=audio_b64)

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=507, detail="CUDA OOM. Reduce max_new_tokens or free VRAM.")
    except Exception as e:
        logger.error("Chat error: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws_chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            req = ChatRequest(**data)
            resp = await chat(req)
            await ws.send_json(resp.dict())
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await ws.close()
        except Exception:
            pass

@app.get("/")
async def root():
    return {
        "name": "Qwen2.5-Omni-7B API",
        "endpoints": {"health": "/healthz", "chat": "/chat", "ws": "/ws_chat", "frontend": "/frontend/"},
        "model": MODEL_ID,
        "device": DEVICE,
        "sr_hz": SAMPLE_RATE,
    }

if __name__ == "__main__":
    import uvicorn
    # Run on 8888 if you exposed that port in RunPod
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8888")), workers=1)
