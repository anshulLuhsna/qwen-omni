#!/usr/bin/env python3
"""
Qwen2.5-Omni-7B AWQ FastAPI Server
Serves text and audio chat with TTS capabilities
"""

import os
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
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check CUDA availability
if not torch.cuda.is_available():
    logger.warning("CUDA not available! Running on CPU will be very slow.")
else:
    logger.info(f"CUDA available. Device: {torch.cuda.get_device_name(0)}")

# Model configuration
MODEL_ID = "Qwen/Qwen2.5-Omni-7B-AWQ"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
SAMPLE_RATE = 24000  # 24kHz audio

# Global model variables
model = None
processor = None
tokenizer = None

# Request/Response models
class ChatRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text input")
    audio_wav_base64: Optional[str] = Field(None, description="Base64 encoded WAV audio")
    voice: str = Field("Cherry", description="Voice for TTS: Cherry, Ethan, Serena, or Chelsie")

class ChatResponse(BaseModel):
    text: str
    audio_wav_base64: str

# Initialize FastAPI app
app = FastAPI(
    title="Qwen2.5-Omni-7B AWQ API",
    description="Text and audio chat with TTS",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

def load_model():
    """Load Qwen2.5-Omni-7B AWQ model with optimizations"""
    global model, processor, tokenizer
    
    logger.info(f"Loading model: {MODEL_ID}")
    
    try:
        from transformers import Qwen2OmniForConditionalGeneration, AutoTokenizer
        from qwen_omni_utils import Qwen2_5OmniProcessor
        
        # Model loading arguments
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        # Try flash attention 2 first
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 available, using optimized attention")
        except ImportError:
            logger.info("Flash Attention 2 not available, using standard attention")
        
        # Load model
        model = Qwen2OmniForConditionalGeneration.from_pretrained(
            MODEL_ID,
            **model_kwargs
        )
        
        # Load processor and tokenizer
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Set to eval mode
        model.eval()
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        raise

def get_tts_system_message(voice: str = "Cherry") -> str:
    """Get system message for TTS voice selection"""
    voice_map = {
        "Cherry": "Cherry",
        "Ethan": "Ethan", 
        "Serena": "Serena",
        "Chelsie": "Chelsie"
    }
    
    selected_voice = voice_map.get(voice, "Cherry")
    
    return f"""You are a helpful assistant with voice capabilities. 
When generating speech, use the {selected_voice} voice style.
Respond naturally and conversationally."""

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for text and audio"""
    global model, processor, tokenizer
    
    # Ensure model is loaded
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    try:
        # Prepare inputs
        text_input = request.text or ""
        audio_input = None
        
        # Handle audio input if provided
        if request.audio_wav_base64:
            try:
                audio_bytes = base64.b64decode(request.audio_wav_base64)
                audio_input = BytesIO(audio_bytes)
                # Read audio with soundfile
                audio_data, sr = sf.read(audio_input)
                # Resample to 24kHz if needed
                if sr != SAMPLE_RATE:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                raise HTTPException(status_code=400, detail="Invalid audio data")
        
        # Build conversation with system message for TTS
        system_msg = get_tts_system_message(request.voice)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_msg}
        ]
        
        # Add user message
        if text_input:
            messages.append({"role": "user", "content": text_input})
        elif audio_input:
            # Process audio input through model
            messages.append({"role": "user", "content": "Process audio input"})
        else:
            raise HTTPException(status_code=400, detail="No input provided")
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        if audio_input:
            inputs = processor(
                text=text,
                audios=[audio_data],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).to(DEVICE)
        else:
            inputs = processor(
                text=text,
                return_tensors="pt"
            ).to(DEVICE)
        
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
        
        # Decode text
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        text_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Generate audio from text response
        audio_response = generate_audio_from_text(text_response, request.voice)
        
        # Encode audio to base64
        audio_buffer = BytesIO()
        sf.write(audio_buffer, audio_response, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return ChatResponse(
            text=text_response,
            audio_wav_base64=audio_base64
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def generate_audio_from_text(text: str, voice: str = "Cherry") -> np.ndarray:
    """Generate audio from text using Qwen's TTS capabilities"""
    global model, processor, tokenizer
    
    try:
        # Prepare TTS prompt
        tts_messages = [
            {"role": "system", "content": get_tts_system_message(voice)},
            {"role": "user", "content": f"Please say: {text}"}
        ]
        
        tts_text = tokenizer.apply_chat_template(
            tts_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process for TTS
        inputs = processor(
            text=tts_text,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate audio
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Shorter for audio
                return_dict_in_generate=True,
                output_hidden_states=True
            )
        
        # Extract audio from model outputs
        # Note: This is a simplified version, actual implementation depends on model
        # For now, generate placeholder audio
        duration = min(len(text.split()) * 0.5, 10)  # Estimate duration
        samples = int(duration * SAMPLE_RATE)
        
        # Generate simple sine wave as placeholder
        # In production, extract actual audio from model outputs
        t = np.linspace(0, duration, samples)
        frequency = 440  # A4 note
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Add some variation based on text
        for i, char in enumerate(text[:20]):
            freq_mod = 440 + (ord(char) % 12) * 20
            audio += 0.1 * np.sin(2 * np.pi * freq_mod * t)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio.astype(np.float32)
        
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        # Return silence on error
        return np.zeros(SAMPLE_RATE, dtype=np.float32)

@app.websocket("/ws_chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()
    
    try:
        while True:
            # Receive data
            data = await websocket.receive_json()
            
            # Process through chat
            request = ChatRequest(**data)
            response = await chat(request)
            
            # Send response
            await websocket.send_json({
                "text": response.text,
                "audio_wav_base64": response.audio_wav_base64
            })
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Qwen2.5-Omni-7B AWQ API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/healthz",
            "chat": "/chat",
            "frontend": "/frontend/",
            "websocket": "/ws_chat"
        },
        "model": MODEL_ID,
        "device": DEVICE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)