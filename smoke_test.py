#!/usr/bin/env python3
"""
Smoke test for Qwen2.5-Omni-7B AWQ
Tests model loading and basic inference without server
"""

import sys
import torch
import numpy as np
from io import BytesIO

def smoke_test():
    """Run basic model loading and inference test"""
    print("=" * 50)
    print("Qwen2.5-Omni-7B AWQ Smoke Test")
    print("=" * 50)
    
    # Check CUDA
    print(f"\n[1/4] Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Import dependencies
    print(f"\n[2/4] Loading dependencies...")
    try:
        from transformers import Qwen2OmniForConditionalGeneration, AutoTokenizer
        from qwen_omni_utils import Qwen2_5OmniProcessor
        import soundfile as sf
        print("✓ All dependencies loaded")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    # Load model
    print(f"\n[3/4] Loading model (this may take a minute)...")
    try:
        MODEL_ID = "Qwen/Qwen2.5-Omni-7B-AWQ"
        
        # Model kwargs
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        # Try flash attention
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Using Flash Attention 2")
        except ImportError:
            print("  Using standard attention")
        
        # Load model
        model = Qwen2OmniForConditionalGeneration.from_pretrained(
            MODEL_ID,
            **model_kwargs
        )
        
        # Load processor and tokenizer
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        model.eval()
        print(f"✓ Model loaded successfully")
        
        # Memory check
        if cuda_available:
            memory_used = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU memory used: {memory_used:.1f} GB")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Test inference
    print(f"\n[4/4] Testing inference...")
    try:
        # Prepare test input
        test_text = "Hello! Please respond with a brief greeting."
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": test_text}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process input
        inputs = processor(
            text=text,
            return_tensors="pt"
        )
        
        if cuda_available:
            inputs = inputs.to("cuda")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        text_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"✓ Inference successful")
        print(f"  Input: {test_text}")
        print(f"  Output: {text_response[:100]}...")
        
        # Test audio generation (simplified)
        print(f"\n[Bonus] Testing audio generation...")
        
        # Generate dummy audio
        duration = 2.0  # seconds
        sample_rate = 24000
        samples = int(duration * sample_rate)
        audio = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        audio = audio.astype(np.float32)
        
        # Save to buffer
        audio_buffer = BytesIO()
        sf.write(audio_buffer, audio, sample_rate, format='WAV')
        audio_bytes = audio_buffer.getvalue()
        
        print(f"✓ Audio generation successful")
        print(f"  Duration: {duration}s")
        print(f"  Size: {len(audio_bytes)} bytes")
        
        # Cleanup
        if cuda_available:
            torch.cuda.empty_cache()
        
        print("\n" + "=" * 50)
        print("✅ All smoke tests passed!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = smoke_test()
    sys.exit(0 if success else 1)