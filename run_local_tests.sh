#!/bin/bash

# Test script for Qwen2.5-Omni-7B AWQ API
# Usage: ./run_local_tests.sh [SERVER_URL]

set -e

# Server URL (default to localhost, can override with RunPod proxy URL)
SERVER_URL="${1:-http://localhost:8000}"

echo "Testing Qwen2.5-Omni-7B AWQ API at: $SERVER_URL"
echo "==========================================="

# Test 1: Health check
echo -e "\n[1/3] Testing health endpoint..."
curl -s "$SERVER_URL/healthz" | python -m json.tool
echo "✓ Health check passed"

# Test 2: Text chat
echo -e "\n[2/3] Testing chat endpoint with text..."
RESPONSE=$(curl -s -X POST "$SERVER_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"text": "Say hello and introduce yourself briefly", "voice": "Cherry"}')

# Save response
echo "$RESPONSE" > test_response.json

# Extract text
echo "Response text:"
echo "$RESPONSE" | python -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('text', 'No text in response'))
"

# Extract and save audio
echo -e "\n[3/3] Extracting audio..."
python -c "
import json, base64

with open('test_response.json', 'r') as f:
    data = json.load(f)
    
if 'audio_wav_base64' in data:
    audio_bytes = base64.b64decode(data['audio_wav_base64'])
    with open('qwen_out.wav', 'wb') as out:
        out.write(audio_bytes)
    print('✓ Audio saved to: qwen_out.wav')
    print(f'  Audio size: {len(audio_bytes)} bytes')
else:
    print('✗ No audio in response')
"

# Test 3: Different voice
echo -e "\n[Bonus] Testing with different voice (Ethan)..."
curl -s -X POST "$SERVER_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"text": "Count from one to five", "voice": "Ethan"}' \
  -o test_ethan.json

python -c "
import json, base64

with open('test_ethan.json', 'r') as f:
    data = json.load(f)
    
if 'audio_wav_base64' in data:
    audio_bytes = base64.b64decode(data['audio_wav_base64'])
    with open('qwen_ethan.wav', 'wb') as out:
        out.write(audio_bytes)
    print('✓ Ethan voice audio saved to: qwen_ethan.wav')
else:
    print('✗ No audio in Ethan response')
"

echo -e "\n==========================================="
echo "All tests completed successfully!"
echo "Audio files saved:"
echo "  - qwen_out.wav (Cherry voice)"
echo "  - qwen_ethan.wav (Ethan voice)"
echo ""
echo "To play audio (if ffplay is available):"
echo "  ffplay qwen_out.wav"
echo "  ffplay qwen_ethan.wav"