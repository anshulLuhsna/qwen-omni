// Qwen2.5-Omni Chat Frontend
// Handles text and audio communication with the API

// Get API URL - update this with your RunPod proxy URL
const API_URL = window.location.origin.replace('/frontend', '');

// Audio recording setup
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Check browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showStatus('Warning: Audio recording not supported in this browser', 'error');
    }
    
    // Add enter key listener
    document.getElementById('textInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Test connection
    testConnection();
});

async function testConnection() {
    try {
        const response = await fetch(`${API_URL}/healthz`);
        const data = await response.json();
        if (data.ok) {
            showStatus('Connected to server', 'success');
        }
    } catch (error) {
        showStatus('Cannot connect to server. Check the API URL.', 'error');
        console.error('Connection error:', error);
    }
}

async function sendMessage() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    if (!text) {
        showStatus('Please enter a message', 'error');
        return;
    }
    
    // Disable input while processing
    setInputsEnabled(false);
    showStatus('Processing...');
    
    // Add user message to chat
    addMessage(text, 'user');
    
    // Clear input
    textInput.value = '';
    
    try {
        // Get selected voice
        const voice = document.getElementById('voiceSelect').value;
        
        // Send request
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                voice: voice
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Add AI response to chat
        addMessage(data.text, 'ai');
        
        // Play audio response
        if (data.audio_wav_base64) {
            playAudioBase64(data.audio_wav_base64);
        }
        
        showStatus('Ready');
        
    } catch (error) {
        console.error('Error:', error);
        showStatus(`Error: ${error.message}`, 'error');
        addMessage('Sorry, an error occurred. Please try again.', 'ai');
    } finally {
        setInputsEnabled(true);
    }
}

async function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        // Request microphone permission
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                sampleRate: 24000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
        
        // Create MediaRecorder
        const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg';
        mediaRecorder = new MediaRecorder(stream, { mimeType });
        
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            // Convert to WAV and send
            const audioBlob = new Blob(audioChunks, { type: mimeType });
            await processRecordedAudio(audioBlob);
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };
        
        // Start recording
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        document.getElementById('recordBtn').textContent = '⏹ Stop Recording';
        document.getElementById('recordBtn').style.background = '#ff4444';
        document.getElementById('recordingIndicator').classList.add('active');
        showStatus('Recording audio...');
        
    } catch (error) {
        console.error('Recording error:', error);
        showStatus('Failed to access microphone', 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Update UI
        document.getElementById('recordBtn').textContent = '🎤 Record Audio';
        document.getElementById('recordBtn').style.background = '';
        document.getElementById('recordingIndicator').classList.remove('active');
        showStatus('Processing audio...');
    }
}

async function processRecordedAudio(audioBlob) {
    try {
        // Convert to WAV using Web Audio API
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 24000
        });
        
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Convert to mono WAV
        const wavBuffer = audioBufferToWav(audioBuffer);
        const base64Audio = arrayBufferToBase64(wavBuffer);
        
        // Add user message indicator
        addMessage('[Audio Message]', 'user');
        
        // Send to API
        setInputsEnabled(false);
        showStatus('Sending audio...');
        
        const voice = document.getElementById('voiceSelect').value;
        
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                audio_wav_base64: base64Audio,
                voice: voice
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Add AI response
        addMessage(data.text, 'ai');
        
        // Play audio response
        if (data.audio_wav_base64) {
            playAudioBase64(data.audio_wav_base64);
        }
        
        showStatus('Ready');
        
    } catch (error) {
        console.error('Audio processing error:', error);
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        setInputsEnabled(true);
    }
}

function audioBufferToWav(buffer) {
    const length = buffer.length * buffer.numberOfChannels * 2 + 44;
    const arrayBuffer = new ArrayBuffer(length);
    const view = new DataView(arrayBuffer);
    const channels = [];
    let offset = 0;
    let pos = 0;
    
    // Write WAV header
    const setUint16 = (data) => {
        view.setUint16(pos, data, true);
        pos += 2;
    };
    
    const setUint32 = (data) => {
        view.setUint32(pos, data, true);
        pos += 4;
    };
    
    // RIFF identifier
    setUint32(0x46464952);
    // file length
    setUint32(length - 8);
    // WAVE identifier
    setUint32(0x45564157);
    // fmt chunk identifier
    setUint32(0x20746d66);
    // fmt chunk length
    setUint32(16);
    // sample format (PCM)
    setUint16(1);
    // channel count
    setUint16(1);  // Mono
    // sample rate
    setUint32(24000);
    // byte rate
    setUint32(24000 * 1 * 2);
    // block align
    setUint16(1 * 2);
    // bits per sample
    setUint16(16);
    // data chunk identifier
    setUint32(0x61746164);
    // data chunk length
    setUint32(length - pos - 4);
    
    // Convert to mono and write PCM data
    const channel = buffer.getChannelData(0);
    let sample;
    
    for (let i = 0; i < channel.length; i++) {
        sample = Math.max(-1, Math.min(1, channel[i]));
        sample = sample * 0x7FFF;
        view.setInt16(pos, sample, true);
        pos += 2;
    }
    
    return arrayBuffer;
}

function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}

function playAudioBase64(base64Audio) {
    try {
        const audioPlayer = document.getElementById('audioPlayer');
        const audioData = window.atob(base64Audio);
        const arrayBuffer = new ArrayBuffer(audioData.length);
        const view = new Uint8Array(arrayBuffer);
        
        for (let i = 0; i < audioData.length; i++) {
            view[i] = audioData.charCodeAt(i);
        }
        
        const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        
        audioPlayer.src = url;
        audioPlayer.style.display = 'block';
        audioPlayer.play().catch(e => {
            console.error('Audio playback error:', e);
            showStatus('Audio playback failed', 'error');
        });
        
    } catch (error) {
        console.error('Audio decode error:', error);
        showStatus('Failed to play audio', 'error');
    }
}

function addMessage(text, sender) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = text;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = 'status';
    
    if (type === 'error') {
        statusDiv.style.color = '#c00';
    } else if (type === 'success') {
        statusDiv.style.color = '#0a0';
    } else {
        statusDiv.style.color = '#666';
    }
}

function setInputsEnabled(enabled) {
    document.getElementById('textInput').disabled = !enabled;
    document.getElementById('sendBtn').disabled = !enabled;
    document.getElementById('recordBtn').disabled = !enabled;
    document.getElementById('voiceSelect').disabled = !enabled;
}