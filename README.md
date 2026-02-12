# Gesture AI Interface

Prototype showing real-time hand gesture detection + local AI response + speech output.

## Features
- 7 hand gestures via MediaPipe (pre-trained)
- Live webcam feed
- Ollama local LLM for natural replies
- Offline TTS (pyttsx3)

## Run
```bash
pip install -r requirements.txt
ollama serve &
ollama pull phi3:mini
python main.py
