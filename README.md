# Gesture AI Interface

First public prototype showing **gesture-controlled AI interaction** – early step toward sovereign AI input for 4D holograms.

## What it does

- Detects 5 hand gestures in real-time using webcam (no training, fully local)
- Maps gestures to simple commands
- Speaks natural-language responses (offline TTS)
- Visual feedback on screen

## Tech stack (all free & local)

- Python 3.10+
- MediaPipe Hands (Google) – gesture detection
- OpenCV – webcam access & drawing
- pyttsx3 – offline text-to-speech

## How to run

```bash
# 1. Clone
git clone https://github.com/yourusername/gesture-ai-interface.git
cd gesture-ai-interface

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```
