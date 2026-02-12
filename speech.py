import pyttsx3

def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 145)     # speaking speed
    engine.setProperty('volume', 0.92)  # 0.0–1.0
    # Try to use a nicer voice if available (Ubuntu)
    voices = engine.getProperty('voices')
    for voice in voices:
        if "english" in voice.name.lower() and "us" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    return engine

def speak(engine, text):
    engine.say(text)
    engine.runAndWait()