# tts_player.py
import pyttsx3
import threading

engine = pyttsx3.init()
engine.setProperty("rate", 150)  # adjust speaking speed

audio_lock = threading.Lock()

def announce(text):
    """
    Non-blocking audio announcement using threading.
    Ensures only one announcement plays at a time.
    """
    def speak():
        with audio_lock:
            engine.say(text)
            engine.runAndWait()

    threading.Thread(target=speak, daemon=True).start()
