from TTS.api import TTS

print("Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

print("Generating speech...")
tts.tts_to_file(
    text="مرحبا كيف حالك",
    speaker_wav="C:/Users/noorw/OneDrive - ESLSCA University Egypt/Desktop/SpeechFinalProject/data/audio/input_1778084369.wav",  # <-- your audio file here
    language="ar",
    file_path="output.wav"
)

print("Done! Check output.wav")