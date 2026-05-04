"""
translation_pipeline.py
-----------------------
Core pipeline for Speech-to-Speech Translation (English <-> Arabic).

Pipeline:
  1. ASR  : Audio  -> Text  (OpenAI Whisper)
  2. MT   : Text   -> Text  (deep-translator / Google Translate)
  3. TTS  : Text   -> Audio (Microsoft Edge Neural TTS via edge-tts)
"""

import asyncio
import os
import tempfile
import threading

# ---------------------------------------------------------------------------
# Inject bundled FFmpeg (from imageio-ffmpeg) into PATH so Whisper can find it
# without requiring a system-level FFmpeg installation.
# ---------------------------------------------------------------------------
try:
    import imageio_ffmpeg
    _ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"[FFmpeg] Using bundled FFmpeg at: {imageio_ffmpeg.get_ffmpeg_exe()}")
except Exception as e:
    print(f"[FFmpeg] Warning: Could not inject bundled FFmpeg: {e}")

import whisper
from deep_translator import GoogleTranslator
import edge_tts

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

WHISPER_LANG_MAP = {
    "English": "en",
    "Arabic": "ar",
}

TRANSLATOR_LANG_MAP = {
    "English": "en",
    "Arabic": "ar",
}

# Edge-TTS voices — pick a high-quality neural voice for each language
TTS_VOICE_MAP = {
    "English": "en-US-JennyNeural",   # Clear, natural female US-English voice
    "Arabic":  "ar-EG-SalmaNeural",   # Clear, natural female Egyptian-Arabic voice
}


# ---------------------------------------------------------------------------
# Load Whisper model once (cached at module level)
# ---------------------------------------------------------------------------

_whisper_model = None


def _get_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """Load and cache the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        print(f"[ASR] Loading Whisper '{model_size}' model …")
        _whisper_model = whisper.load_model(model_size)
        print("[ASR] Whisper model loaded.")
    return _whisper_model


# ---------------------------------------------------------------------------
# Step 1 – ASR: Audio → Text
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, source_language: str) -> str:
    """
    Transcribe an audio file to text using OpenAI Whisper.

    Args:
        audio_path:       Path to the input audio file.
        source_language:  'English' or 'Arabic'.

    Returns:
        Transcribed text string.
    """
    import os
    import whisper
    import whisper.audio
    import imageio_ffmpeg

    # Ensure file exists
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Force Whisper to use bundled FFmpeg (fixes WinError 2)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    whisper.audio.FFMPEG_BINARY = ffmpeg_path

    print(f"[ASR] Using FFmpeg at: {ffmpeg_path}")

    # Load model
    model = _get_whisper_model()

    # Language mapping
    lang_code = WHISPER_LANG_MAP.get(source_language, "en")

    print(f"[ASR] Transcribing audio (language='{lang_code}') …")

    # Transcribe
    result = model.transcribe(
        audio_path,
        language=lang_code,
        fp16=False  # safer for CPU
    )

    text = result["text"].strip()
    print(f"[ASR] Transcription: {text}")

    return text


# ---------------------------------------------------------------------------
# Step 2 – MT: Text → Text
# ---------------------------------------------------------------------------

def translate_text(text: str, source_language: str, target_language: str) -> str:
    """
    Translate text from source language to target language.

    Args:
        text:             The text to translate.
        source_language:  'English' or 'Arabic'.
        target_language:  'English' or 'Arabic'.

    Returns:
        Translated text string.
    """
    src = TRANSLATOR_LANG_MAP.get(source_language, "en")
    tgt = TRANSLATOR_LANG_MAP.get(target_language, "ar")

    if src == tgt:
        return text  # No translation needed

    print(f"[MT] Translating '{src}' -> '{tgt}' …")
    translator = GoogleTranslator(source=src, target=tgt)
    translated = translator.translate(text)
    print(f"[MT] Translation: {translated}")
    return translated


# ---------------------------------------------------------------------------
# Step 3 – TTS: Text → Audio
# ---------------------------------------------------------------------------

async def _synthesize_async(text: str, voice: str, output_path: str) -> None:
    """Async helper to call edge-tts and save the output audio."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def _run_async(coro):
    """Run an async coroutine safely regardless of whether an event loop is running."""
    result = [None]
    exception = [None]

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result[0] = loop.run_until_complete(coro)
        except Exception as e:
            exception[0] = e
        finally:
            loop.close()

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    if exception[0]:
        raise exception[0]
    return result[0]


def generate_audio(text: str, target_language: str, output_path: str = None) -> str:
    import os
    import time

    voice = TTS_VOICE_MAP.get(target_language, "en-US-JennyNeural")

    # Ensure stable output directory
    save_dir = "data/output"
    os.makedirs(save_dir, exist_ok=True)

    # Create stable file path
    filename = f"output_{int(time.time())}.mp3"
    output_path = os.path.join(save_dir, filename)

    print(f"[TTS] Saving audio to: {output_path}")
    print(f"[TTS] Using voice: {voice}")

    try:
        _run_async(_synthesize_async(text, voice, output_path))

        # Verify file exists
        if not os.path.exists(output_path):
            raise FileNotFoundError("TTS output file was not created.")

        print(f"[TTS] Audio saved successfully.")
        return output_path

    except Exception as e:
        print(f"[TTS ERROR]: {e}")
        raise


# ---------------------------------------------------------------------------
# Full end-to-end pipeline
# ---------------------------------------------------------------------------

def speech_to_speech(
    audio_path: str,
    source_language: str,
    target_language: str,
) -> tuple[str, str, str]:
    """
    Run the full Speech-to-Speech translation pipeline.

    Args:
        audio_path:       Path to input audio file.
        source_language:  'English' or 'Arabic'.
        target_language:  'English' or 'Arabic'.

    Returns:
        Tuple of (transcribed_text, translated_text, output_audio_path).
    """
    # Step 1: ASR
    transcribed = transcribe_audio(audio_path, source_language)

    # Step 2: Machine Translation
    translated = translate_text(transcribed, source_language, target_language)

    # Step 3: TTS
    output_audio = generate_audio(translated, target_language)

    return transcribed, translated, output_audio
