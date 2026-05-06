"""
translation_pipeline.py
-----------------------
Core pipeline for Speech-to-Speech Translation (English <-> Arabic).

Pipeline:
  1. ASR  : Audio  -> Text  (OpenAI Whisper)
  2. MT   : Text   -> Text  (deep-translator / Google Translate)
  3. TTS  : Text   -> Audio (Coqui XTTS v2 — voice cloning, or edge-tts fallback)

Voice Cloning:
  XTTS v2 uses the original speaker's audio as a reference to synthesize the
  translated speech in the same voice. Only ~3–6 seconds of clear speech is needed.
  If XTTS v2 is unavailable (e.g., model download fails), edge-tts is used instead.
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

# XTTS v2 language codes (BCP-47 style used internally by XTTS)
XTTS_LANG_MAP = {
    "English": "en",
    "Arabic":  "ar",
}

# Edge-TTS fallback voices (used only when XTTS v2 is unavailable)
TTS_VOICE_MAP = {
    "English": "en-US-JennyNeural",
    "Arabic":  "ar-EG-SalmaNeural",
}

# ---------------------------------------------------------------------------
# Lazy-load Whisper model (cached at module level)
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
# Lazy-load XTTS v2 model (cached at module level)
# ---------------------------------------------------------------------------

_xtts_model = None
_xtts_available = None   # None = unchecked, True = ok, False = failed


def _get_xtts_model():
    """
    Load and cache the Coqui XTTS v2 model.
    Returns None if the model is not available or fails to load.
    """
    global _xtts_model, _xtts_available
    
    # If already loaded, return it
    if _xtts_model is not None:
        return _xtts_model

    try:
        from TTS.api import TTS
        print("[TTS] Initializing XTTS v2 model...")
        print("[TTS] Note: First run will download ~1.8GB. This may take several minutes.")
        
        # Set environment variable to auto-accept terms
        os.environ["COQUI_TOS_AGREED"] = "1"
        
        import torch
        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        
        # Initialize model
        _xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
        _xtts_available = True
        print(f"[TTS] XTTS v2 loaded successfully on {device}.")
        return _xtts_model
    except Exception as e:
        print(f"[TTS ERROR] XTTS v2 initialization failed: {e}")
        print("[TTS] Falling back to Edge-TTS for this request...")
        return None


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
    import whisper.audio
    import imageio_ffmpeg

    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Force Whisper to use bundled FFmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    whisper.audio.FFMPEG_BINARY = ffmpeg_path

    print(f"[ASR] Using FFmpeg at: {ffmpeg_path}")

    model = _get_whisper_model()
    lang_code = WHISPER_LANG_MAP.get(source_language, "en")

    print(f"[ASR] Transcribing audio (language='{lang_code}') …")

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
        return text

    print(f"[MT] Translating '{src}' -> '{tgt}' …")
    translator = GoogleTranslator(source=src, target=tgt)
    translated = translator.translate(text)
    print(f"[MT] Translation: {translated}")
    return translated


# ---------------------------------------------------------------------------
# Step 3a – TTS with voice cloning (XTTS v2)
# ---------------------------------------------------------------------------

def _generate_audio_xtts(
    text: str,
    target_language: str,
    speaker_wav: str,
    output_path: str,
) -> bool:
    """
    Synthesize speech using XTTS v2 in the speaker's own voice.

    Args:
        text:            Translated text to synthesize.
        target_language: 'English' or 'Arabic'.
        speaker_wav:     Path to the original speaker's audio (reference clip).
        output_path:     Where to save the output WAV.

    Returns:
        True on success, False on failure.
    """
    tts_model = _get_xtts_model()
    if tts_model is None:
        return False

    lang = XTTS_LANG_MAP.get(target_language, "en")

    try:
        print(f"[TTS-XTTS] Synthesizing in speaker's voice (lang={lang}) …")
        tts_model.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=lang,
            file_path=output_path,
        )
        print(f"[TTS-XTTS] Audio saved: {output_path}")
        return True
    except Exception as e:
        print(f"[TTS-XTTS] Error during synthesis: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 3b – TTS fallback (edge-tts)
# ---------------------------------------------------------------------------

async def _synthesize_edge_async(text: str, voice: str, output_path: str) -> None:
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


def _generate_audio_edge(text: str, target_language: str, output_path: str) -> None:
    """Synthesize speech using edge-tts (fallback, generic neural voice)."""
    voice = TTS_VOICE_MAP.get(target_language, "en-US-JennyNeural")
    print(f"[TTS-EdgeTTS] Using fallback voice: {voice}")
    _run_async(_synthesize_edge_async(text, voice, output_path))


# ---------------------------------------------------------------------------
# Step 3 – TTS dispatcher: tries XTTS v2 first, falls back to edge-tts
# ---------------------------------------------------------------------------

def generate_audio(
    text: str,
    target_language: str,
    speaker_wav: str = None,
    output_path: str = None,
) -> str:
    """
    Generate speech from text, optionally cloning the speaker's voice.

    Args:
        text:            Translated text to synthesize.
        target_language: 'English' or 'Arabic'.
        speaker_wav:     Path to original speaker audio for voice cloning.
                         If None or cloning fails, falls back to edge-tts.
        output_path:     Optional explicit output path (ignored; stable name used).

    Returns:
        Path to the generated audio file.
    """
    import time

    save_dir = "data/output"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = int(time.time())

    # XTTS v2 outputs WAV; edge-tts outputs MP3
    try_xtts = (speaker_wav is not None and os.path.exists(speaker_wav))

    if try_xtts:
        wav_path = os.path.join(save_dir, f"output_{timestamp}.wav")
        success = _generate_audio_xtts(text, target_language, speaker_wav, wav_path)
        if success and os.path.exists(wav_path):
            print(f"[TTS] Voice-cloned audio ready: {wav_path}")
            return wav_path
        print("[TTS] XTTS v2 failed — falling back to edge-tts …")

    # Fallback: edge-tts generic neural voice
    mp3_path = os.path.join(save_dir, f"output_{timestamp}.mp3")
    print(f"[TTS] Saving edge-tts audio to: {mp3_path}")
    _generate_audio_edge(text, target_language, mp3_path)

    if not os.path.exists(mp3_path):
        raise FileNotFoundError("TTS output file was not created.")

    print(f"[TTS] Edge-TTS audio saved: {mp3_path}")
    return mp3_path


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
        audio_path:       Path to input audio file (also used as voice reference).
        source_language:  'English' or 'Arabic'.
        target_language:  'English' or 'Arabic'.

    Returns:
        Tuple of (transcribed_text, translated_text, output_audio_path).
    """
    # Step 1: ASR
    transcribed = transcribe_audio(audio_path, source_language)

    # Step 2: Machine Translation
    translated = translate_text(transcribed, source_language, target_language)

    # Step 3: TTS — use speaker's own voice via XTTS v2, fall back to edge-tts
    output_audio = generate_audio(
        text=translated,
        target_language=target_language,
        speaker_wav=audio_path,   # <-- the recorded voice is the reference!
    )

    return transcribed, translated, output_audio
