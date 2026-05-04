# 🎙️ Speech-to-Speech Translation System (English ↔ Arabic)

> **ESLSCA University Egypt — Speech Processing Final Project**

An end-to-end, real-time Speech-to-Speech translation system supporting bidirectional translation between **English and Arabic**. Designed for real-world use cases including **tourist assistance**, **lecture translation**, and **bilingual conversation**.

---

## 🌟 Demo

Run the app and open your browser at `http://localhost:7860`.

---

## 🏗️ System Architecture

```
🎙️ Input Audio
      │
      ▼
┌─────────────────────────────────┐
│  ASR — OpenAI Whisper (base)    │  Speech → Text
│  Supports: English & Arabic     │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  MT — Google Translate          │  Text → Text
│  (via deep-translator)          │
│  EN ↔ AR                        │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  TTS — Microsoft Edge Neural    │  Text → Speech
│  EN: en-US-JennyNeural          │
│  AR: ar-EG-SalmaNeural          │
└─────────────────────────────────┘
      │
      ▼
🔊 Translated Audio Output
```

---

## 🧩 Components

| Component | Technology | Notes |
|-----------|-----------|-------|
| **ASR** | [OpenAI Whisper](https://github.com/openai/whisper) (`base` model) | Supports 99+ languages including Arabic |
| **Machine Translation** | [deep-translator](https://github.com/nidhaloff/deep-translator) (Google Translate) | Fast, accurate EN ↔ AR |
| **TTS** | [Microsoft Edge Neural TTS](https://github.com/rany2/edge-tts) | High-quality neural voices for both EN and AR |
| **UI** | [Gradio](https://gradio.app/) | Dark-themed, modern web interface |

> **Note on VALL-E:** Microsoft's VALL-E has not been publicly released, and open-source clones do not support Arabic. We use Microsoft's **Edge Neural TTS** which is based on the same generation of neural codec language models, providing VALL-E-quality speech synthesis for Arabic and English.

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10 or higher
- `pip`
- An active internet connection (for Machine Translation and Edge-TTS)
- A microphone (optional, for real-time recording)
- **FFmpeg** (required by Whisper)

#### Install FFmpeg

- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or use:
  ```bash
  winget install ffmpeg
  ```
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/noorabdelhady/SpeechFinalProject.git
cd SpeechFinalProject

# 2. (Recommended) Create a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python app.py
```

The app will open automatically in your browser at **http://localhost:7860**.

---

## 🖥️ How to Use

1. **Select Source Language** — choose **English** or **Arabic**.
2. **Record or Upload Audio** — use your microphone or upload an `.mp3`/`.wav` file.
3. **Select Target Language** — the language you want to translate *to*.
4. **Click "🚀 Translate"** — the pipeline will run in ~5–15 seconds depending on audio length.
5. **View Results**:
   - 📝 Transcription (what Whisper heard)
   - 🌐 Translation (the translated text)
   - 🔊 Play the synthesised audio in the target language.

---

## 🗺️ Case Studies Supported

| Use Case | Description |
|----------|-------------|
| 🏛️ **Tourist Assistance** | Ask questions like "Where is the nearest hospital?" in English and hear the Arabic answer |
| 🎓 **Lecture Translation** | Translate English lecture snippets into Arabic audio |
| 💬 **Bilingual Conversation** | Use both directions (EN→AR, AR→EN) alternately |
| 🎧 **Customer Service** | Translate customer queries across languages |
| 🚨 **Emergency Communication** | Quickly communicate critical phrases across the language barrier |

---

## 📁 Project Structure

```
SpeechFinalProject/
├── app.py                  # Gradio web application (main entry point)
├── translation_pipeline.py # Core ASR → MT → TTS pipeline
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔧 Configuration

You can adjust the following in `translation_pipeline.py`:

| Setting | Variable | Default |
|---------|----------|---------|
| Whisper model size | `_get_whisper_model(model_size=...)` | `"base"` (`"small"` for better accuracy) |
| English TTS voice | `TTS_VOICE_MAP["English"]` | `"en-US-JennyNeural"` |
| Arabic TTS voice | `TTS_VOICE_MAP["Arabic"]` | `"ar-EG-SalmaNeural"` |

### Available Arabic Voices (Edge-TTS)
- `ar-EG-SalmaNeural` — Egyptian Arabic, Female ⭐ (recommended)
- `ar-EG-ShakirNeural` — Egyptian Arabic, Male
- `ar-SA-ZariyahNeural` — Saudi Arabic, Female
- `ar-SA-HamedNeural` — Saudi Arabic, Male

---

## 📚 References & Acknowledgements

- **OpenAI Whisper** — Radford et al., 2022. *Robust Speech Recognition via Large-Scale Weak Supervision.* [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)
- **VALL-E** — Wang et al., 2023. *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers.* [arxiv.org/abs/2301.02111](https://arxiv.org/abs/2301.02111) *(Architecture inspiration)*
- **Microsoft Edge TTS** — [github.com/rany2/edge-tts](https://github.com/rany2/edge-tts)
- **deep-translator** — [github.com/nidhaloff/deep-translator](https://github.com/nidhaloff/deep-translator)
- **Gradio** — [gradio.app](https://gradio.app/)

---

## 👩‍💻 Authors

**Noor Abdelhady** — ESLSCA University Egypt, Speech Processing Course
