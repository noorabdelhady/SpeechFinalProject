"""
app.py
------
Gradio web interface for the Speech-to-Speech Translation System.
Case Study: Tourist Assistance System (English ↔ Arabic)

Run with:
    python app.py
"""

import gradio as gr
from translation_pipeline import speech_to_speech

# ---------------------------------------------------------------------------
# UI Theme & Custom CSS
# ---------------------------------------------------------------------------

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Tajawal:wght@300;400;500;700&display=swap');

:root {
    --primary:    #6C63FF;
    --secondary:  #48D0C5;
    --accent:     #FF6584;
    --bg-dark:    #0F0F1A;
    --bg-card:    #1A1A2E;
    --bg-card2:   #16213E;
    --text-main:  #E8E8F0;
    --text-muted: #8888AA;
    --border:     rgba(108,99,255,0.25);
    --glow:       rgba(108,99,255,0.15);
}

body, .gradio-container {
    background: var(--bg-dark) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main) !important;
}

/* Header */
#app-header {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
    border-bottom: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px var(--glow);
}

#app-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}

#app-header p {
    color: var(--text-muted);
    font-size: 1rem;
    margin: 0;
}

.emoji-flag { font-size: 1.6rem; margin: 0 0.3rem; }

/* Cards */
.gr-block, .gr-box, .gr-form,
.gradio-container .prose {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

/* Buttons */
#translate-btn {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    color: #fff !important;
    padding: 0.75rem 2rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 18px var(--glow) !important;
}

#translate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(108,99,255,0.4) !important;
}

#clear-btn {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-muted) !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
}

#clear-btn:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Labels */
label span {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}

/* Textboxes */
textarea, input[type="text"] {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
    font-family: 'Inter', 'Tajawal', sans-serif !important;
    font-size: 1rem !important;
}

/* Dropdowns */
.gr-dropdown select, select {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 10px !important;
}

/* Status / Info box */
#status-box {
    border-left: 4px solid var(--secondary) !important;
    background: rgba(72,208,197,0.07) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.9rem !important;
    color: var(--text-muted) !important;
}

/* Section headings */
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--primary);
    margin-bottom: 0.5rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.2rem 0;
}

/* Pipeline step pills */
#step-pills {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1rem 0;
}

.pill {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    color: var(--text-muted);
}

.pill-active {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: #fff;
    border-color: transparent;
}

/* Footer */
#app-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8rem;
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid var(--border);
}
"""

# ---------------------------------------------------------------------------
# Pipeline handler
# ---------------------------------------------------------------------------

def run_pipeline(audio_input, source_lang, target_lang):
    import os
    import shutil
    import time

    print("DEBUG original audio_input:", audio_input)

    if audio_input is None:
        return "", "", None, "⚠️ No audio provided."

    if source_lang == target_lang:
        return "", "", None, "⚠️ Languages must differ."

    try:
        # Ensure destination folder exists
        save_dir = "data/audio"
        os.makedirs(save_dir, exist_ok=True)

        # Wait briefly to ensure file is fully written
        time.sleep(1)

        # Create a stable filename
        filename = f"input_{int(time.time())}.wav"
        stable_path = os.path.join(save_dir, filename)

        # Copy file to stable location
        shutil.copy(audio_input, stable_path)

        print("DEBUG stable_path:", stable_path)
        print("FILE EXISTS:", os.path.exists(stable_path))

        # Run pipeline with stable file
        transcribed, translated, output_audio = speech_to_speech(
            stable_path, source_lang, target_lang
        )

        return transcribed, translated, output_audio, "✅ Done!"

    except Exception as e:
        return "", "", None, f"❌ Error: {str(e)}"


def clear_all():
    return None, "English", "Arabic", "", "", None, "Ready — record or upload audio to begin."


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

HEADER_HTML = """
<div id="app-header">
  <div style="font-size:2rem; margin-bottom:0.4rem;">
    <span class="emoji-flag">🇬🇧</span>
    <span style="color:#6C63FF; font-size:1.4rem; vertical-align:middle;">⇄</span>
    <span class="emoji-flag">🇪🇬</span>
  </div>
  <h1>Speech Translation System</h1>
  <p>Tourist Assistance · Lecture Translation · Bilingual Conversation</p>
  <div id="step-pills">
    <span class="pill pill-active">🎙️ ASR (Whisper)</span>
    <span style="color:#48D0C5; font-size:1rem; line-height:2;">→</span>
    <span class="pill pill-active">🌐 Machine Translation</span>
    <span style="color:#48D0C5; font-size:1rem; line-height:2;">→</span>
    <span class="pill pill-active">🔊 Neural TTS (Edge-TTS)</span>
  </div>
</div>
"""

FOOTER_HTML = """
<div id="app-footer">
  Built with OpenAI Whisper · Google Translate · Microsoft Edge Neural TTS · Gradio<br>
  ESLSCA University Egypt — Speech Processing Final Project
</div>
"""

EXAMPLES_EN = [
    "Where is the nearest hospital?",
    "Can you recommend a good restaurant nearby?",
    "How much does this cost?",
    "I need help, please.",
    "What time does the museum open?",
]

EXAMPLES_AR = [
    "أين أقرب مستشفى؟",
    "كم يكلف هذا؟",
    "أحتاج إلى مساعدة من فضلك.",
    "ما هو وقت افتتاح المتحف؟",
]


with gr.Blocks(title="Speech-to-Speech Translation | EN ↔ AR") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML(HEADER_HTML)

    # ── Main layout ─────────────────────────────────────────────────────────
    with gr.Row():

        # Left column — input
        with gr.Column(scale=1):
            gr.HTML('<p class="section-title">🎤 Input</p>')

            source_lang = gr.Dropdown(
                choices=["English", "Arabic"],
                value="English",
                label="Source Language",
                interactive=True,
            )

            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or Upload Audio",
                format="wav"
            )

            target_lang = gr.Dropdown(
                choices=["English", "Arabic"],
                value="Arabic",
                label="Target Language",
                interactive=True,
            )

            with gr.Row():
                translate_btn = gr.Button(
                    "🚀 Translate", variant="primary", elem_id="translate-btn"
                )
                clear_btn = gr.Button("🗑️ Clear", elem_id="clear-btn")

        # Right column — output
        with gr.Column(scale=1):
            gr.HTML('<p class="section-title">📄 Results</p>')

            transcription_box = gr.Textbox(
                label="📝 Transcription (ASR Output)",
                placeholder="Transcribed text will appear here …",
                lines=3,
                interactive=False,
            )

            translation_box = gr.Textbox(
                label="🌐 Translation (MT Output)",
                placeholder="Translated text will appear here …",
                lines=3,
                interactive=False,
            )

            audio_output = gr.Audio(
                label="🔊 Translated Speech (TTS Output)",
                type="filepath",
                interactive=False,
            )

    # ── Status bar ──────────────────────────────────────────────────────────
    status_box = gr.Textbox(
        value="Ready — record or upload audio to begin.",
        label="",
        interactive=False,
        elem_id="status-box",
        lines=1,
        max_lines=1,
    )

    # ── Examples ────────────────────────────────────────────────────────────
    gr.HTML('<hr class="divider"><p class="section-title">💡 Quick Example Phrases (Tourist Assistance)</p>')

    with gr.Row():
        with gr.Column():
            gr.HTML("<small style='color:#8888AA'>English phrases to try:</small>")
            for phrase in EXAMPLES_EN:
                gr.HTML(
                    f"<div style='background:#16213E;border:1px solid rgba(108,99,255,0.2);"
                    f"border-radius:8px;padding:0.4rem 0.8rem;margin:0.25rem 0;"
                    f"color:#E8E8F0;font-size:0.9rem;'>{phrase}</div>"
                )

        with gr.Column():
            gr.HTML("<small style='color:#8888AA'>Arabic phrases to try:</small>")
            for phrase in EXAMPLES_AR:
                gr.HTML(
                    f"<div style='background:#16213E;border:1px solid rgba(108,99,255,0.2);"
                    f"border-radius:8px;padding:0.4rem 0.8rem;margin:0.25rem 0;"
                    f"color:#E8E8F0;font-size:0.9rem;direction:rtl;'>{phrase}</div>"
                )

    # ── Footer ──────────────────────────────────────────────────────────────
    gr.HTML(FOOTER_HTML)

    # ── Event wiring ────────────────────────────────────────────────────────
    translate_btn.click(
        fn=run_pipeline,
        inputs=[audio_input, source_lang, target_lang],
        outputs=[transcription_box, translation_box, audio_output, status_box],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            audio_input, source_lang, target_lang,
            transcription_box, translation_box,
            audio_output, status_box,
        ],
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        inbrowser=True,
        css=custom_css,
        theme=gr.themes.Base(
            primary_hue="violet",
            neutral_hue="slate",
        ),
    )
