"""
midi
Pipeline: upload audio → Basic Pitch MIDI extraction
"""

import shutil
import tempfile
import threading
from pathlib import Path

# basic-pitch uses scipy.signal.gaussian which was moved in newer scipy
import scipy.signal

if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = scipy.signal.windows.gaussian

import gradio as gr

# ── Model cache (loaded once, reused across requests) ─────────────────────────

_lock = threading.Lock()
_bp_model_path = None


def load_basic_pitch():
    """Return the ONNX model path for basic-pitch (avoids scipy.signal.gaussian issue in TFLite backend)."""
    global _bp_model_path
    if _bp_model_path is None:
        with _lock:
            if _bp_model_path is None:
                from basic_pitch import ICASSP_2022_MODEL_PATH
                _bp_model_path = Path(ICASSP_2022_MODEL_PATH).with_suffix(".onnx")
    return _bp_model_path


# ── Processing functions ──────────────────────────────────────────────────────

def extract_midi(audio_path: str, session_dir: str) -> str:
    """Run Basic Pitch on audio_path, return path to written .mid file."""
    from basic_pitch.inference import predict

    bp = load_basic_pitch()
    _, midi_data, _ = predict(audio_path, bp)

    out_path = str(Path(session_dir) / f"{Path(audio_path).stem}.mid")
    midi_data.write(out_path)
    return out_path


# ── Gradio event handlers ─────────────────────────────────────────────────────

def on_process(audio_file, state):
    """Extract MIDI from uploaded audio."""
    if audio_file is None:
        return state, None, "⚠ Upload an audio file first."

    old_dir = state.get("session_dir")
    if old_dir and Path(old_dir).exists():
        shutil.rmtree(old_dir, ignore_errors=True)

    session_dir = tempfile.mkdtemp(prefix="midi_")
    new_state = {"session_dir": session_dir}

    try:
        midi_path = extract_midi(audio_file, session_dir)
    except Exception as e:
        return new_state, None, f"❌ Extraction failed: {e}"

    return new_state, midi_path, "✓ Done! MIDI extracted."


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="midi") as demo:
    gr.Markdown(
        "# midi\n"
        "Upload any audio → extract MIDI via Basic Pitch."
    )

    state = gr.State({})
    status = gr.Textbox(interactive=False, show_label=False, placeholder="Status…")

    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Audio file", scale=4)
        proc_btn = gr.Button("Extract MIDI →", variant="primary", scale=1)

    gr.Markdown("### Output")
    midi_file = gr.File(label="MIDI (.mid)")

    proc_btn.click(
        fn=on_process,
        inputs=[audio_in, state],
        outputs=[state, midi_file, status],
    )


if __name__ == "__main__":
    demo.launch()
