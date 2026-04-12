"""
stems
Pipeline: upload audio → demucs stem separation → preview + download stems
"""

import shutil
import tempfile
import threading
from pathlib import Path

import gradio as gr

STEMS = ["drums", "bass", "other", "vocals"]

# ── Model cache (loaded once, reused across requests) ─────────────────────────

_lock = threading.Lock()
_demucs_model = None


def load_demucs():
    global _demucs_model
    if _demucs_model is None:
        with _lock:
            if _demucs_model is None:
                from demucs.pretrained import get_model
                model = get_model("htdemucs")
                model.eval()
                _demucs_model = model
    return _demucs_model


# ── Processing functions ──────────────────────────────────────────────────────

def sf_load(audio_path: str):
    """Load audio via soundfile → torch tensor, avoiding torchaudio backend issues."""
    import soundfile as sf
    import torch
    data, sr = sf.read(audio_path, always_2d=True)  # [samples, channels]
    wav = torch.from_numpy(data.T).float()           # [channels, samples]
    return wav, sr


def separate_stems(audio_path: str, session_dir: str) -> dict[str, str]:
    """Run htdemucs on audio_path, write stems to session_dir, return {name: path}."""
    import torch
    import soundfile as sf
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model = load_demucs()

    wav, sr = sf_load(audio_path)
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)

    mix = wav.unsqueeze(0)  # [1, channels, samples]
    with torch.no_grad():
        sources = apply_model(model, mix, progress=False)  # [1, 4, channels, samples]

    paths = {}
    for i, name in enumerate(model.sources):  # ["drums", "bass", "other", "vocals"]
        path = str(Path(session_dir) / f"{name}.wav")
        sf.write(path, sources[0, i].numpy().T, model.samplerate)
        paths[name] = path

    return paths


# ── Gradio event handlers ─────────────────────────────────────────────────────

def on_separate(audio_file, state):
    """Separate stems from uploaded audio."""
    if audio_file is None:
        return state, None, None, None, None, "⚠ Upload an audio file first."

    # Clean up previous session
    old_dir = state.get("session_dir")
    if old_dir and Path(old_dir).exists():
        shutil.rmtree(old_dir, ignore_errors=True)

    session_dir = tempfile.mkdtemp(prefix="stems_")

    try:
        paths = separate_stems(audio_file, session_dir)
    except Exception as e:
        return state, None, None, None, None, f"❌ Separation failed: {e}"

    new_state = {"session_dir": session_dir, "stem_paths": paths}
    return (
        new_state,
        paths.get("drums"),
        paths.get("bass"),
        paths.get("other"),
        paths.get("vocals"),
        "✓ Stems ready.",
    )


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="stems") as demo:
    gr.Markdown(
        "# stems\n"
        "Upload any audio → separate into stems via Demucs (htdemucs)."
    )

    state = gr.State({})
    status = gr.Textbox(interactive=False, show_label=False, placeholder="Status…")

    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Audio file", scale=4)
        sep_btn  = gr.Button("Separate stems →", variant="primary", scale=1)

    gr.Markdown("### Stems")
    with gr.Row():
        drums_out  = gr.Audio(label="Drums",  interactive=False)
        bass_out   = gr.Audio(label="Bass",   interactive=False)
        other_out  = gr.Audio(label="Other",  interactive=False)
        vocals_out = gr.Audio(label="Vocals", interactive=False)

    sep_btn.click(
        fn=on_separate,
        inputs=[audio_in, state],
        outputs=[state, drums_out, bass_out, other_out, vocals_out, status],
    )


if __name__ == "__main__":
    demo.launch()
