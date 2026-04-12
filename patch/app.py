"""
patch
Pipeline: upload stem → syntheon Vital patch
"""

import os
import shutil
import tempfile
import threading
from pathlib import Path

import gradio as gr

# ── Lock (serialises Syntheon calls since CWD is process-wide) ────────────────

_syntheon_lock = threading.Lock()


# ── Processing functions ──────────────────────────────────────────────────────

def sf_load(audio_path: str):
    """Load audio via soundfile → torch tensor, avoiding torchaudio backend issues."""
    import soundfile as sf
    import torch
    data, sr = sf.read(audio_path, always_2d=True)  # [samples, channels]
    wav = torch.from_numpy(data.T).float()           # [channels, samples]
    return wav, sr


def pad_or_trim_to_4s(audio_path: str, session_dir: str) -> str:
    """
    Syntheon's model expects exactly 4 seconds at 16kHz (64000 samples).
    Resample to 16kHz first, then pad/trim — this makes syntheon's internal
    librosa resample a no-op and avoids off-by-one rounding errors.
    """
    import torch
    import torchaudio
    import soundfile as sf

    TARGET_SR = 16_000
    TARGET_SAMPLES = 4 * TARGET_SR  # exactly 64000

    wav, sr = sf_load(audio_path)

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    if wav.shape[-1] < TARGET_SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, TARGET_SAMPLES - wav.shape[-1]))
    else:
        wav = wav[..., :TARGET_SAMPLES]

    out_path = str(Path(session_dir) / "stem_16k_4s.wav")
    sf.write(out_path, wav.numpy().T, TARGET_SR)
    return out_path


def generate_vital_patch(stem_path: str, session_dir: str) -> str:
    """
    Run Syntheon on stem_path, return path to written .vital file.

    Syntheon writes "vital_output.vital" into CWD, so we chdir into a temp
    subdir under session_dir. A lock serialises calls since CWD is process-wide.
    Syntheon's model also requires exactly 4 seconds of audio.
    """
    from syntheon import infer_params

    padded_path = pad_or_trim_to_4s(stem_path, session_dir)

    work_dir = Path(session_dir) / "vital_work"
    work_dir.mkdir(exist_ok=True)

    original_cwd = os.getcwd()
    with _syntheon_lock:
        try:
            os.chdir(str(work_dir))
            out_fname, _ = infer_params(padded_path, "vital", enable_eval=False)
            dest = Path(session_dir) / f"{Path(stem_path).stem}.vital"
            shutil.copy2(str(work_dir / out_fname), str(dest))
        finally:
            os.chdir(original_cwd)

    return str(dest)


# ── Gradio event handlers ─────────────────────────────────────────────────────

def on_process(audio_file, state):
    """Generate Vital patch from uploaded stem."""
    if audio_file is None:
        return state, None, "⚠ Upload a stem file first."

    old_dir = state.get("session_dir")
    if old_dir and Path(old_dir).exists():
        shutil.rmtree(old_dir, ignore_errors=True)

    session_dir = tempfile.mkdtemp(prefix="patch_")
    new_state = {"session_dir": session_dir}

    try:
        vital_path = generate_vital_patch(audio_file, session_dir)
    except Exception as e:
        return new_state, None, f"❌ Generation failed: {e}"

    return new_state, vital_path, "✓ Done! Vital patch generated."


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="patch") as demo:
    gr.Markdown(
        "# patch\n"
        "Upload a stem → generate a Vital synth patch via Syntheon."
    )

    state = gr.State({})
    status = gr.Textbox(interactive=False, show_label=False, placeholder="Status…")

    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Stem file", scale=4)
        proc_btn = gr.Button("Generate Vital patch →", variant="primary", scale=1)

    gr.Markdown("### Output")
    vital_file = gr.File(label="Vital patch (.vital)")

    proc_btn.click(
        fn=on_process,
        inputs=[audio_in, state],
        outputs=[state, vital_file, status],
    )


if __name__ == "__main__":
    demo.launch()
