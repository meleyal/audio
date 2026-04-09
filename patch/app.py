"""
Audiopatch
Pipeline: upload audio → demucs stem separation → pick stem → basic-pitch MIDI + syntheon Vital patch
"""

import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# basic-pitch uses scipy.signal.gaussian which was moved in newer scipy
import scipy.signal

if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = scipy.signal.windows.gaussian

import gradio as gr

STEMS = ["drums", "bass", "other", "vocals"]

# ── Model caches (loaded once, reused across requests) ────────────────────────

_lock = threading.Lock()
_demucs_model = None
_bp_model_path = None
_syntheon_lock = threading.Lock()


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


def load_basic_pitch():
    """Return the ONNX model path for basic-pitch (avoids scipy.signal.gaussian issue in TFLite backend)."""
    global _bp_model_path
    if _bp_model_path is None:
        with _lock:
            if _bp_model_path is None:
                from pathlib import Path

                from basic_pitch import ICASSP_2022_MODEL_PATH
                _bp_model_path = Path(ICASSP_2022_MODEL_PATH).with_suffix(".onnx")
    return _bp_model_path


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
    # convert_audio handles mono→stereo and resampling
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


def extract_midi(stem_path: str, session_dir: str) -> str:
    """Run Basic Pitch on stem_path, return path to written .mid file."""
    from basic_pitch.inference import predict

    bp = load_basic_pitch()
    _, midi_data, _ = predict(stem_path, bp)

    out_path = str(Path(session_dir) / "output.mid")
    midi_data.write(out_path)
    return out_path


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

    # Ensure exactly 4 seconds for the model
    padded_path = pad_or_trim_to_4s(stem_path, session_dir)

    work_dir = Path(session_dir) / "vital_work"
    work_dir.mkdir(exist_ok=True)

    original_cwd = os.getcwd()
    with _syntheon_lock:
        try:
            os.chdir(str(work_dir))
            out_fname, _ = infer_params(padded_path, "vital", enable_eval=False)
            dest = Path(session_dir) / "output.vital"
            shutil.copy2(str(work_dir / out_fname), str(dest))
        finally:
            os.chdir(original_cwd)

    return str(dest)


# ── Gradio event handlers ─────────────────────────────────────────────────────

def on_separate(audio_file, state):
    """Step 1: separate stems from uploaded audio."""
    if audio_file is None:
        return state, None, None, None, None, "⚠ Upload an audio file first."

    # Clean up previous session
    old_dir = state.get("session_dir")
    if old_dir and Path(old_dir).exists():
        shutil.rmtree(old_dir, ignore_errors=True)

    session_dir = tempfile.mkdtemp(prefix="poc_")

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
        "✓ Stems ready — preview them, choose one, and click Process.",
    )


def on_process(stem_choice, state):
    """Step 2: extract MIDI and generate Vital patch from chosen stem, in parallel."""
    paths = state.get("stem_paths", {})
    stem_path = paths.get(stem_choice)
    session_dir = state.get("session_dir")

    if not stem_path or not session_dir:
        return None, None, "⚠ No stem selected or session expired — please re-upload."

    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(extract_midi, stem_path, session_dir): "midi",
            executor.submit(generate_vital_patch, stem_path, session_dir): "vital",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                errors[key] = str(e)

    if errors:
        msg = " | ".join(f"{k}: {v}" for k, v in errors.items())
        return None, None, f"❌ {msg}"

    return (
        results["midi"],
        results["vital"],
        f"✓ Done! MIDI + Vital patch generated from '{stem_choice}'.",
    )


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Audiopatch") as demo:
    gr.Markdown(
        "# Audiopatch\n"
        "Upload any audio → separate stems via Demucs → extract MIDI via Basic Pitch "
        "+ generate a Vital synth patch via Syntheon."
    )

    state = gr.State({})
    status = gr.Textbox(interactive=False, show_label=False, placeholder="Status…")

    # Step 1: upload + separate
    gr.Markdown("### Step 1 · Separate stems")
    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Audio file", scale=4)
        sep_btn  = gr.Button("Separate stems →", variant="primary", scale=1)

    # Step 2: stem preview + pick
    gr.Markdown("### Step 2 · Preview stems and pick one")
    with gr.Row():
        drums_out  = gr.Audio(label="Drums",  interactive=False)
        bass_out   = gr.Audio(label="Bass",   interactive=False)
        other_out  = gr.Audio(label="Other",  interactive=False)
        vocals_out = gr.Audio(label="Vocals", interactive=False)
    with gr.Row():
        stem_radio = gr.Radio(choices=STEMS, value="bass", label="Choose stem", scale=4)
        proc_btn   = gr.Button("Extract MIDI + Vital patch →", variant="primary", scale=1)

    # Step 3: outputs (always visible — empty until processing completes)
    gr.Markdown("### Step 3 · Download outputs")
    with gr.Row():
        midi_file  = gr.File(label="MIDI (.mid)")
        vital_file = gr.File(label="Vital patch (.vital)")

    # Wire events
    sep_btn.click(
        fn=on_separate,
        inputs=[audio_in, state],
        outputs=[state, drums_out, bass_out, other_out, vocals_out, status],
    )

    proc_btn.click(
        fn=on_process,
        inputs=[stem_radio, state],
        outputs=[midi_file, vital_file, status],
    )


if __name__ == "__main__":
    demo.launch()
