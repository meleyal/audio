import logging
import os
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

# Prevent OpenMP runtime conflicts between PyTorch and faiss on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gradio as gr

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline

prequisites_download_pipeline(pretraineds_hifigan=True, models=True, exe=True)

UPLOADS_DIR = os.path.join(now_dir, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


@lru_cache(maxsize=None)
def _voice_converter():
    from rvc.infer.infer import VoiceConverter

    return VoiceConverter()


def _get_saved(ext):
    for f in os.listdir(UPLOADS_DIR):
        if f.endswith(ext):
            path = os.path.join(UPLOADS_DIR, f)
            if os.path.isfile(path):
                return path
    return None


def _save_file(src, ext):
    for f in os.listdir(UPLOADS_DIR):
        if f.endswith(ext):
            os.remove(os.path.join(UPLOADS_DIR, f))
    if src is None:
        return None
    # Preserve original stem but force the correct extension (Gradio may strip it in temp dir)
    stem = os.path.splitext(os.path.basename(src))[0]
    dest = os.path.join(UPLOADS_DIR, stem + ext)
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.copy(src, dest)
    return dest


def convert(model_file, index_file, audio_file):
    if model_file is None:
        return None, "Please upload a model (.pth) file."
    if audio_file is None:
        return None, "Please upload an audio file."

    audio_stem = Path(audio_file).stem
    model_stem = Path(model_file).stem
    output_path = str(Path(tempfile.mkdtemp()) / f"{audio_stem}_{model_stem}.wav")
    index_path = index_file if index_file is not None else ""

    _voice_converter().convert_audio(
        audio_input_path=audio_file,
        audio_output_path=output_path,
        model_path=model_file,
        index_path=index_path,
        pitch=0,
        index_rate=0.75 if index_path else 0,
        volume_envelope=1,
        protect=0.5,
        f0_method="rmvpe",
        split_audio=False,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        proposed_pitch=False,
        proposed_pitch_threshold=155.0,
        clean_audio=False,
        clean_strength=0.7,
        export_format="WAV",
        embedder_model="contentvec",
        embedder_model_custom=None,
        formant_shifting=False,
        formant_qfrency=1.0,
        formant_timbre=1.0,
        post_process=False,
        reverb=False,
        pitch_shift=False,
        limiter=False,
        gain=False,
        distortion=False,
        chorus=False,
        bitcrush=False,
        clipping=False,
        compressor=False,
        delay=False,
        sid=0,
    )

    return output_path, "Done."


with gr.Blocks(title="voice") as demo:
    gr.Markdown(
        "# voice\n"
        "Upload a voice model and vocals → convert via [RVC/Applio](https://github.com/IAHispano/Applio)."
    )

    gr.Markdown("### Voice model")
    gr.Markdown("Search for [RVC voice models](https://www.google.com/search?q=rvc+models).")
    model_file = gr.File(
        label="Model file (.pth)",
        file_types=[".pth"],
        value=lambda: _get_saved(".pth"),
    )
    index_file = gr.File(
        label="Index file (.index)",
        file_types=[".index"],
        value=lambda: _get_saved(".index"),
    )

    gr.Markdown("### Input vocals")
    audio_input = gr.Audio(label="Vocals", type="filepath")

    convert_btn = gr.Button("Convert", variant="primary")
    status = gr.Textbox(label="Status", interactive=False, lines=1, max_lines=1)

    gr.Markdown("### Output vocals")
    audio_output = gr.Audio(label="Converted audio")

    convert_btn.click(
        convert,
        inputs=[model_file, index_file, audio_input],
        outputs=[audio_output, status],
    )

    model_file.change(lambda f: _save_file(f, ".pth"), inputs=model_file, outputs=[])
    index_file.change(lambda f: _save_file(f, ".index"), inputs=index_file, outputs=[])


if __name__ == "__main__":
    demo.launch(allowed_paths=[UPLOADS_DIR])

