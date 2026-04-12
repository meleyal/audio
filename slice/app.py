import importlib.util
import pathlib
import shutil

# allin1 uses an old natten API (split QK/AV) removed in natten 0.21.5.
# Patch dinat.py with a pure-PyTorch replacement before importing allin1.
_spec = importlib.util.find_spec("allin1")
if _spec:
    _dst = pathlib.Path(_spec.origin).parent / "models" / "dinat.py"
    _src = pathlib.Path(__file__).parent / "patches" / "dinat.py"
    if _src.exists():
        shutil.copy(_src, _dst)

# torchaudio 2.11 replaced its save API with torchcodec, which requires
# libnppicc (CUDA NPP) not present on HF Spaces. Patch the file on disk so
# demucs subprocesses (which spawn a fresh Python) also get the fix.
_ta_spec = importlib.util.find_spec("torchaudio")
if _ta_spec:
    _tc_path = pathlib.Path(_ta_spec.origin).parent / "_torchcodec.py"
    if _tc_path.exists():
        _tc_src = _tc_path.read_text()
        _tc_old = (
            "    except ImportError as e:\n"
            "        raise ImportError(\n"
            '            "TorchCodec is required for save_with_torchcodec. " "Please install torchcodec to use this function."\n'
            "        ) from e\n"
        )
        _tc_new = (
            "    except (ImportError, RuntimeError):\n"
            "        import soundfile as _sf_\n"
            "        _wav_ = src.numpy()\n"
            "        if channels_first and _wav_.ndim == 2:\n"
            "            _wav_ = _wav_.T\n"
            "        _sf_.write(str(uri), _wav_, samplerate=sample_rate)\n"
            "        return\n"
        )
        if _tc_old in _tc_src and _tc_new not in _tc_src:
            _tc_path.write_text(_tc_src.replace(_tc_old, _tc_new))

import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import allin1
import gradio as gr
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_state: dict = {}
_result_cache: dict = {}

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[mGKHFABCDJPSTL]|\x1b[()][AB012]')


def _clean_output(text: str) -> str:
    text = _ANSI_ESCAPE.sub('', text)
    # Split on newlines; within each line handle \r by keeping the last segment
    lines = []
    for line in text.split('\n'):
        parts = line.split('\r')
        last = parts[-1]
        if last.strip():
            lines.append(last)
    return '\n'.join(lines).strip()


class _FDCapture:
    """Capture all output at the OS fd level, including subprocesses."""

    def __init__(self):
        self._buf = []
        self._lock = threading.Lock()
        self._saved = {}
        self._write_fd = None
        self._reader = None

    def start(self):
        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except Exception:
                pass
        read_fd, self._write_fd = os.pipe()
        for fd in (1, 2):
            self._saved[fd] = os.dup(fd)
            os.dup2(self._write_fd, fd)
        self._reader = threading.Thread(target=self._read_loop, args=(read_fd,), daemon=True)
        self._reader.start()

    def _read_loop(self, read_fd):
        with os.fdopen(read_fd, 'rb', buffering=0) as f:
            while True:
                chunk = f.read(256)
                if not chunk:
                    break
                with self._lock:
                    self._buf.append(chunk.decode('utf-8', errors='replace'))

    def stop(self):
        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except Exception:
                pass
        for fd, saved in self._saved.items():
            os.dup2(saved, fd)
            os.close(saved)
        os.close(self._write_fd)
        self._reader.join()

    def getvalue(self) -> str:
        with self._lock:
            return ''.join(self._buf)


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _get_allin1_result(audio_path: str):
    mtime = os.path.getmtime(audio_path)
    cached = _result_cache.get(audio_path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        result = allin1.analyze(audio_path, keep_byproducts=False, device=DEVICE)
    except Exception:
        result = allin1.analyze(audio_path, keep_byproducts=False, device="cpu")
    _result_cache[audio_path] = (mtime, result)
    return result


def on_analyze(audio_path, slice_mode, bars_per_slice):
    yield gr.update(), "Analyzing…"

    result_box = [None]
    error_box = [None]
    capture = _FDCapture()

    def _run():
        capture.start()
        try:
            result_box[0] = _get_allin1_result(audio_path)
        except Exception as e:
            error_box[0] = e
        finally:
            capture.stop()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while thread.is_alive():
        time.sleep(0.5)
        captured = _clean_output(capture.getvalue())
        yield gr.update(), captured or "Analyzing…"

    thread.join()

    if error_box[0]:
        yield gr.update(), ""
        raise gr.Error(str(error_box[0])) from error_box[0]

    result = result_box[0]

    duration = result.segments[-1].end if result.segments else 0.0
    downbeats = list(result.downbeats or result.beats or [])

    if slice_mode == "Sections (structural)":
        timestamps = [
            (float(seg.start), float(seg.end))
            for seg in result.segments
            if seg.end - seg.start > 0.5
        ]
    else:
        if not downbeats:
            yield gr.update(), ""
            raise gr.Error("No downbeats detected.")
        timestamps = []
        for i in range(0, len(downbeats), bars_per_slice):
            start = float(downbeats[i])
            end = float(downbeats[i + bars_per_slice]) if i + bars_per_slice < len(downbeats) else duration
            if end - start > 0.1:
                timestamps.append((start, end))

    if not timestamps:
        yield gr.update(), ""
        raise gr.Error("No slices found.")

    _state["timestamps"] = timestamps
    _state["audio_path"] = audio_path

    bpm = f"{result.bpm:.0f}" if result.bpm is not None else "—"
    stats = (
        f"BPM: {bpm}  |  "
        f"Duration: {format_duration(duration)}  |  "
        f"Beats: {len(result.beats)}  |  "
        f"Downbeats: {len(result.downbeats or [])}  |  "
        f"Slices: {len(timestamps)}"
    )
    yield gr.update(interactive=True, variant="primary"), stats


def on_slice():
    timestamps = _state.get("timestamps")
    audio_path = _state.get("audio_path")

    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_dir = tempfile.mkdtemp()

    src_wav = os.path.join(out_dir, "_source.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-c:a", "pcm_s16le", src_wav],
        check=True, capture_output=True,
    )

    def _slice(args):
        i, start, end = args
        out_path = os.path.join(out_dir, f"{base}_slice_{i+1:02d}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-to", str(end),
             "-i", src_wav, "-c:a", "pcm_s16le", out_path],
            check=True, capture_output=True,
        )
        return i, out_path

    with ThreadPoolExecutor() as pool:
        results = list(pool.map(_slice, [(i, s, e) for i, (s, e) in enumerate(timestamps)]))
    paths = [p for _, p in sorted(results)]

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, os.path.basename(p))

    return gr.update(value=tmp.name, visible=True)


with gr.Blocks(title="slice") as demo:
    gr.Markdown(
        "# slice\n"
        "Upload any audio → slice into loop-ready WAV files via [all-in-one](https://github.com/mir-aidj/all-in-one)."
    )

    audio_input = gr.Audio(label="Audio file", type="filepath", sources=["upload"])

    with gr.Row():
        slice_mode = gr.Radio(
            ["Beat-aligned", "Sections (structural)"],
            value="Beat-aligned",
            label="Slice mode",
        )
        bars_per_slice = gr.Slider(
            minimum=1, maximum=32, step=1, value=4,
            label="Bars per slice",
        )

    analyze_btn = gr.Button("Analyze", variant="primary", interactive=False)
    stats_box = gr.Textbox(show_label=False, interactive=False, placeholder="")
    slice_btn = gr.Button("Slice", interactive=False, variant="secondary")
    zip_out = gr.File(label="Download", visible=False)

    audio_input.change(
        lambda f: gr.update(interactive=f is not None),
        inputs=[audio_input],
        outputs=[analyze_btn],
        show_progress="hidden",
    )
    slice_mode.change(
        lambda m: gr.update(visible=(m == "Beat-aligned")),
        inputs=[slice_mode],
        outputs=[bars_per_slice],
        show_progress="hidden",
    )
    analyze_btn.click(
        on_analyze,
        inputs=[audio_input, slice_mode, bars_per_slice],
        outputs=[slice_btn, stats_box],
    )
    slice_btn.click(
        on_slice,
        outputs=[zip_out],
    )


if __name__ == "__main__":
    demo.launch()
