"""
Audio -> MIDI drum pattern extractor for isolated drum loops.

Pipeline:
  1. ADTOF-pytorch (Frame_RNN) transcribes the input audio to onset times + 5 drum classes.
  2. Onsets are written as a General MIDI drum file with pretty_midi.
"""

import tempfile

import gradio as gr
import librosa
import numpy as np
import pretty_midi
from adtof_pytorch import transcribe_to_midi

DRUM_NAMES = {35: "Kick", 36: "Kick", 38: "Snare", 42: "Hi-Hat", 47: "Tom", 49: "Cymbal"}


def transcribe(audio_path: str):
    if audio_path is None:
        return None, None, "Please upload a drum loop."

    out_path = tempfile.mktemp(suffix=".mid")
    transcribe_to_midi(audio_path, out_path)

    pm = pretty_midi.PrettyMIDI(out_path)
    all_notes = [n for inst in pm.instruments for n in inst.notes]
    if not all_notes:
        return None, None, "No drum hits detected."

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    if not (40 < tempo < 300):
        tempo = 120.0

    out_pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    drum_inst.notes = sorted(all_notes, key=lambda n: n.start)
    out_pm.instruments.append(drum_inst)
    out_pm.write(out_path)

    counts: dict[str, int] = {}
    rows = []
    for n in drum_inst.notes[:300]:
        name = DRUM_NAMES.get(n.pitch, str(n.pitch))
        counts[name] = counts.get(name, 0) + 1
        rows.append([f"{n.start:.3f}", name])

    summary = (
        f"**Tempo:** {tempo:.1f} BPM  \n"
        + "  ".join(f"**{k}:** {v}" for k, v in counts.items())
        + f"  \n**Total:** {sum(counts.values())} hits"
    )
    return out_path, rows, summary


with gr.Blocks(title="drums") as demo:
    gr.Markdown(
        "# drums\n"
        "Upload a drum loop → extract a MIDI drum pattern via ADTOF."
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(label="Drum loop", type="filepath", sources=["upload"])
            run = gr.Button("Extract MIDI", variant="primary")
        with gr.Column():
            midi_out = gr.File(label="MIDI file (.mid)")
            info = gr.Markdown()
            table = gr.Dataframe(
                headers=["Time (s)", "Instrument"],
                label="Detected hits (first 300)",
                interactive=False,
            )

    run.click(
        transcribe,
        inputs=[audio_in],
        outputs=[midi_out, table, info],
    )

if __name__ == "__main__":
    demo.launch()
