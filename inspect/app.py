import gradio as gr
import librosa
import numpy as np

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = chroma.mean(axis=1)

    major_scores = [
        np.corrcoef(np.roll(mean_chroma, -i), MAJOR_PROFILE)[0, 1]
        for i in range(12)
    ]
    minor_scores = [
        np.corrcoef(np.roll(mean_chroma, -i), MINOR_PROFILE)[0, 1]
        for i in range(12)
    ]

    best_major = np.argmax(major_scores)
    best_minor = np.argmax(minor_scores)

    if major_scores[best_major] >= minor_scores[best_minor]:
        return f"{NOTE_NAMES[best_major]} major"
    else:
        return f"{NOTE_NAMES[best_minor]} minor"


def analyze(filepath):
    if filepath is None:
        return None

    y, sr = librosa.load(filepath, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    key = detect_key(y, sr)
    return [
        ["Tempo", f"{float(np.squeeze(tempo)):.1f} BPM"],
        ["Key", key],
    ]


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Audio(type="filepath", label="Upload audio file"),
    outputs=gr.Dataframe(headers=["Feature", "Value"], column_count=2, label="Analysis"),
    title="AudioFeatures",
    description="Upload an audio file to detect musical features via [librosa](https://librosa.org/).",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
