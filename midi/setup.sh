#!/bin/bash
# Install basic-pitch without tensorflow (we use the ONNX backend via onnxruntime)
pip install basic-pitch --no-deps
pip install numpy scipy librosa resampy soundfile mir_eval pretty_midi
