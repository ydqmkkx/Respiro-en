
# **Respiro-en**

Official PyTorch implementation of the paper: \
**Frame-Wise Breath Detection with Self-Training: An Exploration of Enhancing Breath Naturalness in Text-to-Speech**, ***INTERSPEECH 2024***.

<a href='https://arxiv.org/abs/2402.00288'><img src='https://img.shields.io/badge/arXiv-red'></a>
<!-- <a href='https://huggingface.co/DongYANG/Respiro-en'><img src='https://img.shields.io/badge/🤗-yellow'></a> -->
<a href='https://ydqmkkx.github.io/breath-detection/'><img src='https://img.shields.io/badge/Demo-blue'></a>


## Introduction

This model is developed for detecting the positions of breath sounds in speech utterances. \
It was trained using **[LibriTTS-R](https://arxiv.org/abs/2305.18802)** corpus.

<img src="model.png" width="50%"/>

## Get Started

```python
import torch
import librosa
import numpy as np
from modules import feature_extractor, DetectionNet

# model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectionNet().to(device)
checkpoint = torch.load("respiro-en.pt")
model.load_state_dict(checkpoint["model"])
model.eval()

# Example. This file is from LibriTTS-R corpus
wav_path = "26_495_000005_000000.wav"
wav, sr = librosa.load(wav_path, sr=16000)
feature, length = feature_extractor(wav, sr)
feature, length = feature.to(device), length.to(device)
output = model(feature, length)

wav_path = "train-clean-100_19_198_000010_000003.wav"
wav, sr = librosa.load(wav_path, sr=16000)
feature, length = feature_extractor(wav)
feature, length = feature.to(device), length.to(device)
output = model(feature, length)

# 0.064 is the threshold obtained from our validation set
# You can try more strict thresholds like 0.5 or 0.9
threshold = 0.064
prediction = (output[0] > 0.064).nonzero().squeeze().tolist()
if isinstance(prediction, list) and len(prediction)>1:
    diffs = np.diff(prediction)
    splits = np.where(diffs != 1)[0] + 1
    splits = np.split(prediction, splits)
    for split in splits:
        print(split)
# The segments of breath are printed
# 229 means 229 ms
```
