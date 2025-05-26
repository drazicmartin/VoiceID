#!/bin/bash

# Install the required packages
pip install speechbrain
pip install datasets
pip install tensorboard

apt update && apt install -y ffmpeg

python train_cv.py hparams/train_ecapa_tdnn_mel_spec_cv.yaml --precision fp16 --grad_accumulation_factor 2 --debug