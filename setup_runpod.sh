#!/bin/bash

# git clone https://github.com/drazicmartin/VoiceID.git /workspace/VoiceID

# Install the required packages
pip install speechbrain
pip install datasets
pip install tensorboard
pip install soundfile

apt update && apt install -y ffmpeg

huggingface-cli login --token $HUGGINGFACE_HUB_TOKEN


python train_cv.py hparams/train_ecapa_tdnn_mel_spec_cv.yaml --precision fp16 --grad_accumulation_factor 2 --debug