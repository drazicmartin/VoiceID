#!/bin/bash

# git clone https://github.com/drazicmartin/VoiceID.git /workspace/VoiceID; cd /workspace/VoiceID; . setup_runpod.sh

# Install the required packages
pip install speechbrain
pip install datasets
pip install tensorboard
pip install soundfile
pip install librosa

apt update
apt install -y ffmpeg nano zip

huggingface-cli login --token $HUGGINGFACE_HUB_TOKEN

python train_cv.py hparams/train_ecapa_tdnn_mel_spec_cv.yaml