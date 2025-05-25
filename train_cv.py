import os
import random
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataset import DynamicItemDataset

from common_voice_utils import get_single_dataset
from train_speaker_embeddings import SpeakerBrain

if __name__ == "__main__":
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exclude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)

    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    # Load the dataset
    debug = run_opts["debug"]
    train_data, label_encoder = get_single_dataset(hparams, lang="fr", split="train", debug=debug)
    valid_data, _             = get_single_dataset(hparams, lang="fr", split="validation", debug=debug)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
