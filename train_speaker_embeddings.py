#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
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


class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"""

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Feature extraction and normalization
        if (
            hasattr(self.hparams, "use_tacotron2_mel_spec")
            and self.hparams.use_tacotron2_mel_spec
        ):
            feats = self.hparams.compute_features(audio=wavs)
            feats = torch.transpose(feats, 1, 2)
        else:
            feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            spkid = self.hparams.wav_augment.replicate_labels(spkid)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

def combined_dataio_prep(hparams):
    # Prepare VoxCeleb dataset
    vox_train = DynamicItemDataset.from_csv(
        csv_path=hparams["vox_train_annotation"],
        replacements={"data_root": hparams["vox_data_folder"]},
    )
    vox_valid = DynamicItemDataset.from_csv(
        csv_path=hparams["vox_valid_annotation"],
        replacements={"data_root": hparams["vox_data_folder"]},
    )

    # Prepare CommonVoice dataset
    cv_train = DynamicItemDataset.from_csv(
        csv_path=hparams["common_voice_fr_train_csv"],
        replacements={"data_root": hparams["common_voice_fr_data_folder"]},
    )
    cv_valid = DynamicItemDataset.from_csv(
        csv_path=hparams["common_voice_fr_valid_csv"],
        replacements={"data_root": hparams["common_voice_fr_data_folder"]},
    )

    # Count before filtering
    num_train_before = len(cv_train)
    num_valid_before = len(cv_valid)
    
    cv_train = cv_train.filtered_sorted(key_test={"duration": lambda d: d >= hparams["sentence_len"]})
    cv_valid = cv_valid.filtered_sorted(key_test={"duration": lambda d: d >= hparams["sentence_len"]})

    # Count after filtering
    num_train_after = len(cv_train)
    num_valid_after = len(cv_valid)

    vox_datasets = [vox_train, vox_valid]
    cv_datasets = [cv_train, cv_valid]

    print(f"Common Voice train samples removed for being too short: {num_train_before - num_train_after}")
    print(f"Common Voice valid samples removed for being too short: {num_valid_before - num_valid_after}")

    # Example: add audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def cv_audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        if info.num_channels > 1:
            sig = torch.mean(sig, dim=1)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"]
        )(sig)
        return resampled
    
    # Only for Common Voice datasets
    @sb.utils.data_pipeline.takes("sig", "duration")
    @sb.utils.data_pipeline.provides("sig", "duration")
    def truncate_pipeline(sig, duration):
        target_len = int(hparams["sample_rate"] * hparams["sentence_len"])
        if sig.shape[0] > target_len:
            start = random.randint(0, sig.shape[0] - target_len)
            sig = sig[start : start + target_len]
            duration = hparams["sentence_len"]
        else:
            duration = sig.shape[0] / hparams["sample_rate"]
        return sig, duration

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def vox_audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig
    
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    label_encoder.expect_len(hparams['out_n_neurons'])

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = Path(hparams["save_folder"]) / "label_encoder.txt"
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[vox_train, cv_train],
        output_key="spk_id",
    )
    
    sb.dataio.dataset.add_dynamic_item([*vox_datasets], vox_audio_pipeline)
    sb.dataio.dataset.add_dynamic_item([*cv_datasets], cv_audio_pipeline)
    sb.dataio.dataset.add_dynamic_item([*cv_datasets], truncate_pipeline)

    # Set output keys as needed
    sb.dataio.dataset.add_dynamic_item([*vox_datasets, *cv_datasets], label_pipeline)
    sb.dataio.dataset.set_output_keys([*vox_datasets, *cv_datasets], ["id", "sig", "spk_id_encoded"])

    # Combine datasets
    train_data = vox_train + cv_train
    valid_data = vox_valid + cv_valid

    return train_data, valid_data, label_encoder

def vox_dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["vox_data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["vox_train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["vox_valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder

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

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa

    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["vox_data_folder"],
            "save_folder": Path(hparams["save_folder"]) / "voxceleb",
            "verification_pairs_file": veri_file_path,
            "splits": ["train", "dev"],
            "split_ratio": hparams["split_ratio"],
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset preparation (parsing CommonVoice)
    from common_voice_prepare import prepare_common_voice  # noqa

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_common_voice,
        kwargs={
            "data_folder": hparams["common_voice_fr_data_folder"],
            "save_folder": str(Path(hparams["save_folder"]) / "common_voice_fr"),
            "train_tsv_file": hparams["common_voice_fr_train_tsv_file"],
            "dev_tsv_file": hparams["common_voice_fr_dev_tsv_file"],
            "test_tsv_file": hparams["common_voice_fr_test_tsv_file"],
            "language": hparams["common_voice_fr_language"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    # train_data, valid_data, label_encoder = voxceleb_dataio_prep(hparams)
    # train_data, valid_data, test_data, train_bsampler, valid_bsampler = common_voice_dataio_prep(hparams)

    train_data, valid_data, label_encoder = combined_dataio_prep(hparams)

    # train_data, valid_data, label_encoder = vox_dataio_prep(hparams)

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
