#!/usr/bin/env python3
"""Compute oracle mel reconstruction loss (real-vs-real) to measure the
irreducible floor introduced by rand_slice_segments + dual STFT paths.

Path A: pre-computed linear spec  -> spec_to_mel_torch -> slice by ids_slice
Path B: raw waveform GT           -> slice by ids_slice -> mel_spectrogram_torch

oracle_loss = L1(path_A, path_B) * c_mel

If oracle_loss << val_loss_mel: generator is still the bottleneck.
If oracle_loss ~= val_loss_mel:  slicing/STFT artifacts dominate the floor.

Usage:
    python -m piper_train.compute_oracle_loss \
        --dataset-dir /home/dev/data_preprocessed \
        --num-batches 50
"""
import argparse
import pathlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .vits.dataset import PiperDataset, UtteranceCollate
from .vits.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .vits.commons import rand_slice_segments, slice_segments

# Must match training config
FILTER_LENGTH = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
MEL_CHANNELS = 80
SAMPLE_RATE = 22050
MEL_FMIN = 0.0
MEL_FMAX = None
SEGMENT_SIZE = 8192
C_MEL = 45
SEED = 1234


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path,
                        default=pathlib.Path("/home/dev/data_preprocessed"))
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-val", type=int, default=100)
    parser.add_argument("--num-test", type=int, default=500)
    args = parser.parse_args()

    full = PiperDataset([args.dataset_dir / "dataset.jsonl"])
    n = len(full)
    train_n = n - args.num_val - args.num_test
    train_ds, _, _ = random_split(
        full, [train_n, args.num_test, args.num_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=UtteranceCollate(is_multispeaker=False, segment_size=SEGMENT_SIZE),
        num_workers=4,
        generator=torch.Generator().manual_seed(SEED),
    )

    seg_frames = SEGMENT_SIZE // HOP_LENGTH
    losses_raw, losses_weighted = [], []

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break

        y = batch.audios          # [B, 1, T_wav]
        y_lengths = batch.audio_lengths
        spec = batch.spectrograms  # [B, 513, T_frames]  (linear spec)

        with torch.no_grad():
            # Path A: pre-computed linear spec -> mel -> slice
            mel_full = spec_to_mel_torch(
                spec.float(), FILTER_LENGTH, MEL_CHANNELS,
                SAMPLE_RATE, MEL_FMIN, MEL_FMAX,
            )
            _, ids_slice = rand_slice_segments(
                mel_full, y_lengths // HOP_LENGTH, seg_frames
            )
            y_mel = slice_segments(mel_full, ids_slice, seg_frames)

            # Path B: GT waveform -> slice -> mel
            y_slice = slice_segments(
                y, ids_slice * HOP_LENGTH, SEGMENT_SIZE
            )
            y_oracle_mel = mel_spectrogram_torch(
                y_slice.float().squeeze(1),
                FILTER_LENGTH, MEL_CHANNELS, SAMPLE_RATE,
                HOP_LENGTH, WIN_LENGTH, MEL_FMIN, MEL_FMAX,
            )

        loss_raw = F.l1_loss(y_mel, y_oracle_mel).item()
        losses_raw.append(loss_raw)
        losses_weighted.append(loss_raw * C_MEL)

        if (i + 1) % 10 == 0:
            print(f"[{i+1:3d}/{args.num_batches}] "
                  f"oracle_raw={sum(losses_raw)/len(losses_raw):.4f}  "
                  f"oracle_weighted={sum(losses_weighted)/len(losses_weighted):.4f}")

    mean_raw = sum(losses_raw) / len(losses_raw)
    mean_weighted = sum(losses_weighted) / len(losses_weighted)

    print()
    print("=" * 50)
    print(f"Oracle loss (raw)      : {mean_raw:.4f}")
    print(f"Oracle loss (×{C_MEL})   : {mean_weighted:.4f}")
    print(f"val_loss_mel (model)   : ~20.2  (epoch 773 best)")
    print(f"Model overhead         : ~{20.2 - mean_weighted:.2f}  (= val - oracle)")
    print(f"Batches evaluated      : {len(losses_raw)}")


if __name__ == "__main__":
    main()
