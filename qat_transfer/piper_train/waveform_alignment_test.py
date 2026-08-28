#!/usr/bin/env python3
"""Measure waveform phase alignment between Generator output and GT slice.

Uses normalized cross-correlation to find the lag (in samples) between
y_hat and y_slice. Also measures mel L1 at lag=0 vs mel L1 after
time-shifting y_hat to best-align with GT.

If aligned_mel_loss << unaligned_mel_loss: phase shift is the main floor.
If aligned_mel_loss ~= unaligned_mel_loss: generator content error dominates.

Usage:
    python -m piper_train.waveform_alignment_test \
        --checkpoint /home/dev/01_Baseline_BigVgan_VITS2_FO/.../best.ckpt \
        --dataset-dir /home/dev/data_preprocessed \
        --num-batches 20
"""
import argparse
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

torch.serialization.add_safe_globals([pathlib.PosixPath])

from .vits.dataset import PiperDataset, UtteranceCollate
from .vits.lightning import VitsModel
from .vits.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .vits.commons import rand_slice_segments, slice_segments

SEED = 1234
HOP_LENGTH = 256
SEGMENT_SIZE = 8192  # 32 frames


def xcorr_lag(y_hat: np.ndarray, y_gt: np.ndarray) -> int:
    """Return sample lag that maximises normalised cross-correlation."""
    corr = np.correlate(y_hat, y_gt, mode="full")
    lag = int(corr.argmax()) - (len(y_gt) - 1)
    return lag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-dir", type=pathlib.Path,
                        default=pathlib.Path("/home/dev/data_preprocessed"))
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-val", type=int, default=100)
    parser.add_argument("--num-test", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading: {args.checkpoint}")
    model = VitsModel.load_from_checkpoint(
        str(args.checkpoint), dataset=None, strict=False, map_location="cpu"
    )
    model.eval()
    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()

    hp = model.hparams
    seg_frames = SEGMENT_SIZE // HOP_LENGTH

    full = PiperDataset([args.dataset_dir / "dataset.jsonl"])
    n = len(full)
    train_n = n - args.num_val - args.num_test
    _, _, val_ds = random_split(
        full, [train_n, args.num_test, args.num_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=UtteranceCollate(is_multispeaker=False, segment_size=SEGMENT_SIZE),
        num_workers=2,
    )

    lags, losses_unaligned, losses_aligned = [], [], []

    def to_mel(wav):
        return mel_spectrogram_torch(
            wav.float().squeeze(1),
            hp.filter_length, hp.mel_channels, hp.sample_rate,
            hp.hop_length, hp.win_length, hp.mel_fmin, hp.mel_fmax,
        )

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break

            spec = batch.spectrograms
            spec_lengths = batch.spectrogram_lengths
            y = batch.audios
            y_lengths = batch.audio_lengths

            # Posterior encoder (teacher-forced, uses GT spec)
            z, m_q, logs_q, y_mask = model.model_g.enc_q(spec, spec_lengths)

            # Consistent slice indices
            _, ids_slice = rand_slice_segments(
                z, y_lengths // HOP_LENGTH, seg_frames
            )
            z_slice = slice_segments(z, ids_slice, seg_frames)
            y_slice = slice_segments(y, ids_slice * HOP_LENGTH, SEGMENT_SIZE)

            # Generate
            y_hat = model.model_g.dec(z_slice)

            # GT mel (path A)
            mel_full = spec_to_mel_torch(
                spec.float(), hp.filter_length, hp.mel_channels,
                hp.sample_rate, hp.mel_fmin, hp.mel_fmax,
            )
            y_mel = slice_segments(mel_full, ids_slice, seg_frames)

            # Unaligned mel loss
            loss_u = F.l1_loss(y_mel, to_mel(y_hat)).item() * hp.c_mel
            losses_unaligned.append(loss_u)

            # Per-utterance cross-correlation
            batch_lags = []
            for b in range(y_hat.shape[0]):
                hat_np = y_hat[b, 0].numpy()
                gt_np  = y_slice[b, 0].numpy()
                lag = xcorr_lag(hat_np, gt_np)
                batch_lags.append(lag)

                # Shift y_hat by -lag and compute aligned mel loss
                if lag != 0:
                    if lag > 0:
                        hat_shifted = torch.cat([y_hat[b:b+1, :, lag:],
                                                 torch.zeros(1, 1, lag)], dim=2)
                    else:
                        hat_shifted = torch.cat([torch.zeros(1, 1, -lag),
                                                 y_hat[b:b+1, :, :lag]], dim=2)
                else:
                    hat_shifted = y_hat[b:b+1]

                gt_mel_b  = y_mel[b:b+1]
                hat_mel_b = to_mel(hat_shifted)
                min_len = min(gt_mel_b.shape[2], hat_mel_b.shape[2])
                losses_aligned.append(
                    F.l1_loss(gt_mel_b[:, :, :min_len],
                              hat_mel_b[:, :, :min_len]).item() * hp.c_mel
                )

            lags.extend(batch_lags)
            print(f"[{i+1:3d}/{args.num_batches}] "
                  f"lag_mean={np.mean(np.abs(batch_lags)):.1f}s  "
                  f"unaligned={loss_u:.3f}  "
                  f"aligned={np.mean(losses_aligned[-len(batch_lags):]):.3f}")

    abs_lags = np.abs(lags)
    print()
    print("=" * 60)
    print(f"Lag stats (samples @ 22050 Hz):")
    print(f"  mean |lag| = {np.mean(abs_lags):.1f}  ({np.mean(abs_lags)/22050*1000:.2f} ms)")
    print(f"  median     = {np.median(abs_lags):.1f}  ({np.median(abs_lags)/22050*1000:.2f} ms)")
    print(f"  max        = {np.max(abs_lags):.0f}  ({np.max(abs_lags)/22050*1000:.2f} ms)")
    print(f"  lag=0      = {(np.array(lags)==0).mean()*100:.1f}% of utterances")
    print()
    print(f"Mel loss (×45):")
    print(f"  unaligned  = {np.mean(losses_unaligned):.4f}")
    print(f"  aligned    = {np.mean(losses_aligned):.4f}")
    print(f"  reduction  = {np.mean(losses_unaligned) - np.mean(losses_aligned):.4f}  "
          f"({(np.mean(losses_unaligned)-np.mean(losses_aligned))/np.mean(losses_unaligned)*100:.1f}%)")
    print(f"  oracle     = ~1.38")
    print(f"Utterances   = {len(lags)}")


if __name__ == "__main__":
    main()
