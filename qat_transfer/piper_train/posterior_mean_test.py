#!/usr/bin/env python3
"""Compare mel reconstruction loss: z sampled vs z = posterior mean.

Quantifies how much of the val_loss_mel gap is due to sampling noise
in the posterior encoder (VAE reparameterization) vs generator capacity.

Usage:
    python -m piper_train.posterior_mean_test \
        --checkpoint /home/dev/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt \
        --dataset-dir /home/dev/data_preprocessed \
        --num-batches 50
"""
import argparse
import pathlib

import torch
import torch.nn.functional as F

torch.serialization.add_safe_globals([pathlib.PosixPath])
from torch.utils.data import DataLoader, random_split

from .vits.dataset import PiperDataset, UtteranceCollate
from .vits.lightning import VitsModel
from .vits.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .vits.commons import rand_slice_segments, slice_segments

SEED = 1234
HOP_LENGTH = 256
SEGMENT_SIZE = 8192


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-dir", type=pathlib.Path,
                        default=pathlib.Path("/home/dev/data_preprocessed"))
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-val", type=int, default=100)
    parser.add_argument("--num-test", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
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

    losses_sample, losses_mean = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break

            spec = batch.spectrograms
            spec_lengths = batch.spectrogram_lengths
            y = batch.audios
            y_lengths = batch.audio_lengths

            # Posterior encoder → m_q, logs_q
            z_sampled, m_q, logs_q, y_mask = model.model_g.enc_q(spec, spec_lengths)
            # z_mean: no noise
            z_mean = m_q * y_mask

            # Use same slice indices for fair comparison
            _, ids_slice = rand_slice_segments(
                z_sampled, y_lengths // HOP_LENGTH, seg_frames
            )

            # Decode both
            z_s_slice = slice_segments(z_sampled, ids_slice, seg_frames)
            z_m_slice = slice_segments(z_mean,    ids_slice, seg_frames)

            y_hat_sample = model.model_g.dec(z_s_slice)
            y_hat_mean   = model.model_g.dec(z_m_slice)

            # GT mel (path A: pre-computed spec → mel → slice)
            mel_full = spec_to_mel_torch(
                spec.float(),
                hp.filter_length, hp.mel_channels,
                hp.sample_rate, hp.mel_fmin, hp.mel_fmax,
            )
            y_mel = slice_segments(mel_full, ids_slice, seg_frames)

            # Generated mel
            def to_mel(wav):
                return mel_spectrogram_torch(
                    wav.float().squeeze(1),
                    hp.filter_length, hp.mel_channels, hp.sample_rate,
                    hp.hop_length, hp.win_length, hp.mel_fmin, hp.mel_fmax,
                )

            loss_s = F.l1_loss(y_mel, to_mel(y_hat_sample)).item() * hp.c_mel
            loss_m = F.l1_loss(y_mel, to_mel(y_hat_mean)).item()   * hp.c_mel

            losses_sample.append(loss_s)
            losses_mean.append(loss_m)

            if (i + 1) % 10 == 0:
                print(f"[{i+1:3d}/{args.num_batches}] "
                      f"sample={sum(losses_sample)/len(losses_sample):.4f}  "
                      f"mean={sum(losses_mean)/len(losses_mean):.4f}")

    avg_s = sum(losses_sample) / len(losses_sample)
    avg_m = sum(losses_mean)   / len(losses_mean)

    print()
    print("=" * 55)
    print(f"z = sampled  (training mode)  : {avg_s:.4f}")
    print(f"z = mean     (no noise)       : {avg_m:.4f}")
    print(f"Sampling noise contribution   : {avg_s - avg_m:.4f}  ({(avg_s-avg_m)/avg_s*100:.1f}%)")
    print(f"Oracle floor (×45)            : ~1.38")
    print(f"Generator gap (mean path)     : {avg_m - 1.38:.4f}")


if __name__ == "__main__":
    main()
