#!/usr/bin/env python3
"""Compute MCD and GT-UTMOS for generated WAVs against ground-truth audio.

Expects:
  --gen-wav-dir   : directory of generated WAVs named sent_0000.wav, sent_0001.wav ...
  --test-jsonl    : JSONL with fields {text, audio_norm_path} in same order as WAVs
  --output        : directory to write results (mcd_gtmos.csv, mcd_gtmos_summary.json)

MCD uses DTW alignment on 24-dim MFCC (C1–C24, C0 excluded) at 22050 Hz.
GT UTMOS runs the same UTMOSv2 predictor used in eval_harness.
"""
import argparse
import csv
import json
import logging
import pathlib

import librosa
import numpy as np
import torch
import utmosv2

_LOGGER = logging.getLogger("piper_train.compute_reference_metrics")

MCD_CONST = 10.0 * np.sqrt(2) / np.log(10)
N_MFCC = 24
SAMPLE_RATE = 22050


def extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return MFCC matrix [T, N_MFCC], C0 excluded."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC + 1, n_fft=1024, hop_length=256)
    return mfcc[1:].T  # drop C0, shape [T, N_MFCC]


def compute_gt_mfcc_stats(gt_paths: list, sr: int):
    """Compute global mean/std over GT corpus for normalization (1st pass)."""
    all_frames = []
    for p in gt_paths:
        audio = torch.load(p, weights_only=True).numpy().squeeze()
        all_frames.append(extract_mfcc(audio, sr))
    corpus = np.concatenate(all_frames, axis=0)  # [total_frames, N_MFCC]
    return corpus.mean(axis=0), corpus.std(axis=0) + 1e-8


def dtw_mcd(mfcc_gen: np.ndarray, mfcc_gt: np.ndarray,
            mean: np.ndarray, std: np.ndarray) -> float:
    """MCD via DTW on globally-normalized MFCCs (librosa compiled backend)."""
    norm_gen = (mfcc_gen - mean) / std
    norm_gt  = (mfcc_gt  - mean) / std
    D, wp = librosa.sequence.dtw(norm_gen.T, norm_gt.T, metric="euclidean")
    return float(MCD_CONST * D[-1, -1] / len(wp))


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="piper_train.compute_reference_metrics")
    parser.add_argument("--gen-wav-dir", type=pathlib.Path, required=True)
    parser.add_argument("--test-jsonl",  type=pathlib.Path, required=True)
    parser.add_argument("--output",      type=pathlib.Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(args.test_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Pass 1: compute global MFCC statistics from GT corpus for normalization
    gt_paths = [pathlib.Path(e["audio_norm_path"]) for e in entries
                if pathlib.Path(e["audio_norm_path"]).exists()]
    _LOGGER.info("Pass 1/2: computing GT MFCC stats over %d files...", len(gt_paths))
    mfcc_mean, mfcc_std = compute_gt_mfcc_stats(gt_paths, args.sample_rate)

    _LOGGER.info("Loading UTMOSv2 (CPU)")
    utmos_model = utmosv2.create_model(pretrained=True, device="cpu")

    mcd_list, gt_utmos_list = [], []
    n_skipped = 0

    _LOGGER.info("Pass 2/2: computing MCD + GT UTMOS...")
    csv_path = args.output / "mcd_gtmos.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["idx", "text", "mcd", "gt_utmos"])

        for idx, entry in enumerate(entries):
            gen_wav = args.gen_wav_dir / f"sent_{idx:04d}.wav"
            gt_path = pathlib.Path(entry["audio_norm_path"])

            if not gen_wav.exists():
                _LOGGER.warning("Missing gen WAV: %s — skipping", gen_wav)
                n_skipped += 1
                continue
            if not gt_path.exists():
                _LOGGER.warning("Missing GT audio: %s — skipping", gt_path)
                n_skipped += 1
                continue

            gen_audio, _ = librosa.load(gen_wav, sr=args.sample_rate, mono=True)
            gt_audio = torch.load(gt_path, weights_only=True).numpy().squeeze()

            mcd = dtw_mcd(extract_mfcc(gen_audio, args.sample_rate),
                          extract_mfcc(gt_audio,  args.sample_rate),
                          mfcc_mean, mfcc_std)
            gt_utmos = float(utmos_model.predict(data=gt_audio, sr=args.sample_rate, device="cpu"))

            mcd_list.append(mcd)
            gt_utmos_list.append(gt_utmos)

            _LOGGER.info(
                "[%d/%d] mcd=%.3f  gt_utmos=%.3f  :: %s",
                idx + 1, len(entries), mcd, gt_utmos, entry["text"][:60],
            )
            writer.writerow([idx, entry["text"], f"{mcd:.4f}", f"{gt_utmos:.4f}"])

    summary = {
        "mean_mcd":        float(np.mean(mcd_list)),
        "median_mcd":      float(np.median(mcd_list)),
        "mean_gt_utmos":   float(np.mean(gt_utmos_list)),
        "median_gt_utmos": float(np.median(gt_utmos_list)),
        "n":               len(mcd_list),
        "n_skipped":       n_skipped,
    }
    summary_path = args.output / "mcd_gtmos_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _LOGGER.info("=== SUMMARY ===")
    _LOGGER.info("MCD      : mean=%.3f  median=%.3f", summary["mean_mcd"], summary["median_mcd"])
    _LOGGER.info("GT UTMOS : mean=%.3f  median=%.3f", summary["mean_gt_utmos"], summary["median_gt_utmos"])
    if n_skipped:
        _LOGGER.warning("Skipped %d / %d entries", n_skipped, len(entries))
    _LOGGER.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
