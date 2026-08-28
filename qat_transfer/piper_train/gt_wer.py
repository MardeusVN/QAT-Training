#!/usr/bin/env python3
"""Measure WER of ground-truth audio through the same Whisper pipeline
used by eval_harness, to establish a GT baseline for intelligibility.

Usage:
    python -m piper_train.gt_wer \
        --test-jsonl /home/dev/test_entries_500.jsonl \
        --num-sentences 100
"""
import argparse
import json
import pathlib
import re

import jiwer
import librosa
import numpy as np
import torch
import whisper

WHISPER_SAMPLE_RATE = 16000
_NORM_RE = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _NORM_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", type=pathlib.Path,
                        default=pathlib.Path("/home/dev/test_entries_500.jsonl"))
    parser.add_argument("--num-sentences", type=int, default=100)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--language", default="vi")
    args = parser.parse_args()

    entries = []
    with open(args.test_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries = entries[: args.num_sentences]
    print(f"Evaluating {len(entries)} GT utterances with Whisper-{args.whisper_model}")

    wmodel = whisper.load_model(args.whisper_model)

    wers = []
    for idx, entry in enumerate(entries):
        gt_path = pathlib.Path(entry["audio_norm_path"])
        if not gt_path.exists():
            print(f"[{idx+1}] MISSING: {gt_path}")
            continue

        audio = torch.load(gt_path, weights_only=True).numpy().squeeze()
        audio_16k = librosa.resample(audio, orig_sr=22050, target_sr=WHISPER_SAMPLE_RATE)

        result = wmodel.transcribe(audio_16k, language=args.language, fp16=False)
        hyp = normalize(result["text"])
        ref = normalize(entry["text"])
        utt_wer = jiwer.wer(ref, hyp) if ref else 1.0
        wers.append(utt_wer)

        if (idx + 1) % 20 == 0:
            print(f"[{idx+1:3d}/{len(entries)}] mean WER so far: {np.mean(wers):.3f}")

    print()
    print("=" * 50)
    print(f"GT mean WER   : {np.mean(wers):.4f}  ({np.mean(wers)*100:.1f}%)")
    print(f"GT median WER : {np.median(wers):.4f}  ({np.median(wers)*100:.1f}%)")
    print(f"Sentences     : {len(wers)}")
    print()
    print("Compare:")
    print(f"  GT audio WER      : {np.mean(wers)*100:.1f}%")
    print(f"  Synthesized WER   : ~8%  (from eval_harness)")
    print(f"  Gap               : {(0.08 - np.mean(wers))*100:+.1f} pp")


if __name__ == "__main__":
    main()
