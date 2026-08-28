#!/usr/bin/env python3
"""Measure CPU real-time factor for a VITS checkpoint.

Usage:
    python -m piper_train.measure_rtf \
        --checkpoint /home/dev/02_Baseline_VanillaVITS/.../best.ckpt \
        --sentences-file /home/dev/test_sentences_500.txt \
        --num-sentences 50
"""
import argparse
import pathlib
import time

import numpy as np
import torch

from .eval_harness import text_to_phoneme_ids
from .vits.lightning import VitsModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--sentences-file", type=pathlib.Path, required=True)
    parser.add_argument("--num-sentences", type=int, default=50)
    parser.add_argument("--language", default="vi")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    args = parser.parse_args()

    sentences = []
    with open(args.sentences_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    sentences = sentences[: args.num_sentences]
    print(f"Loaded {len(sentences)} sentences")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = VitsModel.load_from_checkpoint(
        str(args.checkpoint), dataset=None, strict=False, map_location="cpu"
    )
    model.eval()
    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()

    scales = torch.FloatTensor([args.noise_scale, args.noise_scale_w, args.length_scale])

    rtf_list, dur_list, infer_list = [], [], []
    for i, text in enumerate(sentences):
        phoneme_ids = text_to_phoneme_ids(text, args.language)
        ids_t = torch.LongTensor(phoneme_ids).unsqueeze(0)
        len_t = torch.LongTensor([len(phoneme_ids)])

        t0 = time.perf_counter()
        with torch.no_grad():
            audio = model(ids_t, len_t, scales, sid=None).detach().numpy()
        infer_sec = time.perf_counter() - t0

        audio_sec = audio.shape[-1] / args.sample_rate
        rtf = infer_sec / audio_sec if audio_sec > 0 else 0.0
        rtf_list.append(rtf)
        dur_list.append(audio_sec)
        infer_list.append(infer_sec)

        print(f"[{i+1:3d}/{len(sentences)}] rtf={rtf:.4f} (×{1/rtf:.1f} RT)  "
              f"audio={audio_sec:.2f}s  infer={infer_sec:.2f}s  :: {text[:50]}")

    print()
    print("=" * 50)
    print(f"Sentences   : {len(rtf_list)}")
    print(f"Mean RTF    : {np.mean(rtf_list):.4f}  →  ×{1/np.mean(rtf_list):.1f} real-time")
    print(f"Median RTF  : {np.median(rtf_list):.4f}  →  ×{1/np.median(rtf_list):.1f} real-time")
    print(f"Std RTF     : {np.std(rtf_list):.4f}")
    print(f"Total audio : {sum(dur_list):.1f}s generated in {sum(infer_list):.1f}s")


if __name__ == "__main__":
    main()
