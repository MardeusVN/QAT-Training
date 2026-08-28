#!/usr/bin/env python3
"""Extract the exact 500 test sentences used during training (same random_split seed).

Outputs:
  <output-txt>    — one sentence per line (for eval_harness --sentences-file)
  <output-jsonl>  — full entries with audio paths (for compute_reference_metrics)
"""
import argparse
import json
import pathlib

import torch
from torch.utils.data import random_split

from .vits.dataset import PiperDataset


def main():
    parser = argparse.ArgumentParser(prog="piper_train.extract_test_sentences")
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=pathlib.Path("/home/dev/data_preprocessed"))
    parser.add_argument("--output-txt",  type=pathlib.Path, default=pathlib.Path("/home/dev/test_sentences_500.txt"))
    parser.add_argument("--output-jsonl", type=pathlib.Path, default=pathlib.Path("/home/dev/test_entries_500.jsonl"))
    parser.add_argument("--num-val",  type=int, default=100)
    parser.add_argument("--num-test", type=int, default=500)
    parser.add_argument("--seed",     type=int, default=1234)
    args = parser.parse_args()

    full_dataset = PiperDataset([args.dataset_dir / "dataset.jsonl"])
    n = len(full_dataset)
    train_size = n - args.num_val - args.num_test

    print(f"Total dataset: {n} examples")
    print(f"Split: train={train_size}, test={args.num_test}, val={args.num_val}")

    _, test_dataset, _ = random_split(
        full_dataset,
        [train_size, args.num_test, args.num_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    texts = []
    with open(args.output_jsonl, "w", encoding="utf-8") as jf:
        for idx in test_dataset.indices:
            utt = full_dataset.utterances[idx]   # Utterance has audio_norm_path
            texts.append(utt.text)
            jf.write(json.dumps({"text": utt.text, "audio_norm_path": str(utt.audio_norm_path)}) + "\n")

    args.output_txt.write_text("\n".join(texts), encoding="utf-8")
    print(f"Wrote {len(texts)} sentences to {args.output_txt}")
    print(f"Wrote {len(texts)} entries   to {args.output_jsonl}")


if __name__ == "__main__":
    main()
