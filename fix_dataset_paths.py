#!/usr/bin/env python3
"""One-time fix: rewrite dataset.jsonl cache paths from the original Linux
(/home/dev/...) machine to this Windows checkout's actual location, and
verify every referenced .pt file exists."""
import json
import shutil
from pathlib import Path

DATASET_DIR = Path(r"D:\QAT_Transfer\data_preprocessed")
DATASET_JSONL = DATASET_DIR / "dataset.jsonl"
OLD_PREFIX = "/home/dev/data_preprocessed"
NEW_PREFIX = DATASET_DIR.as_posix()

PATH_FIELDS = ("audio_norm_path", "audio_spec_path", "audio_f0_path")


def fix_path(p: str) -> str:
    if p.startswith(OLD_PREFIX):
        return NEW_PREFIX + p[len(OLD_PREFIX):]
    return p


def main():
    backup = DATASET_JSONL.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copy2(DATASET_JSONL, backup)
        print(f"Backed up original to {backup}")
    else:
        print(f"Backup already exists at {backup}, not overwriting")

    lines_out = []
    n_missing = 0
    n_total = 0
    with open(backup, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt = json.loads(line)
            n_total += 1
            for field in PATH_FIELDS:
                if field in utt and utt[field]:
                    utt[field] = fix_path(utt[field])
                    if not Path(utt[field]).exists():
                        n_missing += 1
                        print(f"MISSING: {utt[field]}")
            lines_out.append(json.dumps(utt, ensure_ascii=False))

    with open(DATASET_JSONL, "w", encoding="utf-8") as f:
        for line in lines_out:
            f.write(line + "\n")

    print(f"Rewrote {n_total} entries. Missing files: {n_missing}")


if __name__ == "__main__":
    main()
