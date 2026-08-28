#!/usr/bin/env python3
"""Portable FP32 vs INT8 speed benchmark for a Piper VITS voice, run on
whatever machine you copy this folder to. Only needs onnxruntime + numpy
(no torch, no training environment).

Setup on the target machine:
    pip install onnxruntime numpy

Usage:
    python bench_vnni.py

What this measures: wall-clock latency for both voice.fp32.onnx and
voice.int8.onnx, at intra_op_num_threads = 1 and 2 (the "1 core / 2
threads" edge budget this was built for), across the 5 sentences in
test_sentences.json. Compares against the same benchmark run on the
original dev machine (Intel i5-10400F, no AVX-VNNI) to see whether a
VNNI-capable CPU (Intel 12th gen+ / AMD Zen4+) gives INT8 a bigger real
speedup than the ~3-8% measured there.

To confirm whether AVX-VNNI is actually available on this machine (Linux):
    cat /proc/cpuinfo | grep -o 'avx_vnni' | head -1
On Windows, there's no simple built-in check; the CPU model name (e.g. via
`wmic cpu get name` or Task Manager) combined with Intel/AMD's published
spec sheet for that model is the reliable way to confirm.
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).parent


def make_session(path: str, num_threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def bench_one(sess: ort.InferenceSession, phoneme_ids, n_warmup=2, n_runs=8):
    ids = np.array(phoneme_ids, dtype=np.int64)[None, :]
    feed = {
        "input": ids,
        "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
    }
    if any(i.name == "sid" for i in sess.get_inputs()):
        feed["sid"] = np.array([], dtype=np.int64)

    for _ in range(n_warmup):
        sess.run(None, feed)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = sess.run(None, feed)
        times.append(time.perf_counter() - t0)

    audio_sec = out[0].shape[-1] / 22050
    times = np.array(times)
    return times.mean(), audio_sec


def main():
    with open(HERE / "test_sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)

    models = [
        ("FP32", str(HERE / "voice.fp32.onnx")),
        ("INT8", str(HERE / "voice.int8.onnx")),
    ]

    for threads in (2, 1):
        print(f"\n{'='*70}")
        print(f"intra_op_num_threads = {threads}")
        print(f"{'='*70}")

        per_model_means = {}
        for name, path in models:
            print(f"\n--- {name} ({path}) ---")
            sess = make_session(path, threads)
            rtfs = []
            means = []
            for s in sentences:
                mean_t, audio_sec = bench_one(sess, s["phoneme_ids"])
                rtf = mean_t / audio_sec
                rtfs.append(rtf)
                means.append(mean_t)
                print(f"  {len(s['phoneme_ids']):4d} phon -> {mean_t*1000:7.1f} ms  "
                      f"RTF={rtf:.4f}  (x{1/rtf:.1f} real-time)  :: {s['text'][:45]}")
            per_model_means[name] = np.mean(means)
            print(f"  Average RTF: {np.mean(rtfs):.4f}")

        fp32_t = per_model_means["FP32"]
        int8_t = per_model_means["INT8"]
        speedup = fp32_t / int8_t
        print(f"\n  INT8 vs FP32 speedup @ {threads} thread(s): "
              f"{speedup:.2f}x ({'FASTER' if speedup > 1 else 'SLOWER'})")


if __name__ == "__main__":
    main()
