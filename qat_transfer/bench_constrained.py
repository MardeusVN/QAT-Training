"""Benchmark FP32 vs QOperator PTQ vs QDQ(QAT) under a 1-core/2-thread CPU
budget -- the actual edge deployment target, not this dev machine's 6c/12t.
ONNX Runtime's own intra_op/inter_op thread settings are what matters here
(they cap the session's internal thread pool regardless of how many
physical cores the host has), so this is a faithful simulation without
needing OS-level CPU affinity pinning.
"""
import json
import time

import numpy as np
import onnxruntime as ort


def make_session(path, num_threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def bench(path, ids, num_threads, n_warmup=2, n_runs=8):
    sess = make_session(path, num_threads)
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
    times = np.array(times)
    audio_sec = out[0].shape[-1] / 22050
    return times.mean(), audio_sec


def main():
    entries = []
    with open("../data_preprocessed/dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    e = entries[-3]  # same 297-phoneme sentence used before
    ids = np.array(e["phoneme_ids"], dtype=np.int64)[None, :]
    print(f"Sentence: {e['text'][:60]} ({ids.shape[1]} phonemes)")
    print()

    models = [
        ("FP32", "fp32_fully_fixed.onnx"),
        ("QOperator PTQ (full)", "qoperator_fully_fixed.onnx"),
        ("QOperator PTQ (flow excluded)", "noflow_qoperator_v2.onnx"),
        ("QDQ (QAT run1)", "qdq_fully_fixed.onnx"),
    ]

    for threads in (2, 1):
        print(f"=== intra_op_num_threads={threads} ===")
        results = {}
        for name, path in models:
            mean_t, audio_sec = bench(path, ids, threads)
            results[name] = mean_t
            print(f"  {name:32s}: {mean_t*1000:8.1f} ms  RTF={mean_t/audio_sec:.4f}")
        fp32_t = results["FP32"]
        print("  --- relative to FP32 ---")
        for name, t in results.items():
            print(f"  {name:32s}: {fp32_t/t:.2f}x")
        print()


if __name__ == "__main__":
    main()
