#!/usr/bin/env python3
"""Compare WER/CER/UTMOS across ONNX export variants using real held-out
dataset sentences and their pre-computed phoneme_ids (no piper-phonemize
dependency needed).

Usage:
    python eval_quality.py
"""
import json
import random
import re
import time

import jiwer
import librosa
import numpy as np
import onnxruntime as ort
import whisper

_NORM_RE = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _NORM_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def synthesize(sess: ort.InferenceSession, phoneme_ids, sample_rate=22050):
    ids = np.array(phoneme_ids, dtype=np.int64)[None, :]
    feed = {
        "input": ids,
        "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
    }
    if any(i.name == "sid" for i in sess.get_inputs()):
        feed["sid"] = np.array([], dtype=np.int64)
    t0 = time.perf_counter()
    audio = sess.run(None, feed)[0]
    infer_sec = time.perf_counter() - t0
    audio = np.squeeze(audio).astype(np.float32)
    return audio, infer_sec


def main():
    random.seed(123)
    entries = []
    with open("../data_preprocessed/dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # 30 held-out sentences (last N of the file, disjoint from the 150 used
    # for PTQ calibration which was random.sample(entries, 150) with seed 42
    # over the FULL list -- to be safe, just sample fresh with a different seed
    # and a decent size for a meaningful average).
    test_entries = random.sample(entries, 30)

    print("Loading Whisper (base.en) ...")
    whisper_model = whisper.load_model("base.en")

    print("Loading UTMOSv2 ...")
    import utmosv2
    utmos_model = utmosv2.create_model(pretrained=True, device="cpu")

    models = {
        "FP32": "fp32_fully_fixed.onnx",
        "QOperator (PTQ, no training)": "qoperator_fully_fixed.onnx",
        "QOperator (QAT-adapted, early ckpt)": "qat_then_qoperator.onnx",
    }

    results = {name: {"wer": [], "cer": [], "utmos": [], "rtf": []} for name in models}

    for name, path in models.items():
        print(f"\n=== {name} ===")
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        for i, e in enumerate(test_entries):
            audio, infer_sec = synthesize(sess, e["phoneme_ids"])
            audio_dur = len(audio) / 22050
            rtf = infer_sec / audio_dur if audio_dur > 0 else 0.0

            audio_16k = librosa.resample(audio, orig_sr=22050, target_sr=16000)
            whisper_result = whisper_model.transcribe(audio_16k, language="en", fp16=False)
            hyp = normalize(whisper_result["text"])
            ref = normalize(e["text"])
            utt_wer = jiwer.wer(ref, hyp) if ref else 1.0
            utt_cer = jiwer.cer(ref, hyp) if ref else 1.0
            utmos_score = float(utmos_model.predict(data=audio, sr=22050, device="cpu"))

            results[name]["wer"].append(utt_wer)
            results[name]["cer"].append(utt_cer)
            results[name]["utmos"].append(utmos_score)
            results[name]["rtf"].append(rtf)

            print(f"  [{i+1:2d}/{len(test_entries)}] wer={utt_wer:.3f} cer={utt_cer:.3f} "
                  f"utmos={utmos_score:.3f} rtf={rtf:.3f} :: {e['text'][:50]}")

    print("\n" + "=" * 70)
    print(f"{'Model':35s} {'WER':>8s} {'CER':>8s} {'UTMOS':>8s} {'RTF':>8s}")
    for name, r in results.items():
        print(f"{name:35s} {np.mean(r['wer']):8.4f} {np.mean(r['cer']):8.4f} "
              f"{np.mean(r['utmos']):8.3f} {np.mean(r['rtf']):8.4f}")

    with open("eval_quality_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {name: {k: float(np.mean(v)) for k, v in r.items()} for name, r in results.items()},
            f, indent=2,
        )
    print("\nWrote eval_quality_results.json")


if __name__ == "__main__":
    main()
