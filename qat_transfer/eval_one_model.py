#!/usr/bin/env python3
"""Eval a single model against the same 30 held-out sentences (seed=123)
used by eval_quality.py, for apples-to-apples comparison without re-running
the already-scored models."""
import json
import random
import re
import sys
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


def main():
    model_name = sys.argv[1]
    model_path = sys.argv[2]

    random.seed(123)
    entries = []
    with open("../data_preprocessed/dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    test_entries = random.sample(entries, 30)

    print("Loading Whisper (base.en) ...")
    whisper_model = whisper.load_model("base.en")
    print("Loading UTMOSv2 ...")
    import utmosv2
    utmos_model = utmosv2.create_model(pretrained=True, device="cpu")

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    wers, cers, utmoses, rtfs = [], [], [], []

    for i, e in enumerate(test_entries):
        ids = np.array(e["phoneme_ids"], dtype=np.int64)[None, :]
        feed = {
            "input": ids,
            "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        }
        if any(inp.name == "sid" for inp in sess.get_inputs()):
            feed["sid"] = np.array([], dtype=np.int64)

        t0 = time.perf_counter()
        audio = sess.run(None, feed)[0]
        infer_sec = time.perf_counter() - t0
        audio = np.squeeze(audio).astype(np.float32)
        audio_dur = len(audio) / 22050
        rtf = infer_sec / audio_dur if audio_dur > 0 else 0.0

        audio_16k = librosa.resample(audio, orig_sr=22050, target_sr=16000)
        whisper_result = whisper_model.transcribe(audio_16k, language="en", fp16=False)
        hyp = normalize(whisper_result["text"])
        ref = normalize(e["text"])
        utt_wer = jiwer.wer(ref, hyp) if ref else 1.0
        utt_cer = jiwer.cer(ref, hyp) if ref else 1.0
        utmos_score = float(utmos_model.predict(data=audio, sr=22050, device="cpu"))

        wers.append(utt_wer)
        cers.append(utt_cer)
        utmoses.append(utmos_score)
        rtfs.append(rtf)
        print(f"  [{i+1:2d}/{len(test_entries)}] wer={utt_wer:.3f} cer={utt_cer:.3f} "
              f"utmos={utmos_score:.3f} rtf={rtf:.3f} :: {e['text'][:50]}")

    summary = {
        "wer": float(np.mean(wers)),
        "cer": float(np.mean(cers)),
        "utmos": float(np.mean(utmoses)),
        "rtf": float(np.mean(rtfs)),
    }
    print()
    print(f"{model_name}: WER={summary['wer']:.4f} CER={summary['cer']:.4f} "
          f"UTMOS={summary['utmos']:.3f} RTF={summary['rtf']:.4f}")

    try:
        with open("eval_quality_results.json", encoding="utf-8") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}
    all_results[model_name] = summary
    with open("eval_quality_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("Updated eval_quality_results.json")


if __name__ == "__main__":
    main()
