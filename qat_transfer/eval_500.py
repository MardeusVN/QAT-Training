#!/usr/bin/env python3
"""Eval a single model on 500 sampled sentences (WER/CER/UTMOS/RTF)."""
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
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    random.seed(500)
    entries = []
    with open("../data_preprocessed/dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    test_entries = random.sample(entries, n_samples)

    print(f"Loading Whisper (base.en) ...", flush=True)
    whisper_model = whisper.load_model("base.en")
    print("Loading UTMOSv2 ...", flush=True)
    import utmosv2
    utmos_model = utmosv2.create_model(pretrained=True, device="cpu")

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    wers, cers, utmoses, rtfs = [], [], [], []

    t_start = time.perf_counter()
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

        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (i + 1) * (len(test_entries) - i - 1)
            print(f"  [{i+1:4d}/{len(test_entries)}] running_wer={np.mean(wers):.4f} "
                  f"running_utmos={np.mean(utmoses):.3f}  elapsed={elapsed/60:.1f}min "
                  f"eta={eta/60:.1f}min", flush=True)

    summary = {
        "n": len(test_entries),
        "wer": float(np.mean(wers)),
        "cer": float(np.mean(cers)),
        "utmos": float(np.mean(utmoses)),
        "rtf": float(np.mean(rtfs)),
        "wer_median": float(np.median(wers)),
        "utmos_median": float(np.median(utmoses)),
    }
    print()
    print(f"{model_name} (n={summary['n']}): WER={summary['wer']:.4f} CER={summary['cer']:.4f} "
          f"UTMOS={summary['utmos']:.3f} RTF={summary['rtf']:.4f}")

    try:
        with open("eval_500_results.json", encoding="utf-8") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}
    all_results[model_name] = summary
    with open("eval_500_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("Updated eval_500_results.json")


if __name__ == "__main__":
    main()
