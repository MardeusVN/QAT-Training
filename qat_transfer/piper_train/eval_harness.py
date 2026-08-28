#!/usr/bin/env python3
"""WER + UTMOS evaluation harness for comparing one or more Piper/VITS checkpoints.

Synthesizes a fixed sentence set on CPU (so it never competes with an active
training run for GPU memory), scores each utterance with Whisper (WER) and
UTMOSv2 (naturalness), and — when 2+ checkpoints are given — runs a paired
Wilcoxon signed-rank test between every pair of checkpoints on both metrics.

Usage:
    python3 -m piper_train.eval_harness \
        --checkpoint lightning_logs/version_8/checkpoints/epoch=124-step=*.ckpt \
        --checkpoint lightning_logs/version_10/checkpoints/epoch=217-step=*.ckpt \
        --sentences-file etc/test_sentences/en.txt \
        --output eval_results/
"""
import argparse
import csv
import json
import logging
import pathlib
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import jiwer
import librosa
import numpy as np
import torch
import whisper
from piper_phonemize import phoneme_ids_espeak, phonemize_espeak
from scipy.stats import wilcoxon

from .vits.lightning import VitsModel
from .vits.utils import audio_float_to_int16
from .vits.wavfile import write as write_wav

_LOGGER = logging.getLogger("piper_train.eval_harness")

torch.serialization.add_safe_globals([pathlib.PosixPath])

WHISPER_SAMPLE_RATE = 16000
_NORM_RE = re.compile(r"[^a-z0-9' ]+")


@dataclass
class UtteranceResult:
    text: str
    wer: float
    utmos: float
    infer_sec: float
    audio_duration_sec: float
    real_time_factor: float


@dataclass
class CheckpointReport:
    checkpoint: str
    results: List[UtteranceResult]
    mean_wer: float
    median_wer: float
    mean_utmos: float
    median_utmos: float
    mean_rtf: float
    f0_rmse: Optional[float] = None
    f0_corr: Optional[float] = None


def text_to_phoneme_ids(text: str, language: str = "en-us"):
    sentences_phonemes = phonemize_espeak(text, language)
    phonemes = [p for sentence in sentences_phonemes for p in sentence]
    return phoneme_ids_espeak(phonemes)


def normalize_for_wer(text: str) -> str:
    text = text.lower().strip()
    text = _NORM_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_sentences(
    sentences_file: Optional[pathlib.Path],
    dataset_jsonl: Optional[pathlib.Path],
    num_dataset_samples: int,
    seed: int,
) -> List[str]:
    sentences: List[str] = []

    if sentences_file is not None:
        with open(sentences_file, "r", encoding="utf-8") as f:
            sentences.extend(line.strip() for line in f if line.strip())

    if dataset_jsonl is not None and num_dataset_samples > 0:
        candidates = []
        with open(dataset_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt = json.loads(line)
                text = utt.get("text")
                if text:
                    candidates.append(text)

        rng = np.random.default_rng(seed)
        if len(candidates) > num_dataset_samples:
            idxs = rng.choice(len(candidates), size=num_dataset_samples, replace=False)
            candidates = [candidates[i] for i in idxs]
        sentences.extend(candidates)

    if not sentences:
        raise ValueError("No sentences loaded — provide --sentences-file and/or --dataset-jsonl")

    return sentences


def synthesize(model: VitsModel, text: str, language: str, scales, sample_rate: int):
    phoneme_ids = text_to_phoneme_ids(text, language)
    text_tensor = torch.LongTensor(phoneme_ids).unsqueeze(0)
    text_lengths = torch.LongTensor([len(phoneme_ids)])

    start_time = time.perf_counter()
    with torch.no_grad():
        audio = model(text_tensor, text_lengths, scales, sid=None).detach().numpy()
    infer_sec = time.perf_counter() - start_time

    audio = np.squeeze(audio).astype(np.float32)
    audio_duration_sec = audio.shape[-1] / sample_rate
    rtf = infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
    return audio, infer_sec, audio_duration_sec, rtf


def collect_f0_pairs(
    model: VitsModel,
    dataset_entries: List[dict],
) -> tuple:
    """Aggregate predicted vs GT log-F0 pairs across all entries for global metrics.

    Predicted phoneme-level log-F0 comes from model.f0_predictor; GT phoneme-level
    log-F0 is derived by averaging the cached frame-level F0 over intervals defined
    by the SDP predicted durations (same projection used during training via MAS).

    Returns (f0_rmse, f0_corr) or (None, None) if model has no F0 predictor.
    """
    if not model.model_g.use_f0:
        return None, None

    all_pred: List[float] = []
    all_gt: List[float] = []

    for entry in dataset_entries:
        phoneme_ids = entry.get("phoneme_ids")
        f0_path_str = entry.get("audio_f0_path")
        if not phoneme_ids or not f0_path_str:
            continue
        f0_path = pathlib.Path(f0_path_str)
        if not f0_path.exists():
            continue

        try:
            ids_t = torch.LongTensor(phoneme_ids).unsqueeze(0)
            lengths_t = torch.LongTensor([len(phoneme_ids)])

            with torch.no_grad():
                x, _, _, x_mask = model.model_g.enc_p(ids_t, lengths_t)
                log_f0_pred = model.model_g.f0_predictor(x, x_mask).squeeze().numpy()

                logw = model.model_g.dp(x, x_mask, reverse=True, noise_scale=0.8)
                logw = torch.nan_to_num(logw, nan=0.0).clamp(-6.0, 6.0)
                w = torch.exp(logw) * x_mask
                durations = torch.ceil(w).clamp(min=0, max=500).squeeze().long().numpy()

            gt_f0_frame = torch.load(f0_path, weights_only=True).numpy()

            frame_ptr = 0
            phoneme_gt: List[float] = []
            for dur in durations:
                dur = int(dur)
                if dur <= 0 or frame_ptr >= len(gt_f0_frame):
                    phoneme_gt.append(1.0)
                else:
                    end = min(frame_ptr + dur, len(gt_f0_frame))
                    phoneme_gt.append(float(np.mean(gt_f0_frame[frame_ptr:end])))
                    frame_ptr = end

            log_gt = np.log(np.maximum(np.array(phoneme_gt, dtype=np.float32), 1.0))
            n = min(len(log_f0_pred), len(log_gt))
            all_pred.extend(log_f0_pred[:n].tolist())
            all_gt.extend(log_gt[:n].tolist())

        except Exception as exc:
            _LOGGER.warning("F0 eval skipped for one entry: %s", exc)

    if len(all_pred) < 2:
        return None, None

    pred_arr = np.array(all_pred, dtype=np.float32)
    gt_arr = np.array(all_gt, dtype=np.float32)
    rmse = float(np.sqrt(np.mean((pred_arr - gt_arr) ** 2)))
    corr = (
        float(np.corrcoef(pred_arr, gt_arr)[0, 1])
        if np.std(pred_arr) > 1e-9 and np.std(gt_arr) > 1e-9
        else 0.0
    )
    return rmse, corr


def evaluate_checkpoint(
    checkpoint: pathlib.Path,
    sentences: List[str],
    whisper_model,
    utmos_model,
    language: str,
    scales,
    sample_rate: int,
    wav_out_dir: Optional[pathlib.Path],
    dataset_entries: Optional[List[dict]] = None,
) -> CheckpointReport:
    _LOGGER.info("Loading checkpoint: %s", checkpoint)
    model = VitsModel.load_from_checkpoint(
        str(checkpoint), dataset=None, strict=False, map_location="cpu"
    )
    model.eval()
    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()

    if wav_out_dir is not None:
        wav_out_dir.mkdir(parents=True, exist_ok=True)

    results: List[UtteranceResult] = []
    for idx, text in enumerate(sentences):
        audio, infer_sec, audio_duration_sec, rtf = synthesize(
            model, text, language, scales, sample_rate
        )

        if wav_out_dir is not None:
            write_wav(
                str(wav_out_dir / f"sent_{idx:04d}.wav"),
                sample_rate,
                audio_float_to_int16(audio),
            )

        audio_16k = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE
        )
        whisper_result = whisper_model.transcribe(audio_16k, language="en", fp16=False)
        hypothesis = normalize_for_wer(whisper_result["text"])
        reference = normalize_for_wer(text)
        utt_wer = jiwer.wer(reference, hypothesis) if reference else 1.0

        utmos_score = float(
            utmos_model.predict(data=audio, sr=sample_rate, device="cpu")
        )

        _LOGGER.info(
            "[%d/%d] wer=%.3f utmos=%.3f rtf=%.3f :: %s",
            idx + 1,
            len(sentences),
            utt_wer,
            utmos_score,
            rtf,
            text[:60],
        )

        results.append(
            UtteranceResult(
                text=text,
                wer=utt_wer,
                utmos=utmos_score,
                infer_sec=infer_sec,
                audio_duration_sec=audio_duration_sec,
                real_time_factor=rtf,
            )
        )

    wers = [r.wer for r in results]
    utmoses = [r.utmos for r in results]
    rtfs = [r.real_time_factor for r in results]

    f0_rmse, f0_corr = None, None
    if dataset_entries:
        _LOGGER.info("Computing F0 metrics over %d dataset entries...", len(dataset_entries))
        f0_rmse, f0_corr = collect_f0_pairs(model, dataset_entries)
        if f0_rmse is not None:
            _LOGGER.info("F0 RMSE=%.4f  F0 Corr=%.4f", f0_rmse, f0_corr)

    return CheckpointReport(
        checkpoint=str(checkpoint),
        results=results,
        mean_wer=float(np.mean(wers)),
        median_wer=float(np.median(wers)),
        mean_utmos=float(np.mean(utmoses)),
        median_utmos=float(np.median(utmoses)),
        mean_rtf=float(np.mean(rtfs)),
        f0_rmse=f0_rmse,
        f0_corr=f0_corr,
    )


def paired_tests(reports: List[CheckpointReport]) -> List[dict]:
    comparisons = []
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            a, b = reports[i], reports[j]
            wer_a = [r.wer for r in a.results]
            wer_b = [r.wer for r in b.results]
            utmos_a = [r.utmos for r in a.results]
            utmos_b = [r.utmos for r in b.results]

            entry = {"checkpoint_a": a.checkpoint, "checkpoint_b": b.checkpoint}
            try:
                stat, p = wilcoxon(wer_a, wer_b)
                entry["wer_wilcoxon_stat"] = float(stat)
                entry["wer_wilcoxon_p"] = float(p)
            except ValueError as e:
                entry["wer_wilcoxon_error"] = str(e)
            try:
                stat, p = wilcoxon(utmos_a, utmos_b)
                entry["utmos_wilcoxon_stat"] = float(stat)
                entry["utmos_wilcoxon_p"] = float(p)
            except ValueError as e:
                entry["utmos_wilcoxon_error"] = str(e)

            comparisons.append(entry)
    return comparisons


def write_report(report: CheckpointReport, output_dir: pathlib.Path):
    stem = pathlib.Path(report.checkpoint).stem
    csv_path = output_dir / f"{stem}_per_sentence.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "wer", "utmos", "infer_sec", "audio_duration_sec", "rtf"])
        for r in report.results:
            writer.writerow(
                [r.text, r.wer, r.utmos, r.infer_sec, r.audio_duration_sec, r.real_time_factor]
            )
    _LOGGER.info("Wrote %s", csv_path)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="piper_train.eval_harness")
    parser.add_argument(
        "--checkpoint", required=True, action="append", help="Path to .ckpt (repeatable)"
    )
    parser.add_argument("--sentences-file", type=pathlib.Path, default=None)
    parser.add_argument("--dataset-jsonl", type=pathlib.Path, default=None)
    parser.add_argument("--num-dataset-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="en-us")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-w", type=float, default=0.8)
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument("--save-wavs", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(
        args.sentences_file, args.dataset_jsonl, args.num_dataset_samples, args.seed
    )
    _LOGGER.info("Loaded %d sentences", len(sentences))

    # Load full dataset entries for F0 evaluation (only when JSONL is provided)
    dataset_entries: Optional[List[dict]] = None
    if args.dataset_jsonl is not None and args.dataset_jsonl.exists():
        dataset_entries = []
        with open(args.dataset_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset_entries.append(json.loads(line))
        _LOGGER.info("Loaded %d dataset entries for F0 eval", len(dataset_entries))

    _LOGGER.info("Loading Whisper model: %s (CPU)", args.whisper_model)
    whisper_model = whisper.load_model(args.whisper_model, device="cpu")

    _LOGGER.info("Loading UTMOSv2 model (CPU)")
    import utmosv2

    utmos_model = utmosv2.create_model(pretrained=True, device="cpu")

    scales = [args.noise_scale, args.length_scale, args.noise_w]

    reports = []
    for checkpoint in args.checkpoint:
        checkpoint_path = pathlib.Path(checkpoint)
        wav_out_dir = (
            args.output / pathlib.Path(checkpoint).stem if args.save_wavs else None
        )
        report = evaluate_checkpoint(
            checkpoint_path,
            sentences,
            whisper_model,
            utmos_model,
            args.language,
            scales,
            args.sample_rate,
            wav_out_dir,
            dataset_entries=dataset_entries,
        )
        write_report(report, args.output)
        reports.append(report)
        _LOGGER.info(
            "Checkpoint %s :: mean_wer=%.3f mean_utmos=%.3f mean_rtf=%.3f",
            checkpoint,
            report.mean_wer,
            report.mean_utmos,
            report.mean_rtf,
        )

    summary = {
        "checkpoints": [
            {
                "checkpoint": r.checkpoint,
                "mean_wer": r.mean_wer,
                "median_wer": r.median_wer,
                "mean_utmos": r.mean_utmos,
                "median_utmos": r.median_utmos,
                "mean_rtf": r.mean_rtf,
                "f0_rmse": r.f0_rmse,
                "f0_corr": r.f0_corr,
                "num_sentences": len(r.results),
            }
            for r in reports
        ],
        "paired_tests": paired_tests(reports) if len(reports) > 1 else [],
    }

    summary_path = args.output / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _LOGGER.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
