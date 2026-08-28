#!/usr/bin/env python3
"""Export a Piper VITS checkpoint to a size/speed/quality-balanced INT8 ONNX
model for CPU edge deployment (validated for a 1-2 thread CPU budget).

This is plain post-training static quantization (PTQ) -- no QAT fine-tuning
needed. Empirically, on this VITS2+BigVGAN architecture, quantizing
*everything* hurts more than it helps (ONNX Runtime CPU has weak Conv1d
INT8 kernel support, and the normalizing flow + final output layer are
disproportionately quality-sensitive), while QAT fine-tuning several GPU-
hours costs about as much runtime speed as it buys back in quality. Simple
PTQ with two targeted exclusions turned out to dominate on every axis:

    Config                                   UTMOS  WER    Size    1-thread speed
    FP32 baseline                             3.85  7.7%   70.6MB  1.00x
    Quantize everything (PTQ)                 2.91  9.6%   21.6MB  1.06x
    Quantize everything except flow+conv_post 3.16  8.5%   46.4MB  1.08x  <- this script
    ^ + hours of QAT fine-tuning on top        3.22  9.3%   46.4MB  1.01x (not worth it)

Why exclude `flow` and `dec.conv_post`:
  - model_g.flow is a normalizing flow (32.9% of model_g's parameters).
    Invertible transforms are numerically sensitive -- errors compound
    across the 4 sequential coupling layers -- and quantizing it gave the
    single biggest quality hit of anything tested.
  - dec.conv_post is the last Conv1d before tanh, directly setting waveform
    sample values. Mixed-precision quantization literature consistently
    flags first/last layers as disproportionately sensitive; excluding this
    one tiny layer cost negligible size for a real quality gain.

Usage:
    python3 -m piper_train.export_int8 \
        qat_transfer/best-epoch=1079-val_loss_mel=19.2943.ckpt \
        qat_transfer/voice.int8.onnx \
        --calibration-sentences 60
"""
import argparse
import logging
import pathlib
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from .onnx_type_fix import fix_index_type_mismatches
from .vits.lightning import VitsModel
from .vits.quantize import remove_all_weight_norm

_LOGGER = logging.getLogger("piper_train.export_int8")
OPSET_VERSION = 15

torch.serialization.add_safe_globals([pathlib.PosixPath])

# Node name prefixes (as emitted by torch.onnx.export from this model's
# module structure) to exclude from INT8 quantization. See module docstring.
EXCLUDE_PREFIXES = ("/flow/",)
EXCLUDE_SUBSTRINGS = ("conv_post",)


class JsonlCalibReader(CalibrationDataReader):
    def __init__(self, dataset_jsonl: Path, n_samples: int, seed: int = 42):
        import json

        entries = []
        with open(dataset_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        rng = random.Random(seed)
        self.samples = rng.sample(entries, min(n_samples, len(entries)))
        self.i = 0

    def get_next(self):
        if self.i >= len(self.samples):
            return None
        entry = self.samples[self.i]
        self.i += 1
        ids = np.array(entry["phoneme_ids"], dtype=np.int64)[None, :]
        return {
            "input": ids,
            "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        }


def main() -> None:
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("output", help="Path to output INT8 model (.onnx)")
    parser.add_argument(
        "--calibration-dataset",
        help="Path to dataset.jsonl for calibration (default: alongside the "
        "checkpoint's own dataset, or pass explicitly)",
    )
    parser.add_argument(
        "--calibration-sentences", type=int, default=60,
        help="Number of real sentences to sample for calibration",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.calibration_dataset is None:
        raise SystemExit("--calibration-dataset is required (path to dataset.jsonl)")

    _LOGGER.info("Loading checkpoint: %s", args.checkpoint)
    model = VitsModel.load_from_checkpoint(
        args.checkpoint, dataset=None, strict=False, map_location="cpu",
        weights_only=False,
    )
    model_g = model.model_g
    remove_all_weight_norm(model_g)

    # Checkpoints from quantize_qat.py already have weight_norm removed
    # *before* the first training step (so the fake-quant wrapper can hold a
    # stable Parameter reference -- see quantize.py's remove_all_weight_norm
    # docstring), so their saved state_dict has plain "weight" keys instead
    # of "weight_g"/"weight_v". The load_from_checkpoint call above expects
    # the latter (freshly-constructed modules still have weight_norm
    # attached) and silently drops the former under strict=False, leaving
    # dec/flow at their random initialization. A raw reload against the
    # now-plain-weight module structure picks up whichever key shape the
    # checkpoint actually has; harmless no-op for checkpoints that still had
    # weight_norm; qat_transfer/export_onnx_qat.py hits and explains the same
    # issue in more detail.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = {
        k[len("model_g."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model_g.")
    }
    model_g.load_state_dict(state_dict, strict=False)

    model_g.eval()

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    def infer_forward(text, text_lengths, scales, sid=None):
        audio, *_ = model_g.infer(
            text, text_lengths, noise_scale=scales[0], length_scale=scales[1],
            noise_scale_w=scales[2], sid=sid,
        )
        return audio.unsqueeze(1) if audio.dim() == 2 else audio

    model_g.forward = infer_forward

    dummy_len = 50
    sequences = torch.randint(low=0, high=num_symbols, size=(1, dummy_len), dtype=torch.long)
    sequence_lengths = torch.LongTensor([sequences.size(1)])
    sid: Optional[torch.LongTensor] = torch.LongTensor([0]) if num_speakers > 1 else None
    scales = torch.FloatTensor([0.667, 1.0, 0.8])

    fp32_path = args.output.with_suffix(".fp32_tmp.onnx")
    _LOGGER.info("Exporting FP32 ONNX graph...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model=model_g,
            args=(sequences, sequence_lengths, scales, sid),
            f=str(fp32_path),
            opset_version=OPSET_VERSION,
            input_names=["input", "input_lengths", "scales", "sid"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "phonemes"},
                "input_lengths": {0: "batch_size"},
                "output": {0: "batch_size", 1: "time"},
            },
            dynamo=False,
        )

    _LOGGER.info("Fixing int32/int64 index-type mismatches (torch.onnx exporter quirk)...")
    onnx_model = onnx.load(str(fp32_path))
    n_fixed = fix_index_type_mismatches(onnx_model)
    _LOGGER.info("Fixed %d node(s)", n_fixed)

    all_conv = [n.name for n in onnx_model.graph.node if n.op_type in ("Conv", "ConvTranspose")]
    exclude_nodes = [
        n for n in all_conv
        if n.startswith(EXCLUDE_PREFIXES) or any(s in n for s in EXCLUDE_SUBSTRINGS)
    ]
    _LOGGER.info(
        "Quantizing %d/%d Conv/ConvTranspose layers (excluding %d: normalizing "
        "flow + final output layer)",
        len(all_conv) - len(exclude_nodes), len(all_conv), len(exclude_nodes),
    )
    onnx.save(onnx_model, str(fp32_path))

    calib_reader = JsonlCalibReader(
        Path(args.calibration_dataset), args.calibration_sentences,
    )

    _LOGGER.info("Running static PTQ (per-channel weights, QOperator format)...")
    quantize_static(
        str(fp32_path),
        str(args.output),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv", "ConvTranspose"],
        nodes_to_exclude=exclude_nodes,
        per_channel=True,
    )
    fp32_path.unlink()

    _LOGGER.info("Exported INT8 model to %s", args.output)


if __name__ == "__main__":
    main()
