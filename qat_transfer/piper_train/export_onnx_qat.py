#!/usr/bin/env python3
"""Export a QAT fine-tuned checkpoint to an INT8 QDQ ONNX graph for ONNX
Runtime.

Loads the checkpoint, re-applies the same weight_norm removal + QAT
wrapping used during fine-tuning (so the wrapped modules' state_dict keys
line up), freezes the FakeQuantize observers (required -- otherwise the
observer's running-stat update ops get traced into the graph and
torch.onnx chokes on them), and exports. The exported graph is then passed
through onnx_type_fix.fix_index_type_mismatches(), which is not just an
optimization here: this torch/onnx version's TorchScript-based exporter
emits a handful of Slice/Gather/etc. nodes with mismatched int32/int64
index types on this model (reproduces even on a plain FP32 export,
unrelated to quantization) that ONNX Runtime refuses to load otherwise.
(An earlier version of this script used onnxsim for this instead -- don't
go back to that: its constant-folding pass was found to also collapse
SynthesizerTrn.infer()'s genuinely input-length-dependent duration
computation down to a fixed length baked in from whatever shape it
happened to trace with, verified against the pure-PyTorch ground truth
across inputs of varying phoneme length. onnx_type_fix only touches the
specific mismatched tensors and leaves dynamic-shape behavior alone.)

Usage:
    python3 -m piper_train.export_onnx_qat \
        qat_transfer/data_preprocessed/qat_runs/lightning_logs/version_0/checkpoints/qat-best-*.ckpt \
        qat_transfer/voice.int8.onnx
"""
import argparse
import logging
import pathlib
import warnings
from pathlib import Path
from typing import Optional

import onnx
import torch

from .onnx_type_fix import fix_index_type_mismatches
from .vits.lightning import VitsModel
from .vits.quantize import (
    count_qat_layers,
    prepare_qat,
    remove_all_weight_norm,
    set_fake_quant_enabled,
    set_observer_enabled,
)

_LOGGER = logging.getLogger("piper_train.export_onnx_qat")
OPSET_VERSION = 15

torch.serialization.add_safe_globals([pathlib.PosixPath])


def main() -> None:
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to a QAT-fine-tuned checkpoint (.ckpt)")
    parser.add_argument("output", help="Path to output model (.onnx)")
    parser.add_argument(
        "--skip-type-fix",
        action="store_true",
        help="Skip the int32/int64 index-type fix (the raw export currently fails "
        "to load in ONNX Runtime on this torch version -- see module docstring)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("Loading QAT checkpoint: %s", args.checkpoint)
    model = VitsModel.load_from_checkpoint(
        args.checkpoint, dataset=None, strict=False, map_location="cpu",
        weights_only=False,
    )
    model_g = model.model_g

    _LOGGER.info("Re-applying QAT wrapping to match the fine-tuned state_dict...")
    remove_all_weight_norm(model_g)
    prepare_qat(model_g)
    _LOGGER.info("QAT layers: %d", count_qat_layers(model_g))

    # Reload the checkpoint's state_dict now that the module tree has the
    # wrapped shapes/keys (load_from_checkpoint above already did this once,
    # but that was strict=False against the *unwrapped* tree, so the
    # weight_fake_quant/act_fake_quant buffers -- and calibrated scale/
    # zero_point -- were silently skipped the first time).
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = {
        k[len("model_g."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model_g.")
    }
    missing, unexpected = model_g.load_state_dict(state_dict, strict=False)
    if missing:
        _LOGGER.warning("Missing keys on reload: %s", missing[:10])
    if unexpected:
        _LOGGER.warning("Unexpected keys on reload: %s", unexpected[:10])

    model_g.eval()
    set_observer_enabled(model_g, False)
    set_fake_quant_enabled(model_g, True)

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio, *_ = model_g.infer(
            text, text_lengths, noise_scale=noise_scale, length_scale=length_scale,
            noise_scale_w=noise_scale_w, sid=sid,
        )
        return audio.unsqueeze(1) if audio.dim() == 2 else audio

    model_g.forward = infer_forward

    dummy_input_length = 50
    sequences = torch.randint(low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long)
    sequence_lengths = torch.LongTensor([sequences.size(1)])
    sid: Optional[torch.LongTensor] = torch.LongTensor([0]) if num_speakers > 1 else None
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = (sequences, sequence_lengths, scales, sid)

    raw_output = args.output.with_suffix(".raw.onnx") if not args.skip_type_fix else args.output

    _LOGGER.info("Exporting to %s ...", raw_output)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model=model_g,
            args=dummy_input,
            f=str(raw_output),
            verbose=False,
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

    if args.skip_type_fix:
        _LOGGER.info("Exported (unfixed) model to %s", args.output)
        return

    _LOGGER.info("Fixing int32/int64 index-type mismatches...")
    m = onnx.load(str(raw_output))
    n_fixed = fix_index_type_mismatches(m)
    onnx.save(m, str(args.output))
    raw_output.unlink()
    _LOGGER.info("Fixed %d node(s). Exported INT8 QDQ model to %s", n_fixed, args.output)


if __name__ == "__main__":
    main()
