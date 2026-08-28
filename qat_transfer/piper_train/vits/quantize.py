"""Quantization-aware training (QAT) support for the VITS/BigVGAN generator.

Targets ONNX Runtime's QDQ (QuantizeLinear/DequantizeLinear) INT8 format:
activations are uint8 per-tensor affine, weights are int8 per-channel
symmetric -- the standard u8s8 scheme ONNX Runtime's CPU EP expects.

Design
------
`torch.ao.quantization.prepare_qat` (eager mode) only ships QAT modules for
Conv2d/Conv3d/Linear -- there is no `nnqat.Conv1d`/`nnqat.ConvTranspose1d`,
so it can't touch this model at all (everything here is 1D). FX-mode
quantization (`prepare_qat_fx`) does support Conv1d/ConvTranspose1d, but
requires successfully `torch.fx.symbolic_trace`-ing each submodule's
forward, which breaks on this codebase's data-dependent branching
(`if g is not None`, `if reverse: ... else: ...` used with different
control flow at different call sites -- e.g. StochasticDurationPredictor
runs its NLL branch *and* its reverse-sampling branch within the same
training step, which can't both live in one traced graph).

So instead of swapping whole submodules via tracing, this module walks the
live module tree and replaces each leaf Conv1d/ConvTranspose1d/Linear
in-place with a thin wrapper that fake-quantizes its input activation and
its weight on every forward call, then runs the original functional op.
Being a drop-in replacement with the exact same call signature, it works
transparently under arbitrary surrounding Python control flow (multiple
call sites, forward/reverse branches, etc.) -- no tracing involved.

The wrapper's `weight_fake_quant`/`act_fake_quant` are genuine
`torch.ao.quantization.FakeQuantize` instances, so they emit the same
`aten::fake_quantize_per_{tensor,channel}_affine` ops the built-in QAT
modules do; `torch.onnx.export` converts those to QuantizeLinear/
DequantizeLinear pairs identically regardless of which Python module
produced them.
"""
import logging
from typing import Iterable

import torch
from torch import nn
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.nn import functional as F

_LOGGER = logging.getLogger("vits.quantize")

# Submodules of SynthesizerTrn that are part of infer() -- these get
# quantized. enc_q (posterior encoder) and the discriminators are training
# -only and are never touched.
INFERENCE_SUBMODULE_NAMES = ("enc_p", "dp", "flow", "dec", "f0_predictor")


def _make_weight_fake_quant(ch_axis: int) -> FakeQuantize:
    # ONNX's QuantizeLinear/DequantizeLinear only accept (quant_min, quant_max)
    # of (0, 127), (0, 255) or (-128, 127) -- torch.onnx.export rejects the
    # (-127, 127) range PyTorch's own default qat qconfigs typically use.
    # Harmless for per_channel_symmetric (zero_point stays 0 either way).
    return FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=ch_axis,
    )()


def _make_act_fake_quant() -> FakeQuantize:
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )()


class QATConv1d(nn.Module):
    """Drop-in replacement for nn.Conv1d that fake-quantizes its input
    activation and weight (per-channel, ch_axis=0 -- out_channels is dim 0
    of a Conv1d weight) on every forward call."""

    def __init__(self, orig: nn.Conv1d):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.stride = orig.stride
        self.padding = orig.padding
        self.dilation = orig.dilation
        self.groups = orig.groups
        self.weight_fake_quant = _make_weight_fake_quant(ch_axis=0)
        self.act_fake_quant = _make_act_fake_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fake_quant(x)
        w = self.weight_fake_quant(self.weight)
        return F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QATConvTranspose1d(nn.Module):
    """Drop-in replacement for nn.ConvTranspose1d. ch_axis=1 here --
    out_channels is dim 1 of a ConvTranspose1d weight ([in, out, k])."""

    def __init__(self, orig: nn.ConvTranspose1d):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.stride = orig.stride
        self.padding = orig.padding
        self.output_padding = orig.output_padding
        self.dilation = orig.dilation
        self.groups = orig.groups
        self.weight_fake_quant = _make_weight_fake_quant(ch_axis=1)
        self.act_fake_quant = _make_act_fake_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fake_quant(x)
        w = self.weight_fake_quant(self.weight)
        return F.conv_transpose1d(
            x, w, self.bias, self.stride, self.padding, self.output_padding,
            self.groups, self.dilation,
        )


class QATLinear(nn.Module):
    """Drop-in replacement for nn.Linear (ch_axis=0 -- out_features)."""

    def __init__(self, orig: nn.Linear):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.weight_fake_quant = _make_weight_fake_quant(ch_axis=0)
        self.act_fake_quant = _make_act_fake_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fake_quant(x)
        w = self.weight_fake_quant(self.weight)
        return F.linear(x, w, self.bias)


_WRAPPABLE = {
    nn.Conv1d: QATConv1d,
    nn.ConvTranspose1d: QATConvTranspose1d,
    nn.Linear: QATLinear,
}
_QAT_WRAPPER_TYPES = tuple(_WRAPPABLE.values())


def remove_all_weight_norm(model_g) -> None:
    """Fuse every weight_norm-parametrized Conv/ConvTranspose in model_g
    (dec's upsampling stack and every WN block inside the normalizing
    flow) into a plain weight Parameter. Required before wrapping: eager
    weight_norm recomputes and *reassigns* `.weight` via a forward
    pre-hook on every call, so a naive wrapper that captures `orig.weight`
    once would end up holding a stale/disconnected tensor instead of the
    live weight_g/weight_v-derived one."""
    with torch.no_grad():
        model_g.dec.remove_weight_norm()
        for flow in model_g.flow.flows:
            if hasattr(flow, "enc"):  # ResidualCouplingLayer / TransformerCouplingLayer
                flow.enc.remove_weight_norm()


def _wrap_leaves(module: nn.Module, exclude_names: frozenset = frozenset()) -> int:
    n_wrapped = 0
    for name, child in list(module.named_children()):
        if name in exclude_names:
            continue  # leaf name explicitly excluded (e.g. dec.conv_post)
        if isinstance(child, _QAT_WRAPPER_TYPES):
            continue  # already wrapped (idempotent re-entry)
        wrapper_cls = _WRAPPABLE.get(type(child))
        if wrapper_cls is not None:
            setattr(module, name, wrapper_cls(child))
            n_wrapped += 1
        else:
            n_wrapped += _wrap_leaves(child, exclude_names)
    return n_wrapped


def prepare_qat(
    model_g,
    submodule_names: Iterable[str] = INFERENCE_SUBMODULE_NAMES,
    exclude_leaf_names: Iterable[str] = (),
):
    """Wrap every Conv1d/ConvTranspose1d/Linear leaf inside the named
    inference-path submodules of model_g (a SynthesizerTrn) with a
    fake-quantizing replacement, in place. Call remove_all_weight_norm()
    first. Returns model_g.

    exclude_leaf_names: leaf attribute names to skip wrapping regardless of
    which submodule they're found in (matched against the last path
    component only, e.g. "conv_post" skips dec.conv_post). Use this to keep
    training's fake-quant wrapping consistent with which layers actually
    end up quantized at export time -- e.g. export_int8.py leaves out
    model_g.flow and dec.conv_post (see its docstring for why), so QAT
    fine-tuning should skip wrapping them too rather than spending capacity
    adapting layers that will be exported as plain FP32 anyway.
    """
    exclude = frozenset(exclude_leaf_names)
    total = 0
    for name in submodule_names:
        submodule = getattr(model_g, name, None)
        if submodule is None:
            continue
        n = _wrap_leaves(submodule, exclude)
        _LOGGER.info("Wrapped %d layer(s) in model_g.%s for QAT", n, name)
        total += n
    _LOGGER.info("Total quantized layers: %d", total)
    return model_g


def set_fake_quant_enabled(model_g, enabled: bool) -> None:
    for m in model_g.modules():
        if isinstance(m, _QAT_WRAPPER_TYPES):
            m.weight_fake_quant.enable_fake_quant(enabled)
            m.act_fake_quant.enable_fake_quant(enabled)


def set_observer_enabled(model_g, enabled: bool) -> None:
    """Freeze/unfreeze the running min/max statistics. Standard QAT practice:
    disable observer updates for the last chunk of fine-tuning so exported
    scale/zero_point stop drifting right before convert/export."""
    for m in model_g.modules():
        if isinstance(m, _QAT_WRAPPER_TYPES):
            m.weight_fake_quant.enable_observer(enabled)
            m.act_fake_quant.enable_observer(enabled)


def count_qat_layers(model_g) -> int:
    return sum(1 for m in model_g.modules() if isinstance(m, _QAT_WRAPPER_TYPES))
