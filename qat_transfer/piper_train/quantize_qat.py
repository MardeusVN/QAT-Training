#!/usr/bin/env python3
"""Quantization-aware fine-tuning: resumes from a converged FP32 checkpoint,
wraps the inference-path Conv1d/ConvTranspose1d/Linear layers of model_g
(enc_p, dp, flow, dec, f0_predictor) with fake-quantize wrappers (see
vits/quantize.py), and continues training at a low learning rate so the
weights adapt to INT8 quantization noise before ONNX export.

Usage:
    python3 -m piper_train.quantize_qat \
        --resume-from-checkpoint qat_transfer/best-epoch=1079-val_loss_mel=19.2943.ckpt \
        --dataset-dir data_preprocessed \
        --max_epochs 20 \
        --batch-size 6 \
        --learning-rate 2e-5
"""
import argparse
import logging
import pathlib
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .vits.dataset import Batch
from .vits.lightning import VitsModel
from .vits.quantize import (
    count_qat_layers,
    prepare_qat,
    remove_all_weight_norm,
    set_observer_enabled,
)

_LOGGER = logging.getLogger("piper_train.quantize_qat")

torch.serialization.add_safe_globals([pathlib.PosixPath])


class QATVitsModel(VitsModel):
    """VitsModel with one QAT-specific change to training_step: an initial
    generator-only warmup, where the discriminator's weights stay frozen
    (not updated) for the first `d_warmup_steps` steps while the generator
    still trains (including its adversarial loss term against D's current,
    frozen weights). Recommended practice for fine-tuning quantized GANs --
    without it, D immediately starts reacting to the generator's fresh
    fake-quant noise from step 1, which can destabilize the adversarial
    balance before the generator has had any chance to adapt."""

    def __init__(self, *args, d_warmup_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_warmup_steps = d_warmup_steps

    def training_step(self, batch: Batch, batch_idx: int):
        opt_g, opt_d = self.optimizers()

        loss_gen_all, _loss_mel = self.training_step_g(batch)
        opt_g.zero_grad()
        self.manual_backward(loss_gen_all)
        if self.hparams.grad_clip is not None:
            self.clip_gradients(opt_g, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")
        opt_g.step()

        if self.global_step >= self.d_warmup_steps:
            loss_disc_all = self.training_step_d(batch)
            opt_d.zero_grad()
            self.manual_backward(loss_disc_all)
            if self.hparams.grad_clip is not None:
                self.clip_gradients(opt_d, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")
            opt_d.step()

        if self.hparams.use_vits2 and self.hparams.mas_noise_scale_decay > 0:
            self.model_g.mas_noise_scale = max(
                0.01 - self.global_step * self.hparams.mas_noise_scale_decay, 0.0
            )


class ResetLRCallback(Callback):
    """When resuming via ckpt_path=, Lightning restores each optimizer's
    full state_dict -- including param_groups[*]['lr'] -- *after*
    configure_optimizers() runs, silently overwriting whatever LR the
    freshly-constructed optimizer started with. That's normally the whole
    point of ckpt_path resume, but if the original run's LR schedule had
    already decayed to near-zero by the time it stopped (e.g. a short
    schedule that bottomed out), literally continuing from that LR barely
    moves the weights for however many more epochs you add. This callback
    force-overwrites the LR back to a useful value once, right after
    Lightning's restore has happened, so the extended run gets a real
    second annealing phase instead of inheriting an exhausted one."""

    def __init__(self, new_lr: float):
        self.new_lr = new_lr
        self._done = False

    def on_train_start(self, trainer, pl_module):
        if self._done:
            return
        for opt in trainer.optimizers:
            old_lr = opt.param_groups[0]["lr"]
            for group in opt.param_groups:
                group["lr"] = self.new_lr
            _LOGGER.info("Reset optimizer LR: %.2e -> %.2e", old_lr, self.new_lr)
        self._done = True


class FreezeObserverCallback(Callback):
    """Standard QAT practice: let the FakeQuantize observers calibrate
    scale/zero_point against real activation statistics for the first N
    epochs, then freeze them so the exported values stop drifting while
    the remaining epochs just let the weights settle around the now-fixed
    quantization grid."""

    def __init__(self, freeze_after_epoch: int):
        self.freeze_after_epoch = freeze_after_epoch
        self._frozen = False

    def on_train_epoch_start(self, trainer, pl_module):
        if (not self._frozen) and trainer.current_epoch >= self.freeze_after_epoch:
            _LOGGER.info(
                "Freezing FakeQuantize observers at epoch %s", trainer.current_epoch
            )
            set_observer_enabled(pl_module.model_g, False)
            self._frozen = True


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Path to a converged FP32 .ckpt to start QAT fine-tuning from (fresh "
        "start: weights only, optimizer/scheduler/epoch state reset)",
    )
    parser.add_argument(
        "--resume-qat-checkpoint",
        help="Path to an existing QAT run's checkpoint (already wrapped, e.g. "
        "qat-last-epoch=N.ckpt) to CONTINUE training from -- restores optimizer "
        "momentum, LR scheduler state, and the epoch/global_step counters, not "
        "just weights. Use this instead of --resume-from-checkpoint to extend "
        "an existing run's step budget without losing progress. Mutually "
        "exclusive with --resume-from-checkpoint.",
    )
    parser.add_argument(
        "--reset-lr-on-resume",
        type=float,
        default=None,
        help="Only meaningful with --resume-qat-checkpoint: force the optimizer LR "
        "to this value right after the checkpoint's optimizer state (including its "
        "own decayed LR) is restored, instead of continuing from wherever the "
        "original run's schedule left off. Use this when extending a run whose LR "
        "had already decayed close to zero by the time it stopped -- continuing "
        "from an exhausted LR barely moves the weights no matter how many more "
        "epochs you add.",
    )
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="QAT fine-tuning LR (default: ~40x lower than the 2e-4 base-training "
        "default -- GAN adversarial training is sensitive to LR, and published QAT "
        "recipes for other architectures land in the 1e-6-1e-5 range, not the "
        "'divide by 10' heuristic used for plain supervised fine-tuning)",
    )
    parser.add_argument(
        "--lr-decay-target",
        type=float,
        default=0.1,
        help="Total multiplicative LR decay to reach by the end of training (default: "
        "10x reduction). The checkpoint's own lr_decay (tuned for ~1000-epoch base "
        "training, gamma=0.999875) barely moves the LR over a short QAT run, so this "
        "computes a per-epoch gamma = lr_decay_target ** (1/max_epochs) instead.",
    )
    parser.add_argument(
        "--d-warmup-steps",
        type=int,
        default=None,
        help="Keep the discriminator's weights frozen for this many initial steps "
        "while the generator still trains (default: one epoch's worth of steps). "
        "Recommended when fine-tuning a quantized GAN: without it, D immediately "
        "reacts to the generator's fresh fake-quant noise from step 1, which can "
        "destabilize the adversarial balance before the generator adapts at all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Override the checkpoint's batch size (QAT's extra fake-quant ops add "
        "VRAM overhead; lower this if you OOM)",
    )
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Override the checkpoint's DataLoader worker count. Windows spawns "
        "(rather than forks) worker processes, and re-importing torch's CUDA "
        "libraries concurrently in many of them is unreliable there -- the "
        "checkpoint's original num_workers (tuned for Linux) is not a safe "
        "default on this platform.",
    )
    parser.add_argument(
        "--freeze-observer-epoch",
        type=int,
        default=None,
        help="Freeze FakeQuantize observers after this epoch (default: 40%% of "
        "max_epochs -- published QAT recipes freeze around the 30-40%% mark, "
        "earlier than this script's original 60%% default)",
    )
    parser.add_argument(
        "--qat-submodules",
        default="enc_p,dp,dec,f0_predictor",
        help="Comma-separated model_g submodule names to QAT-wrap (default matches "
        "export_int8.py's deployed quantization scope for the epoch1079-lineage "
        "architecture: everything except flow). Different checkpoint lineages can "
        "have a different quantization-sensitivity profile -- override this to match "
        "whichever layers actually get quantized for that architecture, found via a "
        "PTQ exclusion sweep (see export_int8.py's module docstring for the method).",
    )
    parser.add_argument(
        "--qat-exclude-leaf-names",
        default="conv_post",
        help="Comma-separated leaf module names to skip within the wrapped "
        "submodules (default: the final output Conv1d before tanh).",
    )
    parser.add_argument("--checkpoint-epochs", type=int, default=1)
    parser.add_argument("--default_root_dir", help="Trainer log/checkpoint directory")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Cap each epoch at this many training batches (Lightning's "
        "limit_train_batches) instead of the full dataset, so checkpoints/"
        "validation/observer-freeze land on a predictable step cadence "
        "regardless of dataset size. Also usable to smoke-test the pipeline "
        "with a small value.",
    )
    args = parser.parse_args()
    _LOGGER.debug(args)

    if bool(args.resume_from_checkpoint) == bool(args.resume_qat_checkpoint):
        raise SystemExit(
            "Pass exactly one of --resume-from-checkpoint (fresh QAT start from an "
            "FP32 checkpoint) or --resume-qat-checkpoint (continue an existing QAT run)"
        )
    source_checkpoint = args.resume_from_checkpoint or args.resume_qat_checkpoint
    is_resume = bool(args.resume_qat_checkpoint)

    dataset_dir = Path(args.dataset_dir)
    dataset_path = dataset_dir / "dataset.jsonl"
    if not args.default_root_dir:
        args.default_root_dir = str(dataset_dir / "qat_runs")

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    # Per-epoch ExponentialLR gamma that reaches lr_decay_target by the final
    # epoch, instead of the checkpoint's own gamma=0.999875 (tuned for ~1000
    # epochs of base training, which barely decays at all across a short QAT run).
    lr_decay = args.lr_decay_target ** (1.0 / args.max_epochs)
    _LOGGER.info(
        "LR schedule: %.2e -> %.2e over %d epochs (gamma=%.5f)",
        args.learning_rate, args.learning_rate * args.lr_decay_target,
        args.max_epochs, lr_decay,
    )

    d_warmup_steps = args.d_warmup_steps
    if d_warmup_steps is None:
        steps_per_epoch = args.steps_per_epoch or 2000  # rough fallback if unset
        d_warmup_steps = steps_per_epoch
    _LOGGER.info("Discriminator warmup: frozen for the first %d step(s)", d_warmup_steps)

    _LOGGER.info(
        "Loading %s checkpoint: %s",
        "QAT (resuming)" if is_resume else "FP32 (fresh start)", source_checkpoint,
    )
    model = QATVitsModel.load_from_checkpoint(
        source_checkpoint,
        dataset=[dataset_path],  # override: point at this machine's real dataset path
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=lr_decay,
        num_workers=args.num_workers,
        d_warmup_steps=d_warmup_steps,
        strict=False,
        map_location="cpu",
        weights_only=False,
    )

    # QATVitsModel.__init__ always constructs a fresh SynthesizerTrn with
    # weight_norm applied (that's unconditional in the model architecture),
    # regardless of whether the checkpoint we're about to resume from had it
    # removed. When resuming, the state_dict load above under strict=False
    # can't restore dec/flow's *values* (the QAT checkpoint has plain
    # "weight" keys, not "weight_g"/"weight_v" -- structurally mismatched, so
    # those keys get silently skipped), leaving dec/flow at random init here
    # -- but that's fine, since it's only the *structure* that needs to be
    # right before wrapping; trainer.fit(..., ckpt_path=...) below does a
    # proper full-state restore (weights, optimizer, scheduler, epoch
    # counter) once the module tree matches the checkpoint's shape.
    _LOGGER.info("Preparing model_g for QAT (removing weight_norm, wrapping layers)...")
    remove_all_weight_norm(model.model_g)
    # Match export_int8.py's deployed quantization scope exactly: flow (a
    # normalizing flow -- errors compound across its 4 sequential coupling
    # layers) and dec.conv_post (the final Conv1d before tanh, setting
    # waveform sample values directly) are excluded there because quantizing
    # them hurt quality far more than their share of model size justified.
    # Wrapping them here anyway would waste training capacity adapting
    # layers that get exported as plain FP32 regardless.
    qat_submodules = tuple(s.strip() for s in args.qat_submodules.split(",") if s.strip())
    qat_exclude_leaf_names = tuple(
        s.strip() for s in args.qat_exclude_leaf_names.split(",") if s.strip()
    )
    _LOGGER.info(
        "QAT wrap scope: submodules=%s exclude_leaf_names=%s",
        qat_submodules, qat_exclude_leaf_names,
    )
    prepare_qat(
        model.model_g,
        submodule_names=qat_submodules,
        exclude_leaf_names=qat_exclude_leaf_names,
    )
    n_layers = count_qat_layers(model.model_g)
    _LOGGER.info("QAT-wrapped %d Conv1d/ConvTranspose1d/Linear layers", n_layers)

    freeze_epoch = args.freeze_observer_epoch
    if freeze_epoch is None:
        freeze_epoch = max(1, int(args.max_epochs * 0.4))

    devices = int(args.devices) if str(args.devices).isdigit() else args.devices
    callbacks = [
        ModelCheckpoint(
            filename="qat-best-{epoch}-{val_loss_mel:.4f}",
            monitor="val_loss_mel",
            mode="min",
            save_top_k=1,
            save_last=False,
        ),
        FreezeObserverCallback(freeze_epoch),
    ]
    if args.reset_lr_on_resume is not None:
        if not is_resume:
            raise SystemExit("--reset-lr-on-resume requires --resume-qat-checkpoint")
        callbacks.append(ResetLRCallback(args.reset_lr_on_resume))
    if args.checkpoint_epochs is not None:
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=args.checkpoint_epochs,
                save_top_k=1,
                save_last=True,
                filename="qat-last-{epoch}",
            )
        )

    trainer_kwargs = dict(
        accelerator=args.accelerator,
        devices=devices,
        strategy="auto",
        precision=args.precision,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        logger=TensorBoardLogger(save_dir=args.default_root_dir),
        callbacks=callbacks,
    )
    if args.steps_per_epoch is not None:
        trainer_kwargs["limit_train_batches"] = args.steps_per_epoch

    trainer = Trainer(**trainer_kwargs)
    if is_resume:
        _LOGGER.info(
            "Resuming full trainer state (weights, optimizer, LR scheduler, "
            "epoch counter) from %s, continuing to max_epochs=%s",
            source_checkpoint, args.max_epochs,
        )
        trainer.fit(model, ckpt_path=source_checkpoint)
    else:
        trainer.fit(model)


if __name__ == "__main__":
    main()
