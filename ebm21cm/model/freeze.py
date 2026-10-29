"""Exact parameter groups for the LoReLi fine-tuning ladder.

The reference model is single-level ``RhizomeOperator2d``.  Its parameters are
partitioned into:

``decoder``
    ``decode.*`` — both output channels, including the structured T_b head.
``ends``
    plus ``lift.*`` and ``scalar_embed.*`` — the input lift and scalar embedding.
``pointwise``
    plus ``cells.*.interaction.gates``, ``.value``, ``.proposal`` and any
    ``static_logits`` — every pointwise map, while the spectral kernel stays frozen.
``spectral``
    ``cells.*.interaction.kernel.*`` (and ``.disc.*``/compact kernel weights).
    Trainable only with ``freeze="none"``; always frozen in the first three.

This module is deliberately independent of the training loop so the counts and
gradient masks can be tested directly.
"""

from __future__ import annotations

import torch

FREEZE_CHOICES = ("decoder", "ends", "pointwise", "none")
CATEGORIES = ("decoder", "ends", "pointwise", "spectral")


def is_spectral(name: str) -> bool:
    return ".interaction.kernel." in name or ".interaction.disc." in name


def category_of(name: str) -> str:
    """One of decoder / ends / pointwise / spectral; raises for an unknown parameter."""
    if name.startswith("decode."):
        return "decoder"
    if name.startswith("lift.") or name.startswith("scalar_embed.") or name.startswith("lift_"):
        return "ends"
    if name.startswith("scalar_"):
        return "ends"
    if is_spectral(name):
        return "spectral"
    if (
        ".gates." in name
        or ".value." in name
        or ".proposal." in name
        or ".merge." in name
        or name.endswith("static_logits")
    ):
        return "pointwise"
    raise ValueError(f"unclassified parameter {name!r}")


def parameter_groups(model: torch.nn.Module) -> dict[str, list[str]]:
    """Every parameter name by category, checked to cover the model exactly."""
    groups = {name: [] for name in CATEGORIES}
    for name, _ in model.named_parameters():
        groups[category_of(name)].append(name)
    total = sum(len(v) for v in groups.values())
    if total != len(list(model.parameters())):
        raise RuntimeError(f"parameter partition covers {total} of {len(list(model.parameters()))}")
    return groups


def freeze_model(model: torch.nn.Module, freeze: str) -> dict:
    """Set ``requires_grad`` for one ladder setting; return counts and names.

    Counts use ``Parameter.numel()`` (complex spectral weights count as complex
    elements, matching ``sum(p.numel() for p in model.parameters())``).
    """
    if freeze not in FREEZE_CHOICES:
        raise ValueError(f"freeze must be one of {FREEZE_CHOICES}")
    groups = parameter_groups(model)
    trainable_categories = {
        "decoder": {"decoder"},
        "ends": {"decoder", "ends"},
        "pointwise": {"decoder", "ends", "pointwise"},
        "none": set(CATEGORIES),
    }[freeze]
    counts = {name: 0 for name in CATEGORIES}
    trainable_counts = {name: 0 for name in CATEGORIES}
    for category, names in groups.items():
        for name in names:
            parameter = dict(model.named_parameters())[name]
            train = category in trainable_categories
            parameter.requires_grad_(train)
            counts[category] += int(parameter.numel())
            if train:
                trainable_counts[category] += int(parameter.numel())
    return {
        "freeze": freeze,
        "counts": counts,
        "trainable_counts": trainable_counts,
        "total": sum(counts.values()),
        "trainable": sum(trainable_counts.values()),
        "groups": groups,
    }


def optimizer_parameter_groups(model: torch.nn.Module, *, scratch: bool = False) -> dict[str, list]:
    """Split trainable parameters into spectral (pretrained LR) and adaptation (new LR).

    For a fine-tune, the spectral kernels are the only pretrained weights that
    can be trainable (``freeze=none``); they use ``lr_pretrained``.  Everything
    else uses ``lr_new``.  A scratch model has no pretrained weights, so all
    parameters use ``lr_new``.
    """
    spectral, adaptation = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (adaptation if scratch or not is_spectral(name) else spectral).append(parameter)
    return {"pretrained": spectral, "new": adaptation}
