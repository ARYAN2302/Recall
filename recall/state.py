"""
recall.state — LoRA state dict helpers.

Snapshot / restore / reset for LoRA adapters. These are the primitives
the trainer and AVR loop build on.

Ported directly from tiny-cl/avr/framework.py + Living-Model v34 — the
shape is identical, just packaged cleanly.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict


def get_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot all LoRA params to CPU. Constant memory: same size as adapter.

    This is the 'neocortex snapshot' — the last known-good state we'll
    interpolate back toward if AVR detects drift.
    """
    return {n: p.data.cpu().clone()
            for n, p in model.named_parameters() if "lora_" in n}


def set_lora_state(model: nn.Module, state: Dict[str, torch.Tensor],
                   device: str = "cuda") -> None:
    """Restore LoRA params from a snapshot."""
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(device).to(p.data.dtype))


def reset_lora_to_peft_init(model: nn.Module) -> None:
    """Reset LoRA to fresh PEFT initialization: lora_A = Kaiming, lora_B = zeros.

    This is the 'hippocampus reset' — at the start of each new correction,
    we wipe the working adapter back to its initial state so it can learn
    the new correction in isolation without being biased by prior knowledge.

    NOTE: zeroing both A and B is wrong — it destroys lora_A's random init
    and the LoRA update B@A is forever zero. We must keep A's Kaiming init.
    """
    import torch.nn.init as init
    for n, p in model.named_parameters():
        if "lora_A" in n:
            init.kaiming_uniform_(p.data, a=math.sqrt(5))
        elif "lora_B" in n:
            p.data.zero_()


def count_lora_params(model: nn.Module) -> int:
    """Count trainable LoRA params — for logging."""
    return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)


def lora_param_names(model: nn.Module) -> list:
    """Return the list of LoRA param names — for sanity-checking snapshots."""
    return [n for n, _ in model.named_parameters() if "lora_" in n]
