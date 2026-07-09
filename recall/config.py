"""
recall.config — central config dataclass.

One place to tune every hyperparameter. Defaults match the validated
setup from the Living-Model experiments on Qwen3-0.6B (seed 42, positive
BWT on TRACE).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RecallConfig:
    """All hyperparameters for a Recall instance.

    Defaults are calibrated for Qwen3-0.6B + LoRA r=32 on a single T4/A10G.
    """

    # ── Model ──
    model_id: str = "Qwen/Qwen3-0.6B"
    """HF model id. Frozen base — Recall never updates base weights."""

    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "v_proj")
    """LoRA target modules. q_proj + v_proj is the memory-light default
    that worked on Qwen3-0.6B in the Living-Model experiments."""

    dtype: str = "bfloat16"
    """torch dtype for the base model. bf16 is the sweet spot for T4+."""

    # ── Training (LEARN phase: hippocampus SFT) ──
    train_lr: float = 2e-4
    train_weight_decay: float = 0.01
    train_max_grad_norm: float = 1.0
    train_epochs: int = 3
    """Epochs per correction on the hippocampus. 3 is calibrated."""

    # ── Consolidation (hippocampus → neocortex distillation) ──
    consolidation_lr: float = 1e-4
    """Half the hippocampus LR — slow integration is the whole point."""
    consolidation_epochs: int = 1

    # ── Shared training shapes ──
    batch_size: int = 4
    """Reduced from 8 because Qwen's vocab is large; prevents OOM on T4."""
    context_length: int = 512

    data_repeat: int = 1
    """How many times to duplicate each (input, target) pair in the training
    set. With a single correction, this is the main lever for getting enough
    gradient steps. Set to 20-30 for single-correction training."""

    # ── AVR: VERIFY ──
    drift_threshold: float = 1.15
    """Fire repair if PPL_now / PPL_best > 1.15. v23 default."""
    avr_probe_samples: int = 50
    """How many prior-correction samples to probe for PPL drift."""

    # ── AVR: REPAIR ──
    repair_alpha: float = 0.1
    """Interpolation strength: θ ← (1-α)·θ + α·θ_snapshot."""
    max_repair_steps: int = 10
    """Cap on the verify-repair loop. v23 shipped with 10."""

    avr_every_n: int = 5
    """Run AVR after every N corrections. 5 balances cost vs drift."""

    # ── Inference ──
    max_new_tokens: int = 64
    """Default generation length."""

    # ── Runtime ──
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    data_dir: Optional[str] = None
    """Where to persist snapshots + queue. None = temp dir."""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lora_targets"] = list(self.lora_targets)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RecallConfig":
        d = dict(d)
        if "lora_targets" in d and not isinstance(d["lora_targets"], tuple):
            d["lora_targets"] = tuple(d["lora_targets"])
        return cls(**d)


# A smaller config for quick tests / smoke runs
SMOKE = RecallConfig(
    model_id="Qwen/Qwen3-0.6B",
    lora_rank=8,
    lora_targets=("q_proj",),
    train_epochs=1,
    consolidation_epochs=1,
    batch_size=2,
    context_length=256,
    avr_probe_samples=10,
    avr_every_n=3,
    max_new_tokens=32,
)
