"""
recall.base — model + tokenizer + LoRA setup.

Loads a frozen HF base model, attaches a LoRA adapter via PEFT, and
returns the wrapped model + tokenizer. This is the only file that
knows about HF/PEFT specifics.

The LoRA targets, rank, alpha, dropout come from RecallConfig.
"""
from __future__ import annotations
import math
import torch
from typing import Tuple

from .config import RecallConfig


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        # T4 doesn't support bf16 natively; fall back to fp16
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def load_model_and_tokenizer(
    config: RecallConfig,
    model_id_override: str = None,
) -> Tuple[torch.nn.Module, "PreTrainedTokenizer"]:
    """Load the frozen base + attach LoRA. Returns (model, tokenizer).

    The model is wrapped by PEFT — only LoRA params are trainable.
    Base weights are frozen and never updated.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype)
    model_id = model_id_override or config.model_id

    print(f"  [base] Loading {model_id} ({dtype}, {device})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        attn_implementation="eager",  # matches the validated setup
    )

    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def get_device(config: RecallConfig) -> str:
    return _resolve_device(config.device)
