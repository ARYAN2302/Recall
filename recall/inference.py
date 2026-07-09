"""
recall.inference — generation helpers.

Thin wrapper around HF generate(). The only trick: atomically swap
the latest committed LoRA state into the model before generating,
so callers never see a half-trained adapter.
"""
from __future__ import annotations
import torch
from typing import Optional

from .state import set_lora_state


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    device: str = "cuda",
    neo_state: Optional[dict] = None,
) -> str:
    """Generate a completion. Optionally apply a neo_state snapshot first.

    Args:
        neo_state: if provided, swap this LoRA state in before generating.
            Useful for evaluation against a specific snapshot version.
    """
    if neo_state is not None:
        set_lora_state(model, neo_state, device)

    model.eval()
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return completion
