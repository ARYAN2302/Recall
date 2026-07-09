"""
recall.trainer — the LEARN + CONSOLIDATE phases.

This is the heart of Recall: the two-stream hippocampus-neocortex
training loop, ported from Living-Model v34 (the experiment that
achieved positive backward transfer on TRACE).

Per correction:
    1. Snapshot neocortex state S_t
    2. Reset hippocampus to fresh PEFT init (Kaiming A, zeros B)
    3. Train hippocampus on the correction (isolated SFT, N epochs)
    4. Consolidate: KL-distill hippocampus → neocortex (M epochs, slow LR)
    5. Discard hippocampus (it's just a state dict on CPU)
    6. Return updated neocortex state

Why two streams?
- Hippocampus learns fast (high LR, fresh init) — it absorbs the new
  correction without fighting prior knowledge.
- Neocortex learns slow (half LR, persistent) — it integrates the new
  knowledge via distillation without catastrophic forgetting.
- AVR (in avr.py) catches any residual drift on prior corrections.

This is the same complementary-learning-systems split biology uses.
"""
from __future__ import annotations
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

from .config import RecallConfig
from .state import get_lora_state, set_lora_state, reset_lora_to_peft_init


# ────────────────────────────────────────────────────────────────────
# Dataset — proper SFT, one example per (prompt, answer) pair
# ────────────────────────────────────────────────────────────────────

class _SFTDataset(Dataset):
    """SFT dataset: one (prompt, answer) pair = one padded training example.

    Unlike the old chunked-stream approach, this gives us N examples per
    pair (via data_repeat), each padded to context_length. Labels mask
    padding with -100 so the loss is only computed on real tokens.

    The prompt is NOT masked — training on the full sequence is fine for
    a base model. The gradient is dominated by the answer tokens (high
    loss → large gradient), so the model naturally focuses on learning
    to produce the answer given the prompt.
    """

    def __init__(self, tokenizer, pairs: List[Tuple[str, str]],
                 context_length: int, data_repeat: int = 1):
        self.pad_id = tokenizer.pad_token_id
        self.examples = []
        for _ in range(max(1, data_repeat)):
            for prompt, answer in pairs:
                text = prompt + " " + answer + tokenizer.eos_token
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids = ids[:context_length]
                # Build input_ids (padded) and labels (padding masked with -100)
                pad_len = context_length - len(ids)
                input_ids = ids + [self.pad_id] * pad_len
                labels = ids + [-100] * pad_len
                # attention_mask: 1 for real tokens, 0 for padding
                attention_mask = [1] * len(ids) + [0] * pad_len
                self.examples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


# ────────────────────────────────────────────────────────────────────
# Phase 1: hippocampus SFT
# ────────────────────────────────────────────────────────────────────

def train_hippocampus(
    model: nn.Module,
    tokenizer,
    pairs: List[Tuple[str, str]],
    config: RecallConfig,
    device: str,
    verbose: bool = True,
) -> Dict:
    """Train the hippocampus (the working LoRA) on a single correction.

    The model's current LoRA state should already be reset to PEFT init
    before calling this. After this returns, the LoRA holds the
    hippocampus's learned state for this correction.

    Returns:
        {"steps": int, "avg_loss": float, "elapsed_s": float}
    """
    dataset = _SFTDataset(
        tokenizer, pairs, config.context_length, config.data_repeat)
    n_examples = len(dataset)
    if verbose:
        print(f"    [hippo] {n_examples} examples (data_repeat={config.data_repeat}), "
              f"context_length={config.context_length}", flush=True)

    # Only LoRA params are trainable — base is frozen by PEFT
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(
        trainable, lr=config.train_lr, weight_decay=config.train_weight_decay)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    t0 = time.time()
    gs, tl = 0, 0.0
    for epoch in range(config.train_epochs):
        for batch in loader:
            model.train()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.train_max_grad_norm)
            opt.step()
            tl += out.loss.item()
            gs += 1
            if verbose and gs % 20 == 0:
                elapsed = time.time() - t0
                print(f"      [hippo] step {gs} | loss={tl/gs:.4f} | "
                      f"{elapsed:.0f}s", flush=True)
    if verbose and gs > 0:
        print(f"    [hippo] done: {gs} steps, avg_loss={tl/max(gs,1):.4f}, "
              f"{time.time()-t0:.0f}s", flush=True)

    return {
        "steps": gs,
        "avg_loss": tl / max(gs, 1),
        "elapsed_s": time.time() - t0,
    }


# ────────────────────────────────────────────────────────────────────
# Phase 2: hippocampus → neocortex consolidation (KL distillation)
# ────────────────────────────────────────────────────────────────────

def consolidate_to_neocortex(
    model: nn.Module,
    tokenizer,
    hippo_state: Dict[str, torch.Tensor],
    neo_state: Dict[str, torch.Tensor],
    pairs: List[Tuple[str, str]],
    config: RecallConfig,
    device: str,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Distill hippocampus → neocortex via KL divergence.

    For each batch:
        1. Load hippocampus state, get logits (no grad)
        2. Load neocortex state, get logits (with grad)
        3. Loss = KL(p_hippo || p_neo) on non-padded positions only
        4. Gradient step on neocortex

    Why KL not MSE? KL preserves the distribution shape — the neocortex
    learns the hippocampus's *distribution over tokens*, not its argmax.
    This means it generalizes better to prompts the hippocampus didn't
    see, which is what we want for an agent that gets varied inputs.

    Returns:
        Updated neocortex state dict.
    """
    dataset = _SFTDataset(
        tokenizer, pairs, config.context_length, config.data_repeat)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    if verbose:
        print(f"    [consolid] distilling hippo → neo "
              f"({config.consolidation_epochs} epoch, {len(dataset)} examples)",
              flush=True)

    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(
        trainable,
        lr=config.consolidation_lr,
        weight_decay=config.train_weight_decay,
    )

    t0 = time.time()
    gs, tl = 0, 0.0
    for epoch in range(config.consolidation_epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 1. Hippocampus logits (frozen — no grad)
            set_lora_state(model, hippo_state, device)
            model.eval()
            with torch.no_grad():
                hippo_out = model(input_ids=input_ids,
                                  attention_mask=attention_mask)
                hippo_logits = hippo_out.logits

            # 2. Neocortex logits (with grad)
            set_lora_state(model, neo_state, device)
            model.train()
            neo_out = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            neo_logits = neo_out.logits

            # 3. KL(p_hippo || p_neo) on non-padded positions only
            # Shift for next-token prediction: predict token t+1 from token t
            shift_hippo = hippo_logits[..., :-1, :].contiguous()
            shift_neo = neo_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()  # (batch, seq-1)

            log_p_neo = F.log_softmax(shift_neo.float(), dim=-1)
            p_hippo = F.softmax(shift_hippo.float(), dim=-1)

            # Per-position KL, then mask out padding
            kl_per_pos = F.kl_div(
                log_p_neo, p_hippo, reduction='none').sum(dim=-1)  # (batch, seq-1)
            # Mask: 1 where the label is not -100 (real token), 0 where padding
            mask = (shift_labels != -100).float()
            kl_loss = (kl_per_pos * mask).sum() / mask.sum().clamp(min=1)

            opt.zero_grad()
            kl_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, config.train_max_grad_norm)
            opt.step()

            tl += kl_loss.item()
            gs += 1
            if verbose and gs % 20 == 0:
                print(f"      [consolid] step {gs} | KL={tl/gs:.4f}",
                      flush=True)

            # Track the evolving neocortex state
            neo_state = get_lora_state(model)

    if verbose:
        print(f"    [consolid] done: {gs} steps, avg_KL={tl/max(gs,1):.4f}, "
              f"{time.time()-t0:.0f}s", flush=True)

    return neo_state


# ────────────────────────────────────────────────────────────────────
# Full per-correction pipeline
# ────────────────────────────────────────────────────────────────────

def learn_correction(
    model: nn.Module,
    tokenizer,
    neo_state: Dict[str, torch.Tensor],
    pairs: List[Tuple[str, str]],
    config: RecallConfig,
    device: str,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Run the full LEARN pipeline for one correction.

    1. Snapshot neocortex (for AVR repair target later)
    2. Reset hippocampus to fresh PEFT init
    3. Train hippocampus on the correction
    4. Consolidate hippocampus → neocortex via KL distillation
    5. Discard hippocampus (just don't return it)

    Returns:
        (new_neo_state, info_dict)
        info_dict has: neo_snapshot (for AVR), hippo_train_info
    """
    # 1. Snapshot neocortex before this correction (AVR repair target)
    import copy
    neo_snapshot = copy.deepcopy(neo_state)
    if verbose:
        print(f"  [learn] neocortex snapshot taken", flush=True)

    # 2. Reset hippocampus to fresh PEFT init
    reset_lora_to_peft_init(model)
    if verbose:
        print(f"  [learn] hippocampus reset to fresh PEFT init", flush=True)

    # 3. Train hippocampus
    hippo_info = train_hippocampus(
        model, tokenizer, pairs, config, device, verbose)
    hippo_state = get_lora_state(model)
    if verbose:
        print(f"  [learn] hippocampus trained ({hippo_info['steps']} steps)",
              flush=True)

    # 4. Consolidate
    set_lora_state(model, neo_state, device)
    new_neo_state = consolidate_to_neocortex(
        model, tokenizer, hippo_state, neo_state, pairs, config, device,
        verbose)
    if verbose:
        print(f"  [learn] consolidation complete", flush=True)

    info = {
        "neo_snapshot": neo_snapshot,
        "hippo_train": hippo_info,
    }
    return new_neo_state, info
