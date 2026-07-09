"""
eval.baselines — comparison systems for the forgetting curve.

The viral chart compares Recall against:
    - Naive SFT: same LoRA training, no AVR, no two-stream
      (proves AVR + two-stream is the value-add)
    - Replay SFT: standard CL baseline with a small replay buffer
      (proves Recall beats the obvious approach)
    - [future] Mem0, Letta, Zep: retrieval-based memory wrappers

All baselines run on the same model (Qwen3-0.6B + LoRA r=32) and the
same corrections, so the comparison is apples-to-apples.
"""
from __future__ import annotations
import copy
import random
import torch
from typing import List, Dict, Optional

from recall.config import RecallConfig
from recall.base import load_model_and_tokenizer, get_device
from recall.state import get_lora_state, set_lora_state
from recall.trainer import _SFTDataset
from recall.inference import generate as _generate


# ────────────────────────────────────────────────────────────────────
# Naive SFT baseline — train each correction sequentially, no protection
# ────────────────────────────────────────────────────────────────────

class NaiveSFTBaseline:
    """Plain sequential SFT. No AVR, no two-stream, no replay.

    This is the 'before' picture — shows catastrophic forgetting clearly.
    After 10 corrections, correction #1 is usually forgotten.
    """

    def __init__(self, config: RecallConfig):
        self.config = config
        self.device = get_device(config)
        self.model, self.tokenizer = load_model_and_tokenizer(config)
        self._train_lr = config.train_lr

    def remember(self, input: str, target: str, **kwargs) -> str:
        """Train on the correction in-place. No protection."""
        from torch.utils.data import DataLoader
        import time

        pairs = [(input, target)]
        dataset = _SFTDataset(
            self.tokenizer, pairs,
            self.config.context_length, self.config.data_repeat)

        for n, p in self.model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        trainable = [p for p in self.model.parameters() if p.requires_grad]

        opt = torch.optim.AdamW(
            trainable, lr=self._train_lr,
            weight_decay=self.config.train_weight_decay)
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size,
            shuffle=True, drop_last=False)

        gs, tl = 0, 0.0
        t0 = time.time()
        for epoch in range(self.config.train_epochs):
            for batch in loader:
                self.model.train()
                out = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable, self.config.train_max_grad_norm)
                opt.step()
                tl += out.loss.item()
                gs += 1
        print(f"    [naive] {gs} steps, loss={tl/max(gs,1):.4f}, "
              f"{time.time()-t0:.0f}s", flush=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"naive_{gs}"

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        return _generate(
            self.model, self.tokenizer, prompt,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            device=self.device,
        )


# ────────────────────────────────────────────────────────────────────
# Replay SFT baseline — mix in 10% of old corrections each step
# ────────────────────────────────────────────────────────────────────

class ReplaySFTBaseline:
    """Standard CL baseline: replay buffer of old corrections, mixed in.

    Memory grows O(N * buffer_per_task). If Recall can't beat this,
    it can't beat anything — this is the lightest possible replay.
    """

    def __init__(self, config: RecallConfig,
                 replay_ratio: float = 0.1,
                 replay_buffer_per_task: int = 5):
        self.config = config
        self.device = get_device(config)
        self.model, self.tokenizer = load_model_and_tokenizer(config)
        self.replay_ratio = replay_ratio
        self.replay_buffer_per_task = replay_buffer_per_task
        self.replay_buffer: List[tuple] = []
        self._train_lr = config.train_lr

    def remember(self, input: str, target: str, **kwargs) -> str:
        from torch.utils.data import DataLoader
        import time

        # Mix in replay pairs
        current_pairs = [(input, target)]
        replay_pairs = []
        if self.replay_buffer:
            n_replay = max(1, int(len(current_pairs) * self.replay_ratio))
            replay_pairs = random.choices(
                self.replay_buffer, k=min(n_replay, len(self.replay_buffer)))
        all_pairs = current_pairs + replay_pairs
        random.shuffle(all_pairs)

        dataset = _SFTDataset(
            self.tokenizer, all_pairs,
            self.config.context_length, self.config.data_repeat)

        for n, p in self.model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        trainable = [p for p in self.model.parameters() if p.requires_grad]

        opt = torch.optim.AdamW(
            trainable, lr=self._train_lr,
            weight_decay=self.config.train_weight_decay)
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size,
            shuffle=True, drop_last=False)

        gs, tl = 0, 0.0
        t0 = time.time()
        for epoch in range(self.config.train_epochs):
            for batch in loader:
                self.model.train()
                out = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable, self.config.train_max_grad_norm)
                opt.step()
                tl += out.loss.item()
                gs += 1
        print(f"    [replay] {gs} steps, loss={tl/max(gs,1):.4f}, "
              f"{time.time()-t0:.0f}s ({len(replay_pairs)} replay)", flush=True)

        # Add this correction to the replay buffer
        self.replay_buffer.extend(current_pairs)
        if len(self.replay_buffer) > self.replay_buffer_per_task * 100:
            self.replay_buffer = self.replay_buffer[-self.replay_buffer_per_task * 100:]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"replay_{gs}"

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        return _generate(
            self.model, self.tokenizer, prompt,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            device=self.device,
        )


BASELINES = {
    "naive_sft": NaiveSFTBaseline,
    "replay_sft": ReplaySFTBaseline,
}
