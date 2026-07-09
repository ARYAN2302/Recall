"""
recall.local — in-process backend.

Runs the full Recall pipeline (LEARN → CONSOLIDATE → AVR → inference)
in the current process. No Modal, no remote calls.

This is what the Kaggle test uses, what local dev uses, and what
single-process deployments use. The Modal backend (modal_app.py) is a
thin remote-call wrapper around the same logic.

The backend holds:
    - the loaded model + tokenizer (lazy)
    - the current neocortex state dict (in CPU memory)
    - the best PPLs seen so far per correction (for AVR drift check)
    - the snapshot from before the latest correction (AVR repair target)
    - the correction queue (SQLite)

State is mutable across `remember()` calls — that's the whole point.
"""
from __future__ import annotations
import copy
import json
import time
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from .config import RecallConfig
from .base import load_model_and_tokenizer, get_device
from .state import get_lora_state, set_lora_state
from .trainer import learn_correction
from .avr import run_avr_loop, eval_correction_ppls
from .inference import generate as _generate
from .queue import CorrectionQueue


class LocalBackend:
    """In-process Recall backend. Loads model on first use."""

    def __init__(self, config: RecallConfig):
        self.config = config
        self.device = get_device(config)

        # Lazy-loaded
        self._model = None
        self._tokenizer = None

        # Continual learning state — persisted in memory across corrections
        self.neo_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_ppls: Dict[str, float] = {}
        self.completed_ids: List[str] = []
        self.last_snapshot: Optional[Dict[str, torch.Tensor]] = None
        # ^ neocortex state BEFORE the latest correction (AVR repair target)

        # Persistence
        data_dir = Path(config.data_dir or "./recall_data")
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.queue = CorrectionQueue(data_dir / "queue.db")

        # Set seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    # ── Lazy model loading ──

    def _ensure_model(self):
        if self._model is None:
            self._model, self._tokenizer = load_model_and_tokenizer(self.config)
            # Initialize neo_state from the fresh PEFT init (zeros for B,
            # Kaiming for A — same as reset_lora_to_peft_init would produce).
            self.neo_state = get_lora_state(self._model)
            print(f"  [backend] model loaded, neo_state initialized "
                  f"({len(self.neo_state)} LoRA tensors)", flush=True)

    # ── Public API ──

    def remember(self, input: str, target: str,
                 metadata: Optional[Dict] = None,
                 eval_pairs: Optional[List[List[str]]] = None) -> str:
        """Learn a single correction. Returns the correction id.

        Pipeline:
            1. Persist to queue
            2. Run LEARN (hippocampus train + consolidate to neocortex)
            3. If avr_every_n reached, run AVR verify-repair
            4. Mark correction as trained
        """
        self._ensure_model()

        # Build eval_pairs if not provided — default: probe on the
        # correction's own (input, target) so AVR can detect forgetting
        # of this specific correction later.
        if eval_pairs is None:
            eval_pairs = [[input, target]]

        # Persist
        cid = self.queue.add(input, target, metadata, eval_pairs)
        n_trained_before = self.queue.count_trained()
        print(f"\n{'='*60}", flush=True)
        print(f"  remember() → correction {cid} "
              f"(#{n_trained_before + 1} trained)", flush=True)
        print(f"{'='*60}", flush=True)

        # 1. LEARN: hippocampus train + consolidate
        pairs = [(input, target)]
        new_neo_state, info = learn_correction(
            self._model, self._tokenizer, self.neo_state, pairs,
            self.config, self.device, verbose=True)

        # Save the snapshot from this correction (for future AVR repairs)
        self.last_snapshot = info["neo_snapshot"]
        self.neo_state = new_neo_state

        # Apply the new neocortex state to the model (so AVR can verify)
        set_lora_state(self._model, self.neo_state, self.device)

        # 2. AVR (if it's time)
        self.queue.mark_trained(cid)
        self.completed_ids.append(cid)
        n_trained = self.queue.count_trained()

        # Initialize best_ppls for this correction
        current_ppls = eval_correction_ppls(
            self._model, self._tokenizer,
            [{"id": cid, "eval_pairs": eval_pairs}],
            trained_so_far=1,
            max_samples=self.config.avr_probe_samples,
            device=self.device,
        )
        if cid in current_ppls:
            self.best_ppls[cid] = current_ppls[cid]

        # Run AVR every N corrections
        avr_info = {"ran": False, "repair_steps": 0, "converged": True}
        if n_trained > 1 and n_trained % self.config.avr_every_n == 0:
            print(f"\n  [backend] AVR triggered (every {self.config.avr_every_n})",
                  flush=True)
            # Reload all trained corrections from queue to probe PPL on them
            all_corrections = self.queue.list_trained()
            avr_result = run_avr_loop(
                self._model, self._tokenizer,
                corrections=all_corrections,
                trained_so_far=len(all_corrections),
                best_ppls=self.best_ppls,
                completed_ids=self.completed_ids,
                neo_snapshot=self.last_snapshot,
                config=self.config,
                device=self.device,
                verbose=True,
            )
            # Update neo_state from the repaired model
            self.neo_state = get_lora_state(self._model)

            # Merge best_ppls updates
            for cid_, ppl_ in avr_result["best_ppls_updated"].items():
                if cid_ not in self.best_ppls or ppl_ < self.best_ppls[cid_]:
                    self.best_ppls[cid_] = ppl_

            # Log to queue
            self.queue.log_avr(
                after_correction=cid,
                repair_steps=avr_result["repair_steps"],
                converged=avr_result["converged"],
                drift_report={
                    "drifted": avr_result["drifted"],
                    "final_ppls": avr_result["final_ppls"],
                },
            )
            avr_info = {
                "ran": True,
                "repair_steps": avr_result["repair_steps"],
                "converged": avr_result["converged"],
                "drifted": list(avr_result["drifted"].keys()),
            }

        # 3. Save snapshot to disk (cheap, LoRA only)
        snap_path = self._save_snapshot(cid)
        self.queue.add_snapshot(str(snap_path), note=f"after {cid}")

        # 4. Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return cid

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate using the current committed neocortex state."""
        self._ensure_model()
        return _generate(
            self._model, self._tokenizer, prompt,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            device=self.device,
            neo_state=self.neo_state,
        )

    def status(self) -> Dict:
        """Return current state for CLI / dashboard."""
        return {
            "model_id": self.config.model_id,
            "device": self.device,
            "n_corrections_total": self.queue.count(),
            "n_corrections_trained": self.queue.count_trained(),
            "best_ppls": dict(self.best_ppls),
            "avr_every_n": self.config.avr_every_n,
            "data_dir": str(self.data_dir),
        }

    # ── Internal ──

    def _save_snapshot(self, cid: str) -> Path:
        """Save the current neocortex state to disk as safetensors."""
        from safetensors.torch import save_file
        snap_dir = self.data_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = snap_dir / f"neo_{cid}.safetensors"
        # safetensors requires contiguous CPU tensors with str keys
        state = {k: v.clone().contiguous() for k, v in self.neo_state.items()}
        save_file(state, str(path))
        return path
