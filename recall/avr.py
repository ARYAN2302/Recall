"""
recall.avr — Anchor-Verify-Repair.

The closed-form forgetting repair that makes Recall actually work.

After every N corrections, AVR:
    1. VERIFY: compute PPL on a sample of prior corrections.
       If PPL_now / PPL_best > threshold (1.15), that correction has drifted.
    2. REPAIR: θ ← (1-α)·θ + α·θ_snapshot  (closed-form interpolation)
       Repeat until drift resolves or max_steps is hit.

No replay buffer. No gradients at repair time. No labels.
Just snapshot interpolation — a convex combination in weight space
that pulls the model back toward the last known-good state for any
correction it's started to forget.

Ported from:
    - tiny-cl/avr/detectors.py (PPLRatioDetector)
    - tiny-cl/avr/operators.py (SnapshotInterp)
    - Living-Model/v34 (verify_drift + repair_toward_snapshot)

The math is identical. The packaging is cleaner.
"""
from __future__ import annotations
import math
import copy
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

from .config import RecallConfig
from .state import get_lora_state, set_lora_state


# ────────────────────────────────────────────────────────────────────
# VERIFY: PPL computation
# ────────────────────────────────────────────────────────────────────

def compute_ppl(
    model: nn.Module,
    tokenizer,
    pairs: List[Tuple[str, str]],
    max_samples: int = 50,
    device: str = "cuda",
) -> float:
    """Compute perplexity on (prompt, answer) pairs.

    PPL measures how well the model predicts the answer text given the
    prompt. A high PPL on a correction we used to predict well = forgetting.

    Args:
        pairs: list of (prompt, answer) — usually the targets of prior
            corrections, used as the drift probe.
        max_samples: cap to bound compute. 50 is enough signal.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for prompt, answer in pairs[:max_samples]:
        text = prompt + " " + answer + tokenizer.eos_token
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        total_loss += out.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def eval_correction_ppls(
    model: nn.Module,
    tokenizer,
    corrections: List[Dict],
    trained_so_far: int,
    max_samples: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    """PPL for each correction seen so far.

    Args:
        corrections: list of correction dicts, each with "id" and
            "eval_pairs" (a list of (prompt, answer) used to probe PPL).
        trained_so_far: only probe corrections[0:trained_so_far].
    """
    ppls = {}
    for i, corr in enumerate(corrections):
        if i >= trained_so_far:
            break
        ppls[corr["id"]] = compute_ppl(
            model, tokenizer, corr["eval_pairs"], max_samples, device)
    return ppls


def verify_drift(
    current_ppls: Dict[str, float],
    best_ppls: Dict[str, float],
    completed_ids: List[str],
    threshold: float = 1.15,
) -> Dict[str, Dict]:
    """Return dict of drifted correction_id → {current, best, ratio}.

    A correction is drifted if PPL_now / PPL_best > threshold.
    The default 1.15 means "15% worse than the best PPL we've seen
    on this correction" — calibrated in v23.
    """
    drifted = {}
    for cid in completed_ids:
        if cid not in current_ppls or cid not in best_ppls:
            continue
        ratio = (current_ppls[cid] / best_ppls[cid]
                 if best_ppls[cid] > 0 else 1.0)
        if ratio > threshold:
            drifted[cid] = {
                "current": current_ppls[cid],
                "best": best_ppls[cid],
                "ratio": ratio,
            }
    return drifted


# ────────────────────────────────────────────────────────────────────
# REPAIR: closed-form weight interpolation
# ────────────────────────────────────────────────────────────────────

def repair_toward_snapshot(
    model: nn.Module,
    snapshot_state: Dict[str, torch.Tensor],
    alpha: float = 0.1,
    device: str = "cuda",
) -> int:
    """Apply ONE repair step: θ ← (1-α)·θ + α·θ_snapshot.

    This is the closed-form repair. No gradients. No replay buffer.
    Just a convex combination in weight space.

    Args:
        snapshot_state: the neocortex state BEFORE the latest correction
            (i.e., the last known-good state). Repair pulls toward this.
        alpha: interpolation strength. 0.1 = 10% pull toward snapshot.

    Returns:
        Number of params adjusted.
    """
    n_adj = 0
    for n, p in model.named_parameters():
        if "lora_" in n and n in snapshot_state:
            snap_val = snapshot_state[n].to(device).to(p.data.dtype)
            p.data.copy_((1.0 - alpha) * p.data + alpha * snap_val)
            n_adj += 1
    return n_adj


# ────────────────────────────────────────────────────────────────────
# Full VERIFY-REPAIR loop
# ────────────────────────────────────────────────────────────────────

def run_avr_loop(
    model: nn.Module,
    tokenizer,
    corrections: List[Dict],
    trained_so_far: int,
    best_ppls: Dict[str, float],
    completed_ids: List[str],
    neo_snapshot: Dict[str, torch.Tensor],
    config: RecallConfig,
    device: str,
    verbose: bool = True,
) -> Dict:
    """Run the full VERIFY-REPAIR loop after a correction.

    1. Compute current PPLs on all prior corrections.
    2. Identify drifted ones (PPL_now / PPL_best > threshold).
    3. Repair loop: interpolate toward snapshot, re-verify, repeat.
    4. Stop when no drift remains OR max_steps is hit.

    Args:
        neo_snapshot: neocortex state BEFORE the latest correction.
            Repair pulls the current neocortex back toward this.

    Returns:
        {
            "drifted": {cid: {current, best, ratio}},
            "repair_steps": int,
            "converged": bool,
            "final_ppls": {cid: ppl},
            "best_ppls_updated": {cid: ppl},  # for caller to merge
        }
    """
    # 1. VERIFY
    current_ppls = eval_correction_ppls(
        model, tokenizer, corrections, trained_so_far,
        config.avr_probe_samples, device)

    drifted = verify_drift(
        current_ppls, best_ppls, completed_ids, config.drift_threshold)

    if verbose:
        if drifted:
            print(f"  [AVR] drift on {list(drifted.keys())}", flush=True)
            for cid, info in drifted.items():
                print(f"    {cid}: PPL={info['current']:.2f} / "
                      f"best={info['best']:.2f} = {info['ratio']:.2f}x",
                      flush=True)
        else:
            print(f"  [AVR] no drift", flush=True)

    if not drifted:
        # No repair needed — just update best_ppls if any improved
        best_updates = {}
        for cid, ppl in current_ppls.items():
            if cid not in best_ppls or ppl < best_ppls[cid]:
                best_updates[cid] = ppl
        return {
            "drifted": {},
            "repair_steps": 0,
            "converged": True,
            "final_ppls": current_ppls,
            "best_ppls_updated": best_updates,
        }

    # 2. REPAIR loop
    # Adaptive step cap: more drift → more steps allowed
    max_ratio = max(info["ratio"] for info in drifted.values())
    # log(ratio) / log(1/(1-α)) = how many α-steps to undo the drift
    if config.repair_alpha > 0 and max_ratio > 1.0:
        adaptive_cap = max(
            config.max_repair_steps,
            int(math.log(max_ratio) / math.log(1.0 / (1.0 - config.repair_alpha))) + 2,
        )
    else:
        adaptive_cap = config.max_repair_steps

    if verbose:
        print(f"  [AVR] adaptive step cap: {adaptive_cap} "
              f"(max ratio {max_ratio:.2f})", flush=True)

    still_drifted = drifted
    n_steps = 0
    for step in range(adaptive_cap):
        n_adj = repair_toward_snapshot(
            model, neo_snapshot, config.repair_alpha, device)
        n_steps += 1

        # Re-verify
        current_ppls = eval_correction_ppls(
            model, tokenizer, corrections, trained_so_far,
            config.avr_probe_samples, device)
        still_drifted = verify_drift(
            current_ppls, best_ppls, completed_ids, config.drift_threshold)

        if verbose:
            print(f"    [AVR] step {step+1}: {n_adj} params adjusted, "
                  f"still drifted: {list(still_drifted.keys()) if still_drifted else 'none'}",
                  flush=True)

        if not still_drifted:
            if verbose:
                print(f"  [AVR] converged at step {step+1}", flush=True)
            break

    if still_drifted and verbose:
        print(f"  [AVR] max steps ({adaptive_cap}) reached, drift remains "
              f"on {list(still_drifted.keys())}", flush=True)

    # Update best_ppls
    best_updates = {}
    for cid, ppl in current_ppls.items():
        if cid not in best_ppls or ppl < best_ppls[cid]:
            best_updates[cid] = ppl

    return {
        "drifted": drifted,
        "repair_steps": n_steps,
        "converged": not still_drifted,
        "final_ppls": current_ppls,
        "best_ppls_updated": best_updates,
    }
