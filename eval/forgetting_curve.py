"""
eval.forgetting_curve — the viral benchmark.

After each new correction, evaluate accuracy on correction #1 (and
optionally all prior corrections). Plot accuracy vs correction index.

Expected shape:
    - Naive SFT: starts ~1.0, decays to ~0.3 by correction 10, flatlines
    - Replay SFT: similar but slower decay
    - Recall: starts ~1.0, stays ≥0.85 across all corrections (AVR catches drift)

That chart is the entire X thread.

Protocol:
    1. For each system (recall, naive_sft, replay_sft):
       a. Initialize fresh model
       b. For i in 1..N:
          - remember(correction[i])
          - evaluate accuracy on correction[0] (always the first one)
          - record (i, accuracy)
    2. Return {system: [(i, accuracy), ...]}
    3. Render chart
"""
from __future__ import annotations
import time
import gc
import torch
from typing import List, Dict, Optional

from recall.config import RecallConfig
from recall import Recall
from recall.base import get_device

from .corrections import get_corrections
from .accuracy import evaluate_accuracy_on, mean_accuracy


def run_curve_for_system(
    system_name: str,
    config: RecallConfig,
    corrections: List[Dict],
    eval_on: str = "first",
    verbose: bool = True,
) -> List[Dict]:
    """Run the forgetting curve for a single system.

    Args:
        system_name: "recall" | "naive_sft" | "replay_sft" | "mem0"
        config: RecallConfig (each system gets a fresh copy)
        corrections: list of correction dicts (from eval.corrections OR eval.locomo)
        eval_on: "first" = always eval on correction[0] (classic curve)
                 "all" = eval on all corrections seen so far (mean acc)
        verbose: print progress

    Returns:
        [{"i": i, "accuracy": float, "elapsed_s": float}, ...]
    """
    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"  Running curve for: {system_name}", flush=True)
        print(f"{'='*60}", flush=True)

    # Fresh config per system so they don't share state
    sys_config = type(config)(**config.to_dict())
    sys_config.data_dir = f"{config.data_dir or './recall_data'}_{system_name}"

    # Initialize the system
    if system_name == "recall":
        system = Recall(config=sys_config)
        generate_fn = system.generate
        remember_fn = system.remember
    elif system_name == "mem0":
        from eval.baselines.mem0_runner import Mem0Baseline
        system = Mem0Baseline(sys_config)
        generate_fn = system.generate
        remember_fn = system.remember
    elif system_name in ("naive_sft", "replay_sft"):
        from eval.baselines import BASELINES
        cls = BASELINES[system_name]
        system = cls(sys_config)
        generate_fn = system.generate
        remember_fn = system.remember
    else:
        raise ValueError(f"Unknown system: {system_name}")

    curve = []
    t0 = time.time()

    for i, corr in enumerate(corrections):
        if verbose:
            print(f"\n  --- correction {i+1}/{len(corrections)}: {corr['id']} ---",
                  flush=True)

        # Teach
        remember_fn(corr["input"], corr["target"])

        # Evaluate
        if eval_on == "first":
            acc = evaluate_accuracy_on(generate_fn, corrections[0])
        elif eval_on == "all":
            # Mean accuracy on all corrections seen so far
            scores = {}
            for prior_corr in corrections[:i + 1]:
                scores[prior_corr["id"]] = evaluate_accuracy_on(
                    generate_fn, prior_corr)
            acc = mean_accuracy(scores)
        else:
            raise ValueError(f"Unknown eval_on: {eval_on}")

        elapsed = time.time() - t0
        curve.append({
            "i": i + 1,
            "correction_id": corr["id"],
            "accuracy": acc,
            "elapsed_s": elapsed,
        })
        if verbose:
            print(f"  → accuracy on correction[0] = {acc:.3f}  "
                  f"({elapsed:.0f}s elapsed)", flush=True)

        # Free GPU between iterations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Cleanup
    del system
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return curve


def run_full_curve(
    config: RecallConfig,
    n_corrections: int = 50,
    include_baselines: bool = True,
    eval_on: str = "first",
    verbose: bool = True,
) -> Dict:
    """Run the forgetting curve for Recall + baselines. Returns chart data.

    Args:
        n_corrections: 15 for smoke, 50 for launch
        include_baselines: include naive_sft + replay_sft (slower but
            needed for the comparison chart)
        eval_on: "first" or "all"

    Returns:
        {
            "config": config.to_dict(),
            "n_corrections": int,
            "corrections": [correction_ids],
            "curves": {
                "recall": [{"i": 1, "accuracy": 1.0, ...}, ...],
                "naive_sft": [...],
                "replay_sft": [...],
            },
            "summary": {
                "recall": {"start_acc": float, "end_acc": float, "delta": float},
                ...
            }
        }
    """
    corrections = get_corrections(n_corrections)
    systems = ["recall"]
    if include_baselines:
        systems.extend(["naive_sft", "replay_sft"])

    curves = {}
    for sys_name in systems:
        try:
            curve = run_curve_for_system(
                sys_name, config, corrections, eval_on=eval_on,
                verbose=verbose)
            curves[sys_name] = curve
        except Exception as e:
            print(f"\n  [!] {sys_name} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            curves[sys_name] = []

    # Summary
    summary = {}
    for sys_name, curve in curves.items():
        if not curve:
            summary[sys_name] = None
            continue
        start_acc = curve[0]["accuracy"]
        end_acc = curve[-1]["accuracy"]
        summary[sys_name] = {
            "start_acc": start_acc,
            "end_acc": end_acc,
            "delta": end_acc - start_acc,
            "min_acc": min(p["accuracy"] for p in curve),
            "total_s": curve[-1]["elapsed_s"],
        }

    return {
        "config": config.to_dict(),
        "n_corrections": n_corrections,
        "corrections": [c["id"] for c in corrections],
        "eval_on": eval_on,
        "curves": curves,
        "summary": summary,
    }
