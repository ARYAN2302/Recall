"""
eval.accuracy — scoring helpers.

Score a generation against a correction's check_tokens.
A generation "passes" if ALL check_tokens appear as substrings.
This is more lenient than exact match (generation length varies) but
stricter than just-contains-any-token.
"""
from __future__ import annotations
import re
from typing import List, Dict


def score_check_tokens(generation: str, check_tokens: List[str]) -> float:
    """Return 1.0 if all check_tokens are substrings of generation, else 0.0.

    Uses substring presence (case-sensitive) because:
      - code-style corrections are about specific tokens (def, :, ->, etc.)
      - generation length varies, so exact match is too strict
      - case matters for code (def vs DEF)
    """
    if not generation:
        return 0.0
    for token in check_tokens:
        if token not in generation:
            return 0.0
    return 1.0


def score_correction(generation: str, correction: Dict) -> float:
    """Score a single generation against a correction."""
    return score_check_tokens(generation, correction["check_tokens"])


def evaluate_accuracy_on(
    generate_fn,
    correction: Dict,
    n_samples: int = 1,
) -> float:
    """Generate n_samples times for correction['input'], return mean accuracy.

    generate_fn: callable(prompt: str) -> str
    """
    scores = []
    for _ in range(n_samples):
        gen = generate_fn(correction["input"])
        scores.append(score_correction(gen, correction))
    return sum(scores) / max(len(scores), 1)


def evaluate_all_corrections(
    generate_fn,
    corrections: List[Dict],
    n_samples: int = 1,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate accuracy on each correction. Returns {correction_id: accuracy}."""
    results = {}
    for i, corr in enumerate(corrections):
        acc = evaluate_accuracy_on(generate_fn, corr, n_samples)
        results[corr["id"]] = acc
        if verbose:
            marker = "✓" if acc > 0.5 else "✗"
            print(f"    {marker} {corr['id']:30s} acc={acc:.2f}", flush=True)
    return results


def mean_accuracy(scores: Dict[str, float]) -> float:
    """Mean across all corrections."""
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)
