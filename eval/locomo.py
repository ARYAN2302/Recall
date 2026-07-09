"""
eval.locomo — LOCOMO-MC10 benchmark loader.

Loads the Multiple-Choice version of LOCOMO (Long Conversation Memory)
from HuggingFace. This is the standard agent-memory benchmark that
Mem0 itself publishes scores on.

Dataset: Percena/locomo-mc10
- 1,986 multiple-choice items
- 5 categories: single-hop, multi-hop, temporal, open-domain, adversarial
- Each item has: question, 10 options, correct answer index, category

We extract (question, correct_answer_text) pairs for the forgetting curve.
The question is the prompt, the correct answer text is the target.

This is a NEUTRAL benchmark — we didn't write it. No bias toward Recall's
fine-tuning approach.
"""
from __future__ import annotations
import json
from typing import List, Dict, Optional
from pathlib import Path


def load_locomo_mc10(n: int = 50, cache_dir: str = None) -> List[Dict]:
    """Load N Q&A pairs from LOCOMO-MC10.

    Each returned item:
        {
            "id": str,
            "question": str,
            "answer": str,        # the correct answer text
            "check_tokens": list,  # key tokens for substring matching
            "category": str,       # SH / MH / TR / OD / ADV
            "options": list,       # all 10 options
            "correct_idx": int,
        }

    Args:
        n: number of Q&A pairs to return
        cache_dir: HF datasets cache dir
    """
    from datasets import load_dataset

    print(f"  [locomo] loading Percena/locomo-mc10 from HuggingFace...", flush=True)
    ds = load_dataset("Percena/locomo-mc10", split="train", cache_dir=cache_dir)
    print(f"  [locomo] loaded {len(ds)} items", flush=True)

    pairs = []
    for i, item in enumerate(ds):
        if i >= n:
            break

        question = item.get("question", "")
        options = item.get("options", [])
        correct_idx = item.get("answer", 0)
        category = item.get("category", "unknown")

        # Handle different possible field names
        if not options and "choices" in item:
            options = item["choices"]
        if isinstance(correct_idx, str):
            try:
                correct_idx = int(correct_idx)
            except ValueError:
                correct_idx = 0

        if correct_idx >= len(options) or not question:
            continue

        answer_text = options[correct_idx] if isinstance(options[correct_idx], str) else str(options[correct_idx])

        # Build check_tokens from the answer text — key words that must appear
        # For MC answers, the full answer text is usually short enough to match
        check_tokens = _extract_check_tokens(answer_text)

        pairs.append({
            "id": f"locomo_{i}",
            "question": question,
            "answer": answer_text,
            "check_tokens": check_tokens,
            "category": category,
            "options": options,
            "correct_idx": correct_idx,
        })

    print(f"  [locomo] extracted {len(pairs)} Q&A pairs", flush=True)
    return pairs


def _extract_check_tokens(answer_text: str) -> List[str]:
    """Extract key tokens from an answer for substring matching.

    For short answers (common in MC), we use the full answer.
    For longer answers, we extract the most distinctive words.
    """
    answer_text = answer_text.strip()
    # For short answers (under 100 chars), use the whole thing
    if len(answer_text) <= 100:
        return [answer_text.lower()]

    # For longer answers, extract content words (skip stopwords)
    stopwords = {"the", "a", "an", "is", "was", "are", "were", "to", "of",
                 "in", "on", "at", "by", "for", "with", "and", "or", "but",
                 "not", "this", "that", "it", "he", "she", "they", "we",
                 "you", "i", "as", "from", "be", "been", "being", "have",
                 "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "may", "might", "can", "shall"}
    words = [w.lower().strip(".,!?;:\"'()[]{}") for w in answer_text.split()]
    content_words = [w for w in words if w and w not in stopwords and len(w) > 2]
    # Take up to 5 most distinctive content words
    return content_words[:5] if content_words else [answer_text.lower()[:50]]


def load_locomo_from_file(path: str, n: int = 50) -> List[Dict]:
    """Fallback: load from a local JSON file if HuggingFace is unavailable.

    Expected format: list of items with question, options, answer (index).
    """
    with open(path) as f:
        data = json.load(f)

    pairs = []
    for i, item in enumerate(data[:n]):
        question = item.get("question", "")
        options = item.get("options", [])
        correct_idx = item.get("answer", 0)
        category = item.get("category", "unknown")

        if correct_idx >= len(options) or not question:
            continue

        answer_text = options[correct_idx] if isinstance(options[correct_idx], str) else str(options[correct_idx])
        check_tokens = _extract_check_tokens(answer_text)

        pairs.append({
            "id": f"locomo_{i}",
            "question": question,
            "answer": answer_text,
            "check_tokens": check_tokens,
            "category": category,
            "options": options,
            "correct_idx": correct_idx,
        })

    return pairs
