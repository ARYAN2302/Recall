"""
eval.baselines.mem0_runner — Mem0 wrapper for the forgetting curve.

Mem0 is the market-leading agent memory tool. It stores facts in a
vector DB and retrieves them at inference time. The model itself never
changes — it just retrieves harder.

This wrapper configures Mem0 with the SAME base model (Qwen3-0.6B) as
Recall, so the comparison is fair: same model, same data, different
memory approach (retrieval vs learning).

Requires: pip install mem0ai
"""
from __future__ import annotations
import time
from typing import Optional, List, Dict

from recall.config import RecallConfig
from recall.base import load_model_and_tokenizer, get_device
from recall.inference import generate as _generate


class Mem0Baseline:
    """Mem0 retrieval-based memory, configured for local Qwen3-0.6B.

    Fair comparison with Recall:
        - Same base model (Qwen3-0.6B)
        - Same corrections
        - Same evaluation

    Difference:
        - Mem0 stores facts in a vector DB, retrieves at inference
        - Recall trains LoRA, generates from weights (no retrieval)
    """

    def __init__(self, config: RecallConfig):
        self.config = config
        self.device = get_device(config)

        # Load the base model for generation (same as Recall uses)
        print("  [mem0] loading Qwen3-0.6B for generation...", flush=True)
        self.model, self.tokenizer = load_model_and_tokenizer(config)
        print("  [mem0] model loaded", flush=True)

        # Initialize Mem0 with local models (no OpenAI key needed)
        print("  [mem0] initializing Mem0 with local Qwen + HF embeddings...",
              flush=True)
        try:
            from mem0 import Memory
        except ImportError:
            raise ImportError(
                "mem0ai not installed. Run: pip install mem0ai")

        # Configure Mem0 to use local models — fair comparison
        mem0_config = {
            "llm": {
                "provider": "huggingface",
                "config": {
                    "model": config.model_id,
                    "temperature": 0,
                    "max_tokens": 200,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                },
            },
        }

        try:
            self.mem0 = Memory.from_config(mem0_config)
            print("  [mem0] initialized with HuggingFace provider", flush=True)
        except Exception as e:
            print(f"  [mem0] HuggingFace config failed ({e}), trying default...",
                  flush=True)
            # Fallback: use Mem0's default local mode
            self.mem0 = Memory()
            print("  [mem0] initialized with default config", flush=True)

        self._corrections_added = 0

    def remember(self, input: str, target: str, **kwargs) -> str:
        """Store a correction in Mem0's vector DB.

        Mem0 extracts facts from the input/target and stores them as
        searchable memories. No model weights change.
        """
        try:
            # Mem0's add() takes messages in chat format
            messages = [
                {"role": "user", "content": input},
                {"role": "assistant", "content": target},
            ]
            self.mem0.add(messages, user_id="benchmark_user")
            self._corrections_added += 1
            print(f"    [mem0] stored correction #{self._corrections_added}",
                  flush=True)
        except Exception as e:
            print(f"    [mem0] add failed: {e}", flush=True)
            return f"mem0_error"

        return f"mem0_{self._corrections_added}"

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate an answer using retrieval-augmented generation.

        1. Search Mem0 for memories relevant to the prompt
        2. Inject retrieved memories as context
        3. Generate with the base model

        This is the standard RAG pipeline — retrieve then generate.
        """
        # 1. Retrieve relevant memories from Mem0
        retrieved_context = ""
        try:
            results = self.mem0.search(prompt, user_id="benchmark_user")
            if results and isinstance(results, list):
                # Extract the memory content
                memories = []
                for r in results[:5]:  # top 5 memories
                    if isinstance(r, dict):
                        memories.append(r.get("memory", r.get("content", str(r))))
                    else:
                        memories.append(str(r))
                if memories:
                    retrieved_context = "\n".join(f"- {m}" for m in memories)
        except Exception as e:
            print(f"    [mem0] search failed: {e}", flush=True)

        # 2. Build the augmented prompt
        if retrieved_context:
            full_prompt = (
                f"Context from memory:\n{retrieved_context}\n\n"
                f"Question: {prompt}\n"
                f"Answer:"
            )
        else:
            full_prompt = prompt

        # 3. Generate with the base model (no LoRA — Mem0 doesn't train)
        return _generate(
            self.model, self.tokenizer, full_prompt,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            device=self.device,
        )

    def status(self) -> Dict:
        return {
            "n_corrections_added": self._corrections_added,
            "memory_backend": "mem0 (vector DB + retrieval)",
        }
