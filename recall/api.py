"""
recall.api — the public Recall class. The only thing users import.

Two methods: generate() and remember(). Everything else is plumbing.

    from recall import Recall

    mem = Recall()                                          # Qwen3-0.6B default
    print(mem.generate("write a function to sort a list"))  # bare function

    mem.remember("always use type hints and docstrings")
    print(mem.generate("write a function to sort a list"))  # typed + documented

The class auto-detects whether to run locally (in-process) or via Modal.
If MODAL_APP_NAME env var is set OR `modal=True` is passed, the backend
is ModalBackend. Otherwise LocalBackend.

Both backends implement the same surface — generate(), remember(), status().
"""
from __future__ import annotations
import os
from typing import Optional, Dict, List

from .config import RecallConfig


class Recall:
    """Agent memory that actually learns.

    Args:
        model: HF model id. Default "Qwen/Qwen3-0.6B".
        modal: if True, use Modal backend. Default False (local).
        config: optional RecallConfig for advanced tuning.
        data_dir: where to persist state. Default ./recall_data.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-0.6B",
        modal: bool = False,
        config: Optional[RecallConfig] = None,
        data_dir: Optional[str] = None,
    ):
        if config is None:
            config = RecallConfig(model_id=model)
        else:
            # Override model id if user passed one
            if model != "Qwen/Qwen3-0.6B":
                config.model_id = model
        if data_dir is not None:
            config.data_dir = data_dir
        self.config = config

        # Pick backend
        use_modal = modal or bool(os.environ.get("MODAL_APP_NAME"))
        if use_modal:
            try:
                from .modal_client import ModalBackend
                self._backend = ModalBackend(config)
            except ImportError as e:
                raise ImportError(
                    "Modal backend requires `pip install modal`. "
                    f"Original error: {e}"
                ) from e
        else:
            from .local import LocalBackend
            self._backend = LocalBackend(config)

    def remember(
        self,
        instruction: str,
        target: str,
        metadata: Optional[Dict] = None,
        eval_pairs: Optional[List[List[str]]] = None,
    ) -> str:
        """Teach the model a correction.

        Triggers a background continual LoRA update + AVR verify. The
        model's weights actually change — no retrieval layer needed.

        Args:
            instruction: the user's input prompt that should now produce
                a different answer. e.g. "write a function to sort a list".
            target: the desired answer the model should now produce for
                this kind of input. e.g. "def sort_list(lst: list) -> list: ...".
            metadata: optional dict, stored alongside the correction for
                audit. e.g. {"source": "user_correction", "timestamp": ...}.
            eval_pairs: optional list of [prompt, answer] pairs used as
                AVR drift probes. Default: [[instruction, target]].

        Returns:
            correction_id (str). Use with .status() to track training.
        """
        return self._backend.remember(
            instruction, target, metadata=metadata, eval_pairs=eval_pairs)

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate using the latest committed adapter.

        The model has been continually updated by prior remember() calls.
        No retrieval step — the weights themselves hold the corrections.
        """
        return self._backend.generate(prompt, max_new_tokens=max_new_tokens)

    def status(self) -> Dict:
        """Return backend state: correction count, PPL drift, AVR history."""
        return self._backend.status()

    # Convenience aliases
    learn = remember
    __call__ = generate
