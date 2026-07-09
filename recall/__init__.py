"""
recall — agent memory that actually learns.

Continual LoRA + closed-form repair (AVR). No retrieval layer at inference.
The model's weights change as you teach it corrections.

    from recall import Recall

    mem = Recall()
    mem.remember("write a function", "def f() -> None:\\n    ...")
    mem.generate("write a function")  # → produces the corrected format

Built on AVR (Anchor-Verify-Repair) and the two-stream hippocampus-neocortex
training loop. See README for the full story.
"""
from .config import RecallConfig
from .api import Recall

__version__ = "0.1.0"

__all__ = [
    "Recall",
    "RecallConfig",
    "__version__",
]
