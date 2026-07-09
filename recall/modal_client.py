"""
recall.modal_client — client-side wrapper around Modal Functions.

The Recall SDK uses this when modal=True. Each method maps to a remote
Modal Function call. State (queue, snapshots, best_ppls) lives on the
Modal Volume — the local SDK is stateless.

Cost notes:
    - remember() spawns a T4 function (async by default — fire-and-forget)
    - generate() calls a T4 function synchronously (small, fast)
    - status() calls a CPU function (cheapest)
    - eval forgetting-curve calls an A10G function (big, only for launch)
"""
from __future__ import annotations
from typing import Optional, Dict, List
import os

from .config import RecallConfig


class ModalBackend:
    """Remote backend. Calls Modal Functions."""

    def __init__(self, config: RecallConfig):
        self.config = config
        # We import modal lazily so the package works without modal installed
        try:
            from .modal_app import (
                train_correction, run_inference, get_status,
                run_forgetting_curve, render_chart,
            )
        except ImportError as e:
            raise ImportError(
                "Modal backend requires `pip install modal`. "
                f"Original error: {e}"
            ) from e

        self._train = train_correction
        self._infer = run_inference
        self._status = get_status
        self._curve = run_forgetting_curve
        self._render = render_chart
        self._config_dict = config.to_dict()

    def remember(
        self,
        input: str,
        target: str,
        metadata: Optional[Dict] = None,
        eval_pairs: Optional[List[List[str]]] = None,
    ) -> str:
        """Queue correction + spawn async training."""
        # First persist to the remote queue (cheap CPU call)
        # Then spawn the T4 training function (async via .spawn)
        import modal
        from .modal_app import app, VOLUME

        # The train_correction function pulls from the queue on the Volume,
        # so we need to add the correction there first.
        # For v1, we do both in one call — train_correction takes the
        # correction payload directly, persists it, then trains.
        # (Simpler than a separate add endpoint.)
        fut = self._train.spawn(
            correction_payload={
                "input": input,
                "target": target,
                "metadata": metadata or {},
                "eval_pairs": eval_pairs or [[input, target]],
            },
            config_dict=self._config_dict,
        )
        return fut.call_id  # caller can poll with .status()

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        return self._infer.remote(
            prompt, self._config_dict,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens)

    def status(self) -> Dict:
        return self._status.remote(self._config_dict)

    def run_forgetting_curve(self, n_corrections: int = 50,
                             include_baselines: bool = True) -> Dict:
        """Run the full benchmark on A10G. Returns curve data."""
        return self._curve.remote(
            self._config_dict, n_corrections, include_baselines)

    def render_chart(self, curve_data: dict) -> str:
        """Render the chart on CPU. Returns path to PNG."""
        return self._render.remote(curve_data)
