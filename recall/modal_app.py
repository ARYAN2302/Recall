"""
recall.modal_app — Modal backend definitions.

Defines the Modal Functions that do the heavy lifting (training, AVR,
inference) on Modal GPUs. The local SDK (recall.modal_client) calls
these remotely.

To use:
    1. pip install modal
    2. modal token new
    3. Set MODAL_APP_NAME=recall-memory in env, OR pass modal=True to Recall()
    4. python your_script.py

The local SDK only needs the modal_client module; this file lives on
the Modal side and is uploaded when you `modal deploy` or via the
recall.cli:deploy command.

Cost discipline:
    - T4 ($0.59/hr) for dev iteration + small training runs
    - A10G ($0.81/hr) for the full forgetting-curve benchmark
    - CPU functions for chart rendering + queue mgmt
"""
from __future__ import annotations
import modal

APP_NAME = "recall-memory"
VOLUME = modal.Volume.from_name("recall-data", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.45.0",
        "peft==0.13.0",
        "accelerate==0.34.0",
        "safetensors==0.4.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "sentencepiece",
    )
    .add_local_python_source("recall", "eval")
)

app = modal.App(APP_NAME, image=IMAGE)


@app.function(
    gpu="T4",
    volumes={"/data": VOLUME},
    timeout=600,  # 10 min cap per training call
    cpu=2,
    memory=8192,
)
def train_correction(correction_id: str, config_dict: dict) -> dict:
    """Train a single correction on Modal T4.

    Args:
        correction_id: id of the correction in the queue
        config_dict: serialized RecallConfig

    Returns:
        {"correction_id": ..., "status": ..., "avr_ran": bool, ...}
    """
    import sys
    sys.path.insert(0, "/root")

    from recall.config import RecallConfig
    from recall.local import LocalBackend

    config = RecallConfig.from_dict(config_dict)
    config.data_dir = "/data"  # use the Modal Volume
    backend = LocalBackend(config)

    # Pull the correction from the queue
    corr = backend.queue.get(correction_id)
    if corr is None:
        return {"correction_id": correction_id, "status": "not_found"}

    cid = backend.remember(
        corr["input"], corr["target"],
        metadata=corr["metadata"], eval_pairs=corr["eval_pairs"])

    VOLUME.commit()  # persist queue + snapshots

    return {
        "correction_id": cid,
        "status": "trained",
        "avr_ran": backend.queue.count_trained() % config.avr_every_n == 0,
        "n_trained": backend.queue.count_trained(),
    }


@app.function(
    gpu="T4",
    volumes={"/data": VOLUME},
    timeout=120,
    cpu=1,
    memory=4096,
)
def run_inference(prompt: str, config_dict: dict,
                  max_new_tokens: int = 64) -> str:
    """Generate using the latest committed adapter."""
    import sys
    sys.path.insert(0, "/root")

    from recall.config import RecallConfig
    from recall.local import LocalBackend

    config = RecallConfig.from_dict(config_dict)
    config.data_dir = "/data"
    backend = LocalBackend(config)
    return backend.generate(prompt, max_new_tokens=max_new_tokens)


@app.function(
    gpu="A10G",
    volumes={"/data": VOLUME},
    timeout=3600,  # 1 hr for the full forgetting-curve run
    cpu=2,
    memory=8192,
)
def run_forgetting_curve(config_dict: dict, n_corrections: int = 50,
                         include_baselines: bool = True) -> dict:
    """Run the forgetting-curve benchmark. Returns the chart data.

    This is the launch artifact — what goes in the README.
    """
    import sys
    sys.path.insert(0, "/root")

    from recall.config import RecallConfig
    from eval.forgetting_curve import run_full_curve

    config = RecallConfig.from_dict(config_dict)
    config.data_dir = "/data"

    results = run_full_curve(
        config, n_corrections=n_corrections,
        include_baselines=include_baselines)

    VOLUME.commit()
    return results


@app.function(
    cpu=2,
    volumes={"/data": VOLUME},
    timeout=300,
    memory=2048,
)
def render_chart(curve_data: dict, output_path: str = "/data/forgetting_curve.png") -> str:
    """Render the forgetting-curve chart. CPU-only — no GPU needed."""
    import sys
    sys.path.insert(0, "/root")

    from eval.render import render_forgetting_curve
    path = render_forgetting_curve(curve_data, output_path)
    VOLUME.commit()
    return path


@app.function(
    cpu=1,
    volumes={"/data": VOLUME},
    timeout=30,
    memory=512,
)
def get_status(config_dict: dict) -> dict:
    """Return backend status. Cheap CPU call."""
    import sys
    sys.path.insert(0, "/root")

    from recall.config import RecallConfig
    from recall.local import LocalBackend

    config = RecallConfig.from_dict(config_dict)
    config.data_dir = "/data"
    backend = LocalBackend(config)
    return backend.status()


if __name__ == "__main__":
    # `python recall/modal_app.py` deploys the app to Modal.
    print(f"Deploying {APP_NAME} to Modal...")
