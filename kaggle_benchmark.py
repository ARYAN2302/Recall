"""
kaggle_benchmark.py — LOCOMO forgetting curve: Recall vs Mem0.

Uses the LOCOMO-MC10 benchmark (the standard agent-memory benchmark
that Mem0 itself publishes scores on). NOT our custom corrections.

Protocol:
    1. Load N Q&A pairs from LOCOMO-MC10 (HuggingFace)
    2. For each system (recall, mem0):
       a. Feed Q&A pairs sequentially (remember)
       b. After each new pair, test: can the system still answer Q1?
    3. Plot the forgetting curve

This is a NEUTRAL benchmark — we didn't write the questions. No bias
toward Recall's fine-tuning approach.

USAGE:
    !curl -s https://raw.githubusercontent.com/ARYAN2302/Recall/main/kaggle_benchmark.py -o /kaggle/working/kaggle_benchmark.py && python /kaggle/working/kaggle_benchmark.py
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path

# ============================================================
# 0. Environment
# ============================================================

IS_KAGGLE = Path("/kaggle/working").exists()
OUTPUT_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("./benchmark_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "benchmark_log.txt"

print(f"[env] Output dir: {OUTPUT_DIR}")

class _Tee:
    def __init__(self, *streams):
        object.__setattr__(self, "_streams", streams)
    def write(self, s):
        for st in self._streams:
            try: st.write(s); st.flush()
            except: pass
    def flush(self):
        for st in self._streams:
            try: st.flush()
            except: pass
    def isatty(self): return False
    def fileno(self): return self._streams[0].fileno()
    def __getattr__(self, name):
        return getattr(self._streams[0], name)

_log_fh = open(LOG_PATH, "w")
_running_as_script = "__file__" in globals() and not hasattr(__builtins__, "__IPYTHON__")
if _running_as_script:
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)

def log(msg, flush=True):
    print(msg, flush=flush)
    _log_fh.write(str(msg) + "\n")
    _log_fh.flush()


# ============================================================
# 1. Setup
# ============================================================

def setup():
    log("\n" + "="*60)
    log("SETUP")
    log("="*60)

    # Uninstall torchao (incompatible with our PEFT)
    log("  uninstalling torchao...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False, capture_output=True)

    # Install deps — including mem0ai and datasets for LOCOMO
    pkgs = [
        "torch>=2.2.0",
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "sentencepiece",
        "datasets>=2.14.0",           # for loading LOCOMO-MC10
        "sentence-transformers>=2.2.0",  # for Mem0 embeddings
        "mem0ai>=0.1.0",              # the baseline we're comparing against
    ]
    log(f"  installing {len(pkgs)} packages (including mem0ai + datasets)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
    log("  done.")

    # Clone repo
    repo_dir = OUTPUT_DIR / "recall_repo"
    if not (repo_dir / "recall" / "api.py").exists():
        if repo_dir.exists():
            import shutil; shutil.rmtree(repo_dir)
        log("  cloning https://github.com/ARYAN2302/Recall ...")
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ARYAN2302/Recall.git", str(repo_dir)],
                       check=True)
    sys.path.insert(0, str(repo_dir))

    # Verify imports
    from recall import Recall, RecallConfig
    from eval.locomo import load_locomo_mc10
    from eval.forgetting_curve import run_curve_for_system
    from eval.render import render_forgetting_curve
    log("  imports OK")
    return repo_dir


# ============================================================
# 2. Run the LOCOMO forgetting curve
# ============================================================

def run_benchmark():
    log("\n" + "="*60)
    log("BENCHMARK: LOCOMO-MC10 forgetting curve")
    log("  Systems: recall vs mem0")
    log("  Data: LOCOMO-MC10 (neutral benchmark, not our custom corrections)")
    log("="*60)

    from recall.config import RecallConfig
    from eval.locomo import load_locomo_mc10
    from eval.forgetting_curve import run_curve_for_system
    from eval.render import render_forgetting_curve
    from eval.accuracy import evaluate_accuracy_on
    import torch, gc

    # Load LOCOMO-MC10 Q&A pairs
    N_QUESTIONS = 20  # enough for a meaningful curve, fits in Kaggle time
    corrections = load_locomo_mc10(n=N_QUESTIONS,
                                   cache_dir=str(OUTPUT_DIR / "hf_cache"))

    if not corrections:
        log("  FAILED to load LOCOMO-MC10. Cannot proceed.")
        return None, None

    log(f"  loaded {len(corrections)} Q&A pairs from LOCOMO-MC10")
    log(f"  categories: {set(c['category'] for c in corrections)}")
    log(f"  sample Q: {corrections[0]['question'][:100]}...")
    log(f"  sample A: {corrections[0]['answer'][:100]}...")

    # Config — same as the smoke test that passed
    config = RecallConfig(
        model_id="Qwen/Qwen3-0.6B",
        lora_rank=16,
        lora_targets=("q_proj", "v_proj"),
        train_epochs=15,
        consolidation_epochs=8,
        data_repeat=30,
        batch_size=2,
        context_length=128,
        train_lr=3e-4,
        consolidation_lr=1.5e-4,
        avr_probe_samples=10,
        avr_every_n=5,
        max_new_tokens=64,
        seed=42,
    )

    # Run each system
    curves = {}
    for sys_name in ["recall", "mem0"]:
        log(f"\n{'='*60}")
        log(f"  SYSTEM: {sys_name}")
        log(f"{'='*60}")

        sys_config = type(config)(**config.to_dict())
        sys_config.data_dir = str(OUTPUT_DIR / f"locomo_{sys_name}")

        try:
            curve = run_curve_for_system(
                sys_name, sys_config, corrections,
                eval_on="first", verbose=True)
            curves[sys_name] = curve

            # Print curve
            log(f"\n  --- {sys_name} curve ---")
            for p in curve:
                log(f"    [{sys_name}] corr {p['i']:2d}: "
                    f"acc={p['accuracy']:.3f} ({p['elapsed_s']:.0f}s)")
        except Exception as e:
            log(f"  [!] {sys_name} FAILED: {e}")
            import traceback; traceback.print_exc()
            curves[sys_name] = []

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build chart data
    chart_data = {
        "config": config.to_dict(),
        "n_corrections": len(corrections),
        "benchmark": "LOCOMO-MC10",
        "corrections": [c["id"] for c in corrections],
        "categories": [c["category"] for c in corrections],
        "eval_on": "first",
        "curves": curves,
        "summary": {},
    }
    for sys_name, curve in curves.items():
        if curve:
            chart_data["summary"][sys_name] = {
                "start_acc": curve[0]["accuracy"],
                "end_acc": curve[-1]["accuracy"],
                "delta": curve[-1]["accuracy"] - curve[0]["accuracy"],
                "min_acc": min(p["accuracy"] for p in curve),
                "total_s": curve[-1]["elapsed_s"],
            }

    # Render chart
    chart_path = str(OUTPUT_DIR / "forgetting_curve.png")
    try:
        render_forgetting_curve(chart_data, chart_path,
            title="Recall vs Mem0: forgetting curve on LOCOMO-MC10")
        log(f"\n  chart saved to: {chart_path}")
    except Exception as e:
        log(f"  [!] chart render failed: {e}")

    return chart_data, chart_path


# ============================================================
# 3. Main
# ============================================================

def main():
    t0 = time.time()
    log("="*60)
    log("RECALL vs MEM0 — LOCOMO-MC10 BENCHMARK")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)

    setup()
    chart_data, chart_path = run_benchmark()

    if chart_data is None:
        log("\nBenchmark failed — could not load LOCOMO-MC10.")
        _log_fh.close()
        return

    # Final summary
    elapsed = time.time() - t0
    log("\n" + "="*60)
    log("FINAL SUMMARY")
    log("="*60)
    log(f"  Benchmark: LOCOMO-MC10 (neutral, not our custom corrections)")
    log(f"  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    for sys_name, s in chart_data["summary"].items():
        log(f"  {sys_name}:")
        log(f"    start={s['start_acc']:.3f} end={s['end_acc']:.3f} "
            f"delta={s['delta']:+.3f} min={s['min_acc']:.3f} "
            f"({s['total_s']:.0f}s)")
    log(f"  Chart: {chart_path}")

    # Save results
    results = {
        "started_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "elapsed_s": elapsed,
        "benchmark": "LOCOMO-MC10",
        "data": chart_data,
        "chart_path": chart_path,
    }
    results_path = OUTPUT_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results: {results_path}")
    log(f"  Log: {LOG_PATH}")
    _log_fh.close()


if __name__ == "__main__":
    main()
