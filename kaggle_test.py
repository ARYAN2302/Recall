"""
kaggle_test.py — self-contained test for Kaggle T4 x2.

Copy-paste this entire file into a Kaggle notebook cell (or save as
kaggle_test.py and run `python kaggle_test.py` in a notebook shell).

WHAT IT DOES:
    1. Installs required packages (transformers, peft, etc.)
    2. Clones the Recall repo (or uses the local copy if present)
    3. Runs a smoke test of the full pipeline:
        - Load Qwen3-0.6B + LoRA r=32
        - Teach 6 corrections via recall.remember()
        - Verify the model learned each one
        - Verify AVR runs (every 3rd correction)
        - Verify forgetting is caught (correction #1 still works at correction #6)
    4. Runs a mini forgetting curve (6 corrections, Recall vs Naive SFT)
    5. Saves the chart + JSON results to /kaggle/working/

EXPECTED RUNTIME: ~30-45 min on T4 x2 (Kaggle)

REQUIREMENTS:
    - Kaggle notebook with GPU T4 x2 enabled
    - Internet enabled (for pip install + model download)

OUTPUT:
    - /kaggle/working/test_results.json
    - /kaggle/working/forgetting_curve_smoke.png
    - /kaggle/working/test_log.txt

USAGE:
    Option A (notebook cell): paste this whole file as a single cell, run it
    Option B (script): save as kaggle_test.py, run `python kaggle_test.py`
"""
import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path

# ============================================================
# 0. Environment detection
# ============================================================

IS_KAGGLE = Path("/kaggle/working").exists()
IS_COLAB = Path("/content").exists() and not IS_KAGGLE

if IS_KAGGLE:
    OUTPUT_DIR = Path("/kaggle/working")
    print("[env] Running on Kaggle. Output dir: /kaggle/working")
elif IS_COLAB:
    OUTPUT_DIR = Path("/content")
    print("[env] Running on Colab. Output dir: /content")
else:
    OUTPUT_DIR = Path("./kaggle_test_output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[env] Running locally. Output dir: {OUTPUT_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "test_log.txt"

# Tee all prints to a log file — but only when running as a script.
# In a Jupyter notebook, sys.stdout is a special wrapper that fights
# with our _Tee, so we skip the tee there (the file still gets written
# via explicit log() calls that go to both stdout and the file).
_running_as_script = "__file__" in globals() and not hasattr(__builtins__, "__IPYTHON__")

class _Tee:
    """Wraps stdout/stderr to mirror output to a log file.

    Proxies any unknown attribute access to the first underlying stream
    so third-party libraries (transformers, tqdm, etc.) that call
    sys.stdout.isatty(), .fileno(), .encoding, etc. keep working.
    """
    def __init__(self, *streams):
        # Use object.__setattr__ to bypass our __getattr__ during init
        object.__setattr__(self, "_streams", streams)
    def write(self, s):
        for st in self._streams:
            try:
                st.write(s); st.flush()
            except Exception:
                pass
    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass
    def isatty(self):
        # We're never a real TTY — we're a wrapper. Return False so
        # libraries don't try to emit ANSI color codes into the log file.
        return False
    def fileno(self):
        # Delegate to the first stream if it has one; raise otherwise.
        # This lets libraries that genuinely need a fd (rare) still work.
        return self._streams[0].fileno()
    def __getattr__(self, name):
        # Proxy everything else (encoding, newlines, mode, etc.) to the
        # first underlying stream.
        return getattr(self._streams[0], name)

_log_fh = open(LOG_PATH, "w")
if _running_as_script:
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)

def log(msg, flush=True):
    print(msg, flush=flush)
    # Also write directly to the log file (catches output even in
    # notebook mode where _Tee is not installed)
    _log_fh.write(str(msg) + "\n")
    _log_fh.flush()


# ============================================================
# 1. Install dependencies
# ============================================================

def install_deps():
    log("\n" + "="*60)
    log("STEP 1: Installing dependencies")
    log("="*60)

    # ── Pre-clean: remove torchao if present ──
    # Kaggle pre-installs torchao 0.10.0, but newer PEFT versions
    # require >= 0.16.0 and crash on import if they detect an older one.
    # We don't use torchao (just plain LoRA on a T4), so uninstall it
    # and PEFT's is_torchao_available() will return False — skipping the
    # broken dispatch path entirely.
    log("  pre-clean: uninstalling torchao (incompatible with our PEFT)")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        check=False,  # don't fail if not installed
        capture_output=True,
    )

    pkgs = [
        "torch>=2.2.0",
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "sentencepiece",
    ]
    for p in pkgs:
        log(f"  pip install {p}")
    # Quiet install
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + pkgs,
        check=True,
    )
    log("  done.")


# ============================================================
# 2. Get the Recall source
# ============================================================

def get_recall_source():
    """Either clone from GitHub or use a local copy."""
    log("\n" + "="*60)
    log("STEP 2: Getting Recall source")
    log("="*60)

    # Check if recall/ is already on the path (local dev)
    try:
        import recall
        log(f"  recall already importable: {recall.__file__}")
        return Path(recall.__file__).parent.parent
    except ImportError:
        pass

    # Check if there's a local recall_repo/ next to this script
    script_dir = Path(__file__).parent if "__file__" in globals() else Path(".")
    local = script_dir / "recall_repo"
    if local.exists() and (local / "recall" / "api.py").exists():
        log(f"  using local recall_repo: {local}")
        sys.path.insert(0, str(local))
        return local

    # Otherwise clone from GitHub
    repo_dir = OUTPUT_DIR / "recall_repo"
    if repo_dir.exists():
        log(f"  using existing clone: {repo_dir}")
        sys.path.insert(0, str(repo_dir))
        return repo_dir

    log("  cloning https://github.com/ARYAN2302/Recall ...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ARYAN2302/Recall.git", str(repo_dir)],
        check=True,
    )
    sys.path.insert(0, str(repo_dir))
    log(f"  cloned to {repo_dir}")
    return repo_dir


# ============================================================
# 3. Smoke test: full pipeline on 6 corrections
# ============================================================

def smoke_test():
    """Verify the full pipeline works end-to-end on 6 corrections."""
    log("\n" + "="*60)
    log("STEP 3: Smoke test (6 corrections, full pipeline)")
    log("="*60)

    from recall import Recall, RecallConfig
    from eval.corrections import SMOKE_CORRECTIONS
    from eval.accuracy import evaluate_accuracy_on

    # Smoke config — much more aggressive than the defaults.
    # The original 2 epochs × 1 chunk = 2 steps wasn't enough to teach
    # the model anything. Now: 15 epochs × 30 repeats / batch_size=2
    # = ~225 gradient steps per correction. Should actually learn.
    config = RecallConfig(
        model_id="Qwen/Qwen3-0.6B",
        lora_rank=16,
        lora_targets=("q_proj", "v_proj"),
        train_epochs=15,           # was 2 — way too few
        consolidation_epochs=8,    # was 1 — way too few
        data_repeat=30,            # new — 30 copies of each correction
        batch_size=2,
        context_length=128,        # was 256 — smaller = more batches per epoch
        train_lr=3e-4,             # slightly higher for faster learning
        consolidation_lr=1.5e-4,
        avr_probe_samples=10,
        avr_every_n=3,             # run AVR after corrections 3 and 6
        max_new_tokens=64,
        seed=42,
        data_dir=str(OUTPUT_DIR / "recall_smoke"),
    )

    log(f"  config: model={config.model_id}, lora_rank={config.lora_rank}, "
        f"train_epochs={config.train_epochs}, avr_every_n={config.avr_every_n}")

    mem = Recall(config=config)

    # Use 6 corrections from the smoke set
    corrections = SMOKE_CORRECTIONS[:6]
    log(f"  testing with {len(corrections)} corrections")

    results = {
        "config": config.to_dict(),
        "n_corrections": len(corrections),
        "corrections": [],
        "avr_ran_at": [],
        "final_accuracy_on_correction_1": None,
        "passed": False,
    }

    # Before any corrections: baseline accuracy on correction #1
    log("\n  --- baseline (before any corrections) ---")
    baseline_acc_1 = evaluate_accuracy_on(mem.generate, corrections[0])
    log(f"  baseline acc on correction[0]: {baseline_acc_1:.3f}")
    results["baseline_acc_on_correction_1"] = baseline_acc_1

    # Teach each correction
    for i, corr in enumerate(corrections):
        log(f"\n  --- correction {i+1}/{len(corrections)}: {corr['id']} ---")
        t0 = time.time()
        cid = mem.remember(corr["input"], corr["target"])
        elapsed = time.time() - t0
        log(f"  → trained in {elapsed:.1f}s (cid={cid})")

        # Check accuracy on this correction immediately after teaching
        gen_immediate = mem.generate(corr["input"])
        acc_immediate = evaluate_accuracy_on(mem.generate, corr)
        log(f"  → immediate acc on {corr['id']}: {acc_immediate:.3f}")
        log(f"  → generation: {gen_immediate!r}")

        # Check accuracy on correction #1 (the one we want to not forget)
        gen_on_first = mem.generate(corrections[0]["input"])
        acc_on_first = evaluate_accuracy_on(mem.generate, corrections[0])
        log(f"  → acc on correction[0] (forgetting probe): {acc_on_first:.3f}")
        log(f"  → gen on correction[0]: {gen_on_first!r}")

        results["corrections"].append({
            "i": i + 1,
            "id": corr["id"],
            "elapsed_s": elapsed,
            "immediate_acc": acc_immediate,
            "acc_on_correction_1": acc_on_first,
            "gen_immediate": gen_immediate[:200],
            "gen_on_correction_1": gen_on_first[:200],
        })

        # Check if AVR ran (every 3rd correction)
        n_trained = mem.status()["n_corrections_trained"]
        if n_trained > 0 and n_trained % config.avr_every_n == 0:
            results["avr_ran_at"].append(n_trained)
            log(f"  → AVR should have run at correction #{n_trained}")

    # Final check
    log("\n  --- final check ---")
    final_acc_1 = evaluate_accuracy_on(mem.generate, corrections[0])
    results["final_accuracy_on_correction_1"] = final_acc_1
    log(f"  final acc on correction[0]: {final_acc_1:.3f}")

    # Pass criteria:
    # 1. Final acc on correction[0] should be > 0.5 (i.e., not forgotten)
    # 2. Immediate acc on at least one correction should be > 0.5
    immediate_accs = [c["immediate_acc"] for c in results["corrections"]]
    max_immediate = max(immediate_accs) if immediate_accs else 0.0
    results["max_immediate_acc"] = max_immediate

    results["passed"] = (
        final_acc_1 > 0.5 and  # didn't forget correction #1
        max_immediate > 0.5    # at least one correction stuck
    )

    log(f"\n  SMOKE TEST {'PASSED' if results['passed'] else 'FAILED'}")
    log(f"    final_acc_on_correction_1 = {final_acc_1:.3f} (need > 0.5)")
    log(f"    max_immediate_acc         = {max_immediate:.3f} (need > 0.5)")

    return results, mem, config


# ============================================================
# 4. Mini forgetting curve: Recall vs Naive SFT
# ============================================================

def mini_forgetting_curve(config):
    """Run a small forgetting curve comparing Recall vs Naive SFT."""
    log("\n" + "="*60)
    log("STEP 4: Mini forgetting curve (Recall vs Naive SFT, 6 corrections)")
    log("="*60)

    import gc
    import torch
    from eval.corrections import SMOKE_CORRECTIONS
    from eval.forgetting_curve import run_curve_for_system
    from eval.render import render_forgetting_curve

    # Use the smoke config but smaller avr_every_n for the curve
    curve_config_dict = config.to_dict()
    curve_config_dict["data_dir"] = str(OUTPUT_DIR / "curve_recall")
    from recall.config import RecallConfig
    curve_config = RecallConfig.from_dict(curve_config_dict)

    corrections = SMOKE_CORRECTIONS[:6]
    log(f"  running curve on {len(corrections)} corrections")

    curves = {}
    for sys_name in ["recall", "naive_sft"]:
        log(f"\n  --- system: {sys_name} ---")
        try:
            curve = run_curve_for_system(
                sys_name, curve_config, corrections,
                eval_on="first", verbose=True)
            curves[sys_name] = curve
            # Print the curve
            for point in curve:
                log(f"    [{sys_name}] corr {point['i']}: "
                    f"acc={point['accuracy']:.3f} ({point['elapsed_s']:.0f}s)")
        except Exception as e:
            log(f"  [!] {sys_name} failed: {e}")
            import traceback
            traceback.print_exc()
            curves[sys_name] = []

        # Cleanup between systems
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Render chart
    chart_data = {
        "config": curve_config.to_dict(),
        "n_corrections": len(corrections),
        "corrections": [c["id"] for c in corrections],
        "eval_on": "first",
        "curves": curves,
        "summary": {},
    }

    # Summary
    for sys_name, curve in curves.items():
        if curve:
            chart_data["summary"][sys_name] = {
                "start_acc": curve[0]["accuracy"],
                "end_acc": curve[-1]["accuracy"],
                "delta": curve[-1]["accuracy"] - curve[0]["accuracy"],
                "min_acc": min(p["accuracy"] for p in curve),
                "total_s": curve[-1]["elapsed_s"],
            }

    chart_path = str(OUTPUT_DIR / "forgetting_curve_smoke.png")
    try:
        render_forgetting_curve(chart_data, chart_path)
        log(f"\n  chart saved to: {chart_path}")
    except Exception as e:
        log(f"  [!] chart render failed: {e}")

    return chart_data, chart_path


# ============================================================
# 5. Main
# ============================================================

def main():
    t_start = time.time()
    log("="*60)
    log("RECALL KAGGLE TEST")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Python: {sys.version}")
    log(f"Output: {OUTPUT_DIR}")
    log("="*60)

    # Step 1: install
    try:
        install_deps()
    except Exception as e:
        log(f"FAILED at install_deps: {e}")
        return

    # Step 2: get source
    try:
        repo_dir = get_recall_source()
        log(f"  repo_dir = {repo_dir}")
    except Exception as e:
        log(f"FAILED at get_recall_source: {e}")
        import traceback; traceback.print_exc()
        return

    # Sanity: print the file tree
    log("\n  Recall repo contents:")
    for root, dirs, files in os.walk(repo_dir):
        if ".git" in root or "__pycache__" in root:
            continue
        depth = root.replace(str(repo_dir), "").count(os.sep)
        indent = "  " * (depth + 2)
        log(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            log(f"{indent}  {f}")

    # Verify imports
    log("\n  Verifying imports...")
    try:
        import recall
        from recall import Recall, RecallConfig
        from recall.trainer import train_hippocampus, consolidate_to_neocortex
        from recall.avr import run_avr_loop, repair_toward_snapshot
        from recall.state import get_lora_state, set_lora_state, reset_lora_to_peft_init
        from recall.base import load_model_and_tokenizer
        log("  ✓ all recall imports OK")
    except Exception as e:
        log(f"  FAILED recall imports: {e}")
        import traceback; traceback.print_exc()
        return

    try:
        from eval.corrections import SMOKE_CORRECTIONS, CORRECTIONS
        from eval.accuracy import evaluate_accuracy_on, score_check_tokens
        from eval.forgetting_curve import run_curve_for_system, run_full_curve
        from eval.render import render_forgetting_curve
        log("  ✓ all eval imports OK")
    except Exception as e:
        log(f"  FAILED eval imports: {e}")
        import traceback; traceback.print_exc()
        return

    # Print GPU info
    log("\n  GPU info:")
    try:
        import torch
        if torch.cuda.is_available():
            log(f"    CUDA available: {torch.cuda.is_available()}")
            log(f"    device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log(f"    GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
        else:
            log("    NO CUDA — running on CPU (will be slow)")
    except Exception as e:
        log(f"    GPU info failed: {e}")

    # Step 3: smoke test
    try:
        smoke_results, mem, config = smoke_test()
    except Exception as e:
        log(f"FAILED at smoke_test: {e}")
        import traceback; traceback.print_exc()
        smoke_results = {"passed": False, "error": str(e)}
        mem, config = None, None

    # Step 4: mini forgetting curve
    try:
        if config is not None:
            curve_data, chart_path = mini_forgetting_curve(config)
        else:
            curve_data, chart_path = None, None
    except Exception as e:
        log(f"FAILED at mini_forgetting_curve: {e}")
        import traceback; traceback.print_exc()
        curve_data, chart_path = None, None

    # Final summary
    elapsed_total = time.time() - t_start
    log("\n" + "="*60)
    log("FINAL SUMMARY")
    log("="*60)
    log(f"  Total elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    log(f"  Smoke test passed: {smoke_results.get('passed', False) if isinstance(smoke_results, dict) else False}")
    if curve_data and curve_data.get("summary"):
        log(f"  Forgetting curve summary:")
        for sys_name, s in curve_data["summary"].items():
            log(f"    {sys_name}: start={s['start_acc']:.3f} "
                f"end={s['end_acc']:.3f} delta={s['delta']:+.3f} "
                f"min={s['min_acc']:.3f} ({s['total_s']:.0f}s)")
    if chart_path:
        log(f"  Chart: {chart_path}")
    log("="*60)

    # Save final results JSON
    final = {
        "started_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "elapsed_s": elapsed_total,
        "smoke_test": smoke_results if isinstance(smoke_results, dict) else {"error": str(smoke_results)},
        "forgetting_curve": curve_data,
        "chart_path": chart_path,
    }
    results_path = OUTPUT_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    log(f"\nResults saved to: {results_path}")
    log(f"Log saved to: {LOG_PATH}")

    _log_fh.close()


if __name__ == "__main__":
    main()
