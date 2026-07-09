"""
recall.cli — command-line interface.

    recall status                          # show backend state
    recall eval forgetting-curve --n 50    # run the viral benchmark
    recall demo                            # run the 5-line quickstart
    recall deploy                          # deploy to Modal (optional)
"""
from __future__ import annotations
import argparse
import json
import sys
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        prog="recall",
        description="Agent memory that actually learns.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # status
    p_status = sub.add_parser("status", help="Show backend state")
    p_status.add_argument("--data-dir", default=None)

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluations")
    eval_sub = p_eval.add_subparsers(dest="eval_cmd", required=True)
    p_curve = eval_sub.add_parser("forgetting-curve",
                                  help="Run the forgetting-curve benchmark")
    p_curve.add_argument("--n", type=int, default=50,
                         help="number of corrections (default 50)")
    p_curve.add_argument("--output", default="forgetting_curve.png")
    p_curve.add_argument("--no-baselines", action="store_true",
                         help="skip naive-SFT baseline (faster)")
    p_curve.add_argument("--data-dir", default=None)

    # demo
    p_demo = sub.add_parser("demo", help="Run the 5-line quickstart")
    p_demo.add_argument("--data-dir", default=None)

    # deploy
    p_dep = sub.add_parser("deploy", help="Deploy to Modal (optional)")

    args = parser.parse_args()

    if args.cmd == "status":
        _cmd_status(args)
    elif args.cmd == "eval":
        _cmd_eval(args)
    elif args.cmd == "demo":
        _cmd_demo(args)
    elif args.cmd == "deploy":
        _cmd_deploy(args)


def _cmd_status(args):
    from recall import Recall, RecallConfig
    config = RecallConfig(data_dir=args.data_dir) if args.data_dir else None
    mem = Recall(config=config) if config else Recall()
    status = mem.status()
    print(json.dumps(status, indent=2, default=str))


def _cmd_eval(args):
    from recall import RecallConfig
    from eval.forgetting_curve import run_full_curve
    from eval.render import render_forgetting_curve

    config = RecallConfig(data_dir=args.data_dir) if args.data_dir else RecallConfig()
    print(f"Running forgetting-curve benchmark (n={args.n})...", flush=True)
    results = run_full_curve(
        config,
        n_corrections=args.n,
        include_baselines=not args.no_baselines,
    )
    print(f"\nResults:", flush=True)
    for system, curve in results["curves"].items():
        if curve:
            start_acc = curve[0][1]
            end_acc = curve[-1][1]
            print(f"  {system:20s} start={start_acc:.3f} end={end_acc:.3f} "
                  f"delta={end_acc - start_acc:+.3f}", flush=True)

    chart_path = render_forgetting_curve(results, args.output)
    print(f"\nChart saved to: {chart_path}", flush=True)


def _cmd_demo(args):
    """Run the 5-line quickstart inline."""
    from recall import Recall

    print("=== Recall demo ===\n", flush=True)
    mem = Recall(data_dir=args.data_dir)

    print("1. Before any corrections:", flush=True)
    print(f"   prompt: 'write a function to sort a list'", flush=True)
    out1 = mem.generate("write a function to sort a list")
    print(f"   output: {out1!r}\n", flush=True)

    print("2. Teaching correction: 'always use type hints and docstrings'",
          flush=True)
    mem.remember(
        "write a function to sort a list",
        "def sort_list(lst: list) -> list:\n    \"\"\"Sort a list in place.\"\"\"\n    return sorted(lst)",
    )

    print("3. After correction:", flush=True)
    out2 = mem.generate("write a function to sort a list")
    print(f"   output: {out2!r}\n", flush=True)

    print("4. Status:", flush=True)
    print(f"   {mem.status()}\n", flush=True)


def _cmd_deploy(args):
    """Deploy the Modal app.

    The actual deploy is handled by the Modal CLI — this command just
    prints the instructions. We don't shell out from inside Python so
    users can see Modal's deployment logs in their own terminal.
    """
    try:
        import modal  # noqa: F401
    except ImportError:
        print("Error: modal not installed. Run `pip install modal` first.")
        sys.exit(1)
    print("To deploy Recall to Modal, run this in your shell:\n")
    print("  modal deploy recall/modal_app.py\n")
    print("This uploads the app to Modal and creates the Functions.")
    print("After deploy, set MODAL_APP_NAME=recall-memory in your env")
    print("and pass modal=True to Recall():")
    print("    from recall import Recall")
    print("    mem = Recall(modal=True)")


if __name__ == "__main__":
    main()
