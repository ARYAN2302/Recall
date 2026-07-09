"""
eval.render — chart rendering for the forgetting curve.

Produces the PNG that goes in the README and on X.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font fallback — works on Linux/Mac/Modal
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


# Color palette — sharp contrast, color-blind friendly
COLORS = {
    "recall":      "#2563eb",  # blue
    "naive_sft":   "#dc2626",  # red
    "replay_sft":  "#f59e0b",  # amber
}
LABELS = {
    "recall":      "Recall (AVR + two-stream)",
    "naive_sft":   "Naive SFT (no protection)",
    "replay_sft":  "Replay SFT (10% buffer)",
}


def render_forgetting_curve(
    curve_data: Dict,
    output_path: str = "forgetting_curve.png",
    title: Optional[str] = None,
) -> str:
    """Render the forgetting curve chart.

    Args:
        curve_data: output of eval.forgetting_curve.run_full_curve
        output_path: where to save the PNG

    Returns:
        Path to the saved chart.
    """
    curves = curve_data.get("curves", {})
    n_corr = curve_data.get("n_corrections", 50)
    eval_on = curve_data.get("eval_on", "first")

    fig, ax = plt.subplots(
        figsize=(10, 6), dpi=150, constrained_layout=True)

    # Plot each system
    for sys_name, curve in curves.items():
        if not curve:
            continue
        xs = [p["i"] for p in curve]
        ys = [p["accuracy"] for p in curve]
        color = COLORS.get(sys_name, "#666666")
        label = LABELS.get(sys_name, sys_name)
        ax.plot(xs, ys, color=color, linewidth=2.5, label=label,
                marker='o', markersize=4, alpha=0.9)

    # Drift threshold line — visualize "what counts as forgetting"
    ax.axhline(y=0.85, color='#94a3b8', linestyle='--', linewidth=1,
               alpha=0.7, label='_nolegend_')
    ax.text(n_corr * 0.02, 0.855, 'forgetting threshold (0.85)',
            color='#64748b', fontsize=9, va='bottom')

    # Labels
    eval_label = "correction #1" if eval_on == "first" else "all prior corrections"
    ax.set_xlabel(f"Corrections taught (in sequence)", fontsize=12)
    ax.set_ylabel(f"Accuracy on {eval_label}", fontsize=12)
    ax.set_title(
        title or "Recall vs baselines: forgetting curve over 50 corrections",
        fontsize=14, fontweight='bold', pad=12)

    ax.set_xlim(0, n_corr + 1)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xticks(range(0, n_corr + 1, max(1, n_corr // 10)))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', frameon=True, fontsize=11,
              fancybox=True, shadow=False)

    # Subtitle with summary
    summary = curve_data.get("summary", {})
    subtitle_parts = []
    for sys_name in ["recall", "naive_sft", "replay_sft"]:
        if sys_name in summary and summary[sys_name]:
            s = summary[sys_name]
            subtitle_parts.append(
                f"{LABELS[sys_name].split(' ')[0]}: {s['start_acc']:.2f}→{s['end_acc']:.2f}")
    if subtitle_parts:
        fig.text(0.5, 0.01,
                 "  •  ".join(subtitle_parts),
                 ha='center', fontsize=10, color='#475569',
                 style='italic')

    # Save
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor='white')
    plt.close(fig)
    return output_path


def render_summary_table(curve_data: Dict) -> str:
    """Render a markdown summary table of the curves."""
    summary = curve_data.get("summary", {})
    n = curve_data.get("n_corrections", 50)

    lines = [
        f"| System | Start acc | End acc (after {n}) | Δ | Min acc |",
        f"|---|---|---|---|---|",
    ]
    for sys_name in ["recall", "naive_sft", "replay_sft"]:
        s = summary.get(sys_name)
        if not s:
            continue
        lines.append(
            f"| {LABELS.get(sys_name, sys_name)} | "
            f"{s['start_acc']:.3f} | {s['end_acc']:.3f} | "
            f"{s['delta']:+.3f} | {s['min_acc']:.3f} |"
        )
    return "\n".join(lines)
