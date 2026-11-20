"""Draw docs/media/before_after.png (+ .svg) from the published benchmark results.

The numbers are read from results/baseline/summary.json (original Gemma 4 E2B + MTP) and
results/trimmed/summary.json (en50v + MTP); nothing is typed in by hand.

Usage:  python docs/make_chart.py [--out docs/media] [--web path/for/web-copy.png]
Needs matplotlib (not in requirements.txt; only used for this chart).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
ORIGINAL, TRIMMED = "#a3a29c", "#2a78d6"


def load(path, name):
    rows = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return next(r for r in rows if r["model"] == name)


def panel(ax, title, values, fmt, change, ymax):
    xs = [0, 1]
    bars = ax.bar(xs, values, width=0.56, color=[ORIGINAL, TRIMMED], edgecolor=SURFACE, linewidth=2, zorder=3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, fmt(v), ha="center", va="bottom",
                fontsize=13, color=INK, fontweight="bold")
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=26)
    ax.text(0, 1.03, change, transform=ax.transAxes, fontsize=11, color=INK2, va="bottom")
    ax.set_xticks(xs, ["Original", "Trimmed\n(en50v)"], fontsize=10.5, color=INK2)
    ax.set_ylim(0, ymax)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_facecolor(SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"))
    ap.add_argument("--web", help="also write a smaller PNG here (for a website)")
    a = ap.parse_args()

    base = load("results/baseline/summary.json", "gemma-4-e2b+mtp")
    trim = load("results/trimmed/summary.json", "gemma-4-e2b-en50v+mtp")
    size = [base["file_gb"], trim["file_gb"]]
    secs = [base["avg_task_seconds"], trim["avg_task_seconds"]]
    acc = [100 * base["full_accuracy"], 100 * trim["full_accuracy"]]
    n_kept = len(json.loads((ROOT / "keep_lists" / "keep_ids_50v.json").read_text()))
    pct = lambda new, old: f"{100 * (new / old - 1):+.0f}%".replace("-", "−")  # noqa: E731

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), facecolor=SURFACE)
    panel(axes[0], "File size (Q4_0)", size, lambda v: f"{v:.2f} GB", f"{pct(size[1], size[0])} smaller", 3.3)
    panel(axes[1], "Time per task (with MTP draft)", secs, lambda v: f"{v:.2f} s",
          f"{pct(secs[1], secs[0])} per task", 2.35)
    panel(axes[2], "Tool-calling score (30 tasks)", acc, lambda v: f"{v:.0f}%",
          "same score" if acc[0] == acc[1] else f"{acc[1] - acc[0]:+.0f} points", 117)
    fig.suptitle("Gemma 4 E2B: original vs English-only vocabulary trim", x=0.02, y=0.98, ha="left",
                 fontsize=15, color=INK, fontweight="bold")
    fig.text(0.02, 0.015, "CPU only: Intel i5-11300H, 4 threads, llama.cpp b12360, temperature 0. "
             f"Vocabulary 262,144 → {n_kept:,} tokens. Source: results/*/summary.json",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=3)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "before_after.png", dpi=160, facecolor=SURFACE)
    fig.savefig(out / "before_after.svg", facecolor=SURFACE)
    if a.web:
        Path(a.web).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.web, dpi=110, facecolor=SURFACE)
    print("size", size, "s/task", secs, "accuracy", acc)


if __name__ == "__main__":
    main()
