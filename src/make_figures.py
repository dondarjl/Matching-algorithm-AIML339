"""
Regenerates the two report figures from the numbers in
results/compare_three_models_real.txt (the final `--real` run).

Values are transcribed from that log (pooled AUC at full precision, per-participant
means and bootstrap intervals at the precision printed there). If you re-run
compare_three_models.py with different settings, update the values below.

Usage (from src/):
    python make_figures.py
Outputs: ../results/auc_comparison.png, ../results/differences_forest.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

# ---- Fig. 1: pooled vs per-participant AUC ----------------------------------
models = ["Baseline", "Structured", "Fusion"]
pooled = [0.540963, 0.551382, 0.567470]
per_participant = [0.5532, 0.5507, 0.5719]


def fig1():
    fig, ax = plt.subplots(figsize=(1273 / 300, 848 / 300), dpi=300)
    x = np.arange(3)
    w = 0.35
    b1 = ax.bar(x - w / 2, pooled, w, color="#4C72B0", label="Pooled AUC")
    b2 = ax.bar(x + w / 2, per_participant, w, color="#DD8452", label="Per-participant mean AUC")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance (AUC=0.5)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.001,
                    f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_ylim(0.45, 0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("AUC", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    order = [l.index("Chance (AUC=0.5)"), l.index("Pooled AUC"), l.index("Per-participant mean AUC")]
    ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=5.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "auc_comparison.png", dpi=300)
    plt.close(fig)


# ---- Fig. 2: differences with 95% cluster-bootstrap intervals ---------------
# label, point estimate, ci_lo, ci_hi, group
ROWS = [
    ("F – B (primary)", 0.026507, 0.0032, 0.0518, "Pooled ΔAUC"),
    ("S – B", 0.010419, -0.0257, 0.0491, "Pooled ΔAUC"),
    ("F – S", 0.016088, -0.0014, 0.0328, "Pooled ΔAUC"),
    ("F – B (primary)", 0.005608, -0.0095, 0.0180, "Pooled ΔPR-AUC"),
    ("S – B", -0.006154, -0.0286, 0.0173, "Pooled ΔPR-AUC"),
    ("F – S", 0.011762, -0.0044, 0.0250, "Pooled ΔPR-AUC"),
    ("F – B", 0.0187, -0.0019, 0.0412, "Per-participant ΔAUC (exploratory)"),
    ("F – S", 0.0213, 0.0023, 0.0413, "Per-participant ΔAUC (exploratory)"),
]
COLORS = {"Pooled ΔAUC": "#4C72B0", "Pooled ΔPR-AUC": "#55A868",
          "Per-participant ΔAUC (exploratory)": "#DD8452"}


def fig2():
    fig, ax = plt.subplots(figsize=(1273 / 300, 1133 / 300), dpi=300)
    ys, y, prev = [], 0.0, None
    for r in ROWS:
        if prev is not None and r[4] != prev:
            y -= 0.8
        ys.append(y)
        y -= 1
        prev = r[4]
    for (lab, p, lo, hi, g), yy in zip(ROWS, ys):
        ax.plot([lo, hi], [yy, yy], color=COLORS[g], lw=1.4, solid_capstyle="butt")
        ax.plot(p, yy, "o", color=COLORS[g], ms=3.2)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=6)
    seen = set()
    for r, yy in zip(ROWS, ys):
        if r[4] not in seen:
            seen.add(r[4])
            ax.text(-0.0375, yy + 0.62, r[4], fontsize=6, fontweight="bold",
                    color=COLORS[r[4]], ha="left", va="center")
    ax.set_xlim(-0.04, 0.06)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 1.0)
    ax.set_xlabel("Difference (95% cluster-bootstrap interval)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "differences_forest.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print(f"Wrote figures to {OUT}")