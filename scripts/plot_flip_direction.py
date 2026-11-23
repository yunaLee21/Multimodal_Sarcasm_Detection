"""Figure 1 (WS-A): flip direction by gold class under image swap / mask, private test.

    python scripts/plot_flip_direction.py --dir data/server/outputs/p12_srw/image_dependence

Reads flip_matrix.csv, arm_condition.csv and one probs.npz (for class counts). Writes fig_flip_direction.{pdf,png}
and fig_flip_direction_table.csv (the same numbers as the figure).
A flip rate = flips / (gold-class size × model seeds × condition draws).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARMS = {"ca_la": "X-attn + LA", "ca_ce": "X-attn + CE"}
FAMILIES = {"swap_random": "Random image swap", "mask_mean": "Mean-image mask"}
GOLD = ["not", "image", "multi"]  # text (n = 7) omitted, reported in the table
KINDS = [("sarcastic → not", "#2a78d6"), ("not → sarcastic", "#eb6834"), ("between sarcastic classes", "#1baf7a")]


def kind(a, b):
    if a != "not" and b == "not":
        return KINDS[0][0]
    if a == "not" and b != "not":
        return KINDS[1][0]
    return KINDS[2][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/server/outputs/p12_srw/image_dependence"))
    args = ap.parse_args()
    fm = pd.read_csv(args.dir / "flip_matrix.csv")
    arm = pd.read_csv(args.dir / "arm_condition.csv")
    labels = np.load(next(args.dir.rglob("probs.npz")))["labels"]  # every run shares the test order
    n_gold = {g: int((labels == k).sum()) for k, g in enumerate(["not", "text", "image", "multi"])}

    fm = fm[(fm.set == "private_test") & (fm["from"] != fm["to"])].copy()  # flips only; the csv also holds unchanged
    fm["kind"] = [kind(a, b) for a, b in zip(fm["from"], fm["to"])]
    rows = []
    for a in ARMS:
        for f in FAMILIES:
            draws = int(arm[(arm.set == "private_test") & (arm.arm == a) & (arm.family == f)].n_condition_draws.iloc[0])
            for g in ["not", "text", "image", "multi"]:
                sub = fm[(fm.arm == a) & (fm.family == f) & (fm.gold == g)]
                for k, _ in KINDS:
                    rows.append({"arm": a, "family": f, "gold": g, "n_gold": n_gold[g], "draws": draws, "kind": k,
                                 "flips": int(sub[sub.kind == k]["count"].sum()),
                                 "rate": sub[sub.kind == k]["count"].sum() / (n_gold[g] * draws)})
    tab = pd.DataFrame(rows)
    tab.to_csv(args.dir / "fig_flip_direction_table.csv", index=False)

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(6.3, 2.4), sharey=True, sharex=True)
    ylabels = [f"{ARMS[a]} · {g} (n={n_gold[g]})" for a in ARMS for g in GOLD]
    y = np.arange(len(ylabels))[::-1]
    for ax, f in zip(axes, FAMILIES):
        left = np.zeros(len(ylabels))
        for k, color in KINDS:
            vals = np.array([tab[(tab.arm == a) & (tab.family == f) & (tab.gold == g) & (tab.kind == k)].rate.iloc[0]
                             for a in ARMS for g in GOLD])
            ax.barh(y, vals, left=left, height=0.62, color=color, edgecolor="#fcfcfb", linewidth=1, label=k)
            left += vals
        for yi, tot in zip(y, left):
            ax.text(tot + 0.01, yi, f"{tot:.2f}", va="center", fontsize=7, color="#0b0b0b")
        ax.set_title(FAMILIES[f], fontsize=8.5, color="#0b0b0b")
        ax.set_xlabel("Flip rate")
        ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, ylabels)
    axes[0].set_xlim(0, 0.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.55, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(args.dir / f"fig_flip_direction.{ext}", dpi=200, bbox_inches="tight")
    print(tab[tab.gold.isin(GOLD)].pivot_table(index=["family", "arm", "gold"], columns="kind", values="rate").round(3))


if __name__ == "__main__":
    main()
