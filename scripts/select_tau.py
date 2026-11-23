"""WS-B Gate B: pick the LA τ from out-of-fold predictions of plan p12_tau_screen (reports/plan/eacl_srw_run_plan.md §3.2).

    python scripts/select_tau.py [--dir data/server/outputs/p12_srw]

Reads <dir>/tau{025,050,075,100}_f{0,1,2}_s42/val_predictions.csv (val = held-out group fold of train).
Per τ: 4-class macro-F1 (selection metric) and 3-class macro-F1 (no text) on the concatenated OOF predictions,
the per-fold mean ± SD, per-class F1, and a paired bootstrap of (τ − τ=1.0) over OOF ids (B = 10,000, same ids).
Gate B rule: a τ replaces 1.0 only if its OOF macro-F1 (4) beats τ=1.0 by more than one fold-SD.
Writes <dir>/tau_screen_summary.{csv,json}. Never reads the public or private test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]
NO_TEXT = [l for l in LABELS if l != "text-sarcasm"]
TAUS = {"025": 0.25, "050": 0.5, "075": 0.75, "100": 1.0}


def mf1(y, p, labels):
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/server/outputs/p12_srw"))
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    oof = {}
    for key in TAUS:
        parts = []
        for k in range(3):
            f = args.dir / f"tau{key}_f{k}_s42" / "val_predictions.csv"
            if not f.exists():
                raise SystemExit(f"missing {f}; the screen is not finished")
            parts.append(pd.read_csv(f, usecols=["id", "label", "pred"], dtype={"id": str}).assign(fold=k))
        oof[key] = pd.concat(parts).set_index("id").sort_index()
    ref = oof["100"]
    for key, d in oof.items():
        assert d.index.equals(ref.index) and (d.label == ref.label).all(), f"τ {key}: OOF ids/labels differ"
    y = ref.label.to_numpy()

    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(y), size=(args.n_boot, len(y)))
    rows = []
    for key, tau in TAUS.items():
        d = oof[key]
        p = d.pred.to_numpy()
        folds = [mf1(d.label[d.fold == k], d.pred[d.fold == k], LABELS) for k in range(3)]
        row = {"tau": tau, "n_oof": len(d), "macro_f1_4": mf1(y, p, LABELS), "macro_f1_3": mf1(y, p, NO_TEXT),
               "accuracy": float((y == p).mean()), "fold_mean": float(np.mean(folds)),
               "fold_sd": float(np.std(folds, ddof=1)), **{f"fold{k}": v for k, v in enumerate(folds)},
               **{f"f1_{l.split('-')[0]}": f for l, f in zip(LABELS, f1_score(y, p, labels=LABELS, average=None,
                                                                              zero_division=0))}}
        if key != "100":
            pr = ref.pred.to_numpy()
            diffs = np.array([mf1(y[b], p[b], LABELS) - mf1(y[b], pr[b], LABELS) for b in boot[:args.n_boot]])
            row.update(diff_vs_1=row["macro_f1_4"] - mf1(y, pr, LABELS),
                       diff_ci_lo=float(np.quantile(diffs, .025)), diff_ci_hi=float(np.quantile(diffs, .975)))
        rows.append(row)
    tab = pd.DataFrame(rows)
    sd_ref = float(tab.loc[tab.tau == 1.0, "fold_sd"].iloc[0])
    base = float(tab.loc[tab.tau == 1.0, "macro_f1_4"].iloc[0])
    better = tab[(tab.tau != 1.0) & (tab.macro_f1_4 - base > sd_ref)]
    choice = float(better.sort_values("macro_f1_4").tau.iloc[-1]) if len(better) else 1.0
    tab.to_csv(args.dir / "tau_screen_summary.csv", index=False)
    (args.dir / "tau_screen_summary.json").write_text(json.dumps(
        {"rule": "replace τ=1.0 only if OOF macro-F1(4) − τ1.0 > fold-SD of τ1.0",
         "fold_sd_tau1": sd_ref, "selected_tau": choice, "gate_b_pass": choice != 1.0,
         "table": tab.to_dict(orient="records")}, indent=2))
    print(tab.round(4).to_string(index=False))
    print(f"\nGate B: selected τ = {choice} ({'retrain on full train, 3 seeds' if choice != 1.0 else 'keep τ = 1.0, stop'})")


if __name__ == "__main__":
    main()
