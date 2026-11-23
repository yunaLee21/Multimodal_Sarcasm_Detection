"""WS-B final: τ=0.75 vs τ=1.0 on the private test (plan p12_tau_final). Scored once; see _private_test_log.md.

    python scripts/compare_tau_final.py [--dir data/server/outputs/p12_srw] [--p11 data/server/outputs/p11_controlled]

Per seed + mean ± SD macro-F1 (4 / 3 class), per-class F1, paired bootstrap of the seed-averaged
macro-F1 difference (B = 10,000, same resampled ids), and a check that final_tau100_s* reproduces p11 ca_la_s*.
Writes <dir>/tau_final_comparison.json.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]
NO_TEXT = [l for l in LABELS if l != "text-sarcasm"]
SEEDS = [42, 1, 2]
PROB = [f"prob_{l}" for l in LABELS]

def mf1(y, p, labels=LABELS):
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)

def load(f):
    return pd.read_csv(f, usecols=["id", "label", "pred", *PROB], dtype={"id": str}).set_index("id").sort_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/server/outputs/p12_srw"))
    ap.add_argument("--p11", type=Path, default=Path("data/server/outputs/p11_controlled"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    runs = {(t, s): load(a.dir / f"final_tau{t}_s{s}" / f"{a.split}_predictions.csv") for t in ["075", "100"] for s in SEEDS}
    ref = runs[("100", 42)]
    for k, d in runs.items():
        assert d.index.equals(ref.index) and (d.label == ref.label).all(), k
    y = ref.label.to_numpy()
    out = {"split": a.split, "n": len(y), "per_seed": [], "reproduce_p11": []}
    for s in SEEDS:
        p11 = load(a.p11 / f"ca_la_s{s}" / f"{a.split}_predictions.csv").reindex(ref.index)
        d = runs[("100", s)]
        out["reproduce_p11"].append({"seed": s, "pred_mismatch": int((p11.pred != d.pred).sum()),
                                     "max_abs_dprob": float(np.abs(p11[PROB].to_numpy() - d[PROB].to_numpy()).max())})
    for (t, s), d in runs.items():
        p = d.pred.to_numpy()
        out["per_seed"].append({"tau": t, "seed": s, "macro_f1_4": mf1(y, p), "macro_f1_3": mf1(y, p, NO_TEXT),
                                "accuracy": float((y == p).mean()),
                                **{f"f1_{l.split('-')[0]}": v for l, v in zip(LABELS, f1_score(y, p, labels=LABELS, average=None, zero_division=0))}})
    ps = pd.DataFrame(out["per_seed"])
    agg = ps.drop(columns="seed").groupby("tau").agg(["mean", "std"])
    out["summary"] = {t: {c: [float(agg.loc[t, (c, "mean")]), float(agg.loc[t, (c, "std")])] for c in ps.columns if c not in ("tau", "seed")} for t in agg.index}
    # paired bootstrap: mean over seeds of macro-F1 on the same resampled ids
    rng = np.random.default_rng(0)
    P = {t: [runs[(t, s)].pred.to_numpy() for s in SEEDS] for t in ["075", "100"]}
    def stat(idx, labels):
        return np.mean([mf1(y[idx], p[idx], labels) for p in P["075"]]) - np.mean([mf1(y[idx], p[idx], labels) for p in P["100"]])
    full = np.arange(len(y))
    for name, labels in [("diff_4", LABELS), ("diff_3", NO_TEXT)]:
        boots = np.array([stat(rng.integers(0, len(y), len(y)), labels) for _ in range(a.n_boot)])
        out[name] = {"point": float(stat(full, labels)), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                     "p_le_0": float((boots <= 0).mean())}
    (a.dir / f"tau_final_comparison_{a.split}.json").write_text(json.dumps(out, indent=2))
    print(ps.round(4).to_string(index=False)); print(agg.round(4).T.to_string())
    print(json.dumps({k: out[k] for k in ["reproduce_p11", "diff_4", "diff_3"]}, indent=1))

if __name__ == "__main__":
    main()
