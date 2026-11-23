"""Full private-test comparison: caption-only + LA vs cross-attention + LA (and cross-attention + CE for reference).

Run from the repo root: python data/server/outputs/p11_controlled/analyze_caption_vs_xattn.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

O = Path("data/server/outputs")
OUT = O / "p11_controlled"
RAW = Path("data/server/raw")
LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]
SHORT = ["not", "text", "image", "multi"]
SEEDS = [42, 1, 2]
MODELS = {"caption_la": "p11_caption_la_s{}", "xattn_la": "p11_controlled/ca_la_s{}", "xattn_ce": "p11_controlled/ca_ce_s{}"}
B = 10000
rng = np.random.default_rng(20261122)


def metrics(y, p):
    f1s = f1_score(y, p, labels=LABELS, average=None, zero_division=0)
    pr, rc, _, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    d = {"macro_f1_4": float(np.mean(f1s)), "macro_f1_3": float(np.mean(f1s[[0, 2, 3]])),
         "accuracy": float(accuracy_score(y, p)),
         "weighted_f1": float(f1_score(y, p, labels=LABELS, average="weighted", zero_division=0))}
    for i, s in enumerate(SHORT):
        d[f"{s}_precision"], d[f"{s}_recall"], d[f"{s}_f1"] = float(pr[i]), float(rc[i]), float(f1s[i])
        d[f"{s}_correct"] = int(((y == LABELS[i]) & (p == LABELS[i])).sum())
        d[f"{s}_predicted"] = int((p == LABELS[i]).sum())
    return d


preds, rows = {}, []
for m, pat in MODELS.items():
    for s in SEEDS:
        run = O / pat.format(s)
        df = pd.read_csv(run / "test_predictions.csv", dtype={"id": str}).set_index("id")
        preds[(m, s)] = df
        hist = json.loads((run / "history.json").read_text())
        rows.append({"model": m, "seed": s, "run": str(run), "commit": json.loads((run / "config.json").read_text()).get("commit"),
                     "epochs_run": len(hist), **metrics(df["label"].to_numpy(), df["pred"].to_numpy())})
table = pd.DataFrame(rows)

# pairing checks: identical ids and labels across all nine runs
ids = preds[("xattn_la", 42)].index
y = preds[("xattn_la", 42)]["label"]
for k, v in preds.items():
    assert v.index.is_unique and set(v.index) == set(ids), k
    assert (v.loc[ids, "label"] == y).all(), k
y = y.to_numpy()
counts = pd.Series(y).value_counts().reindex(LABELS).to_dict()

li = {c: i for i, c in enumerate(LABELS)}
yi = np.array([li[c] for c in y])
Pi = {k: np.array([li[c] for c in v.loc[ids, "pred"]]) for k, v in preds.items()}


def mf1(yt, yp, classes=(0, 1, 2, 3)):
    f = []
    for c in classes:
        tp = np.sum((yt == c) & (yp == c)); fp = np.sum((yt != c) & (yp == c)); fn = np.sum((yt == c) & (yp != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def acc(yt, yp, classes=None):
    return float(np.mean(yt == yp))


def diff_fn(a, b, fn, **kw):
    return lambda idx: np.mean([fn(yi[idx], Pi[(a, s)][idx], **kw) - fn(yi[idx], Pi[(b, s)][idx], **kw) for s in SEEDS])


comparisons = {"xattn_la - caption_la": ("xattn_la", "caption_la"), "xattn_ce - caption_la": ("xattn_ce", "caption_la")}
stats = {}
n = len(yi)
boot_idx = [rng.integers(0, n, n) for _ in range(B)]
for name, (a, b) in comparisons.items():
    for mname, fn, kw in (("macro_f1_4", mf1, {}), ("macro_f1_3", mf1, {"classes": (0, 2, 3)}), ("accuracy", acc, {})):
        f = diff_fn(a, b, fn, **kw)
        obs = f(np.arange(n))
        boot = np.array([f(ix) for ix in boot_idx])
        per_seed = {s: fn(yi, Pi[(a, s)], **kw) - fn(yi, Pi[(b, s)], **kw) for s in SEEDS}
        stats[f"{name} | {mname}"] = {"per_seed": per_seed, "mean": obs, "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                                      "share_le_0": float(np.mean(boot <= 0))}

# flips: xattn_la vs caption_la by true class
flip_rows = []
for s in SEEDS:
    cap_ok = Pi[("caption_la", s)] == yi; xa_ok = Pi[("xattn_la", s)] == yi
    for c, lab in enumerate(LABELS):
        mask = yi == c
        flip_rows.append({"seed": s, "true": lab, "n": int(mask.sum()),
                          "xattn right, caption wrong": int((xa_ok & ~cap_ok & mask).sum()),
                          "caption right, xattn wrong": int((cap_ok & ~xa_ok & mask).sum())})
flips = pd.DataFrame(flip_rows)

# how often caption-only predicts each class, and image-sarcasm predictions without seeing the image
pred_mix = {m: {SHORT[c]: float(np.mean([np.mean(Pi[(m, s)] == c) for s in SEEDS])) for c in range(4)} for m in MODELS}

table.to_csv(OUT / "caption_vs_xattn_per_seed.csv", index=False)
flips.to_csv(OUT / "caption_vs_xattn_flips_by_class.csv", index=False)
(OUT / "caption_vs_xattn_stats.json").write_text(json.dumps(
    {"n_test": n, "class_counts": counts, "bootstrap_B": B, "paired": stats, "prediction_mix": pred_mix}, indent=2))

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
cols = ["model", "seed", "epochs_run", "macro_f1_4", "macro_f1_3", "accuracy", "weighted_f1",
        "not_f1", "multi_f1", "image_precision", "image_recall", "image_f1", "image_correct", "image_predicted",
        "text_correct", "text_predicted"]
print(table[cols].round(4).to_string(index=False))
print(table.groupby("model", sort=False)[cols[3:]].agg(["mean", "std"]).round(4).T.to_string())
print(flips.groupby("true", sort=False)[["n", "xattn right, caption wrong", "caption right, xattn wrong"]].sum())
print(json.dumps(stats, indent=1)); print(json.dumps(pred_mix, indent=1)); print(counts)
