"""Paired CE vs LA analysis for the Phase 11 controlled comparison.

Usage: python analyze.py  (from the repo root; reads data/server/outputs/p11_controlled)
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

OUT = Path("data/server/outputs/p11_controlled")
RAW = Path("data/server/raw")
LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]
THREE = ["not-sarcasm", "image-sarcasm", "multi-sarcasm"]
SEEDS = [42, 1, 2]
ARMS = ["ce", "la"]
B = 10000
rng = np.random.default_rng(20261122)


def metrics(y, p):
    f1s = f1_score(y, p, labels=LABELS, average=None, zero_division=0)
    pr, rc, _, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    d = {
        "macro_f1_4": float(np.mean(f1s)),
        "macro_f1_3": float(np.mean([f1s[LABELS.index(c)] for c in THREE])),
        "accuracy": float(accuracy_score(y, p)),
        "weighted_f1": float(f1_score(y, p, labels=LABELS, average="weighted", zero_division=0)),
    }
    for c, short in (("text-sarcasm", "text"), ("image-sarcasm", "image"), ("not-sarcasm", "not"), ("multi-sarcasm", "multi")):
        i = LABELS.index(c)
        d[f"{short}_precision"], d[f"{short}_recall"], d[f"{short}_f1"] = float(pr[i]), float(rc[i]), float(f1s[i])
        d[f"{short}_correct"] = int(((y == c) & (p == c)).sum())
    return d


def selected_epoch(run):
    log = (run / "run.log") if (run / "run.log").exists() else run / "train_eval.log"
    text = log.read_text(encoding="utf-8")
    rows = re.findall(r"epoch (\d+) \| train loss ([\d.]+) \| val chọn checkpoint \([^)]*\) ([\d.]+)", text)
    best, best_ep = -1, None
    for ep, _, score in rows:  # trainer keeps strictly better scores
        if float(score) > best:
            best, best_ep = float(score), int(ep)
    return best_ep, len(rows), best


# ---------- load predictions ----------
preds, rows = {}, []
for seed in SEEDS:
    for arm in ARMS:
        run = OUT / f"ca_{arm}_s{seed}"
        df = pd.read_csv(run / "test_predictions.csv", dtype={"id": str})
        preds[(arm, seed)] = df.set_index("id")
        ep, n_ep, val_sel = selected_epoch(run)
        cfg = json.loads((run / "config.json").read_text())
        m = metrics(df["label"].to_numpy(), df["pred"].to_numpy())
        tm = json.loads((run / "test_metrics.json").read_text())
        assert abs(tm["macro_f1"] - m["macro_f1_4"]) < 1e-6, (run, tm["macro_f1"], m["macro_f1_4"])
        rows.append({"arm": arm, "seed": seed, "run": run.name, "commit": cfg["commit"], "selected_epoch": ep,
                     "epochs_run": n_ep, "val_macro_f1_selected": val_sel, "n_test": len(df), **m})
table = pd.DataFrame(rows)
table.to_csv(OUT / "per_seed_metrics.csv", index=False)

# ---------- pairing checks ----------
checks = []
for seed in SEEDS:
    a, b = preds[("ce", seed)], preds[("la", seed)]
    assert a.index.is_unique and b.index.is_unique
    assert set(a.index) == set(b.index), f"seed {seed}: id sets differ"
    b = b.loc[a.index]
    assert (a["label"] == b["label"]).all(), f"seed {seed}: labels differ"
    checks.append({"seed": seed, "n": len(a), **a["label"].value_counts().reindex(LABELS).to_dict()})
ids = preds[("ce", 42)].index
for k, v in preds.items():
    assert set(v.index) == set(ids) and (v.loc[ids, "label"] == preds[("ce", 42)]["label"]).all()
y = preds[("ce", 42)]["label"].to_numpy()

# ---------- summary mean ± sd ----------
metric_cols = [c for c in table.columns if c not in ("arm", "seed", "run", "commit", "n_test")]
summ = table.groupby("arm")[metric_cols].agg(["mean", "std"])

# ---------- paired bootstrap ----------
P = {k: v.loc[ids, "pred"].to_numpy() for k, v in preds.items()}
lab_idx = {c: i for i, c in enumerate(LABELS)}
yi = np.array([lab_idx[c] for c in y])
Pi = {k: np.array([lab_idx[c] for c in v]) for k, v in P.items()}


def macro_f1_fast(yt, yp):
    f = []
    for c in range(4):
        tp = np.sum((yt == c) & (yp == c)); fp = np.sum((yt != c) & (yp == c)); fn = np.sum((yt == c) & (yp != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.mean(f)


def mean_diff(idx):
    return np.mean([macro_f1_fast(yi[idx], Pi[("la", s)][idx]) - macro_f1_fast(yi[idx], Pi[("ce", s)][idx]) for s in SEEDS])


obs = mean_diff(np.arange(len(yi)))
n = len(yi)
boot = np.array([mean_diff(rng.integers(0, n, n)) for _ in range(B)])
ci = np.percentile(boot, [2.5, 97.5])

# ---------- group bootstrap: exact duplicate caption or identical image bytes ----------
test_json = json.loads((RAW / "vimmsd-private-test.json").read_text(encoding="utf-8"))
img_dir = RAW / "private-test-images" / "test-images"
parent = {i: i for i in ids}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x


def union(a, b):
    parent[find(a)] = find(b)


by_cap, by_img = {}, {}
for i in ids:
    rec = test_json[i]
    cap = re.sub(r"\s+", " ", (rec.get("caption") or "").strip().lower())
    if len(cap) >= 10:
        by_cap.setdefault(cap, []).append(i)
    p = img_dir / rec["image"]
    if p.exists():
        by_img.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append(i)
for groups in (by_cap, by_img):
    for g in groups.values():
        for x in g[1:]:
            union(g[0], x)
gid = np.array([find(i) for i in ids])
uniq, ginv = np.unique(gid, return_inverse=True)
members = [np.where(ginv == g)[0] for g in range(len(uniq))]
n_multi = sum(len(m) > 1 for m in members)
gboot = []
for _ in range(B):
    pick = rng.integers(0, len(members), len(members))
    gboot.append(mean_diff(np.concatenate([members[g] for g in pick])))
gci = np.percentile(gboot, [2.5, 97.5])

# ---------- helped / hurt by true class ----------
flip_rows = []
for seed in SEEDS:
    ce_ok = P[("ce", seed)] == y; la_ok = P[("la", seed)] == y
    for c in LABELS:
        mask = y == c
        flip_rows.append({"seed": seed, "true": c, "n": int(mask.sum()),
                          "LA right, CE wrong": int((la_ok & ~ce_ok & mask).sum()),
                          "CE right, LA wrong": int((ce_ok & ~la_ok & mask).sum())})
flips = pd.DataFrame(flip_rows)
flips.to_csv(OUT / "paired_flips_by_class.csv", index=False)

result = {
    "pairing_checks": checks,
    "per_seed_diff_macro_f1_4": {s: float(table.query("arm=='la' and seed==@s").macro_f1_4.iloc[0]
                                       - table.query("arm=='ce' and seed==@s").macro_f1_4.iloc[0]) for s in SEEDS},
    "mean_diff_macro_f1_4": float(obs),
    "paired_bootstrap_95ci": ci.tolist(), "bootstrap_B": B,
    "paired_bootstrap_p_le_0": float(np.mean(boot <= 0)),
    "group_bootstrap_95ci": gci.tolist(), "n_groups": len(members), "n_multi_member_groups": n_multi,
    "n_dup_caption_groups": sum(len(g) > 1 for g in by_cap.values()),
    "n_dup_image_groups": sum(len(g) > 1 for g in by_img.values()),
}
(OUT / "paired_analysis.json").write_text(json.dumps(result, indent=2))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print(table.round(4).to_string())
print(summ.round(4).T.to_string())
print(flips.groupby("true")[["n", "LA right, CE wrong", "CE right, LA wrong"]].sum())
print(flips.to_string())
print(json.dumps(result, indent=2))
