"""WS-C round-2 human-audit sample (reports/plan/eacl_srw_run_plan.md §4, reports/plan/protocol.md).

    python scripts/sample_audit_round2.py                 # dry run: print pools and the drawn strata, write nothing
    python scripts/sample_audit_round2.py --write [--copy-images]

Draws 120 private-test items, excluding the 40 pilot ids (reports/results/phase11_label_audit/coordinator_key.csv):

  stratified (60, random within gold class; Gate A is tested on this half only):
      image: all remaining (23), not: 12, multi: 25. Each row has a weight = class pool / drawn, for
      population estimates. `text` has no items left outside the pilot.
  purposive (60, descriptive only; selected on model behaviour, so never used for the Gate A test):
      disagree: 15 not + 15 multi where the majority vote of xattn-LA (ca_la_s{42,1,2}) differs from caption-LA
                (p11_caption_la_s{42,1,2});
      flip:     15 not + 15 multi with a WS-A xattn-LA random-swap flip rate >= 0.5 (over 3 seeds x 5 swaps),
                not already drawn.

Model signals come from data/server/outputs/p12_srw/image_dependence/*/probs.npz (WS-A). With --write the script
writes to --out: blank forms batch{1,2}/01_caption.csv, 02_image.csv, 03_pair.csv (same columns as the pilot;
batch1 = stratified half, rated first; batch2 = purposive; each pass shuffled independently; audit ids are shuffled
over all 120 so they reveal nothing), coordinator_key.csv (gold, stratum, weight and model
signals; coordinator only) and, with --copy-images, images/R###.jpg.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]
WSA = Path("data/server/outputs/p12_srw/image_dependence")
PILOT_KEY = Path("reports/results/phase11_label_audit/coordinator_key.csv")
RAW = Path("data/server/raw")
STRAT = {"image-sarcasm": None, "not-sarcasm": 12, "multi-sarcasm": 25}  # None = take all
PURP = {"disagree": {"not-sarcasm": 15, "multi-sarcasm": 15}, "flip": {"not-sarcasm": 15, "multi-sarcasm": 15}}
FLIP_MIN = 0.5
N_SWAP = 5


def majority(preds):  # (n_seeds, n) -> (n,)
    return np.array([np.bincount(c, minlength=4).argmax() for c in preds.T])


def model_signals():
    la = [np.load(WSA / f"p11_controlled/ca_la_s{s}/probs.npz") for s in (42, 1, 2)]
    cap = [np.load(WSA / f"p11_caption_la_s{s}/probs.npz") for s in (42, 1, 2)]
    ids = la[0]["ids"].astype(str)
    for z in la + cap:
        assert (z["ids"].astype(str) == ids).all()
    orig = np.stack([z["cond__original"].argmax(1) for z in la])
    flip = np.mean([[z[f"cond__swap_random_s{k}"].argmax(1) != z["cond__original"].argmax(1) for k in range(N_SWAP)]
                    for z in la], axis=(0, 1))
    mx = majority(orig)
    mc = majority(np.stack([z["cond__original"].argmax(1) for z in cap]))
    return pd.DataFrame({"test_id": ids, "gold_idx": la[0]["labels"], "xattn_la_major": [LABELS[i] for i in mx],
                         "caption_la_major": [LABELS[i] for i in mc], "disagree": mx != mc,
                         "la_swap_flip_rate": flip}).set_index("test_id")


def draw(pool, n, rng):
    ids = pool.index.to_numpy()
    if n is None or n >= len(ids):
        return list(ids)
    return list(rng.choice(ids, size=n, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("reports/results/phase11_label_audit_r2"))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--copy-images", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    meta = json.loads((RAW / "vimmsd-private-test.json").read_text(encoding="utf-8"))
    gold = json.loads((RAW / "vimmsd-private-test-ref.json").read_text(encoding="utf-8"))
    gold = gold.get("results", gold)
    sig = model_signals()
    sig["gold"] = [gold[i] for i in sig.index]
    assert (sig.gold.map(LABELS.index) == sig.gold_idx).all(), "WS-A labels differ from the reference labels"
    pilot = set(pd.read_csv(PILOT_KEY, dtype=str).test_id)
    pool = sig[~sig.index.isin(pilot)]
    print(f"pool outside pilot: {len(pool)}  ", pool.gold.value_counts().to_dict())

    rows = []
    for g, n in STRAT.items():
        p = pool[pool.gold == g]
        picked = draw(p, n, rng)
        rows += [dict(test_id=i, half="stratified", stratum=f"strat_{g.split('-')[0]}", weight=len(p) / len(picked))
                 for i in sorted(picked, key=int)]
    taken = {r["test_id"] for r in rows}
    for name, per in PURP.items():
        for g, n in per.items():
            p = pool[(pool.gold == g) & ~pool.index.isin(taken)]
            p = p[p.disagree] if name == "disagree" else p[p.la_swap_flip_rate >= FLIP_MIN]
            picked = draw(p, n, rng)
            if len(picked) < n:
                raise SystemExit(f"{name}/{g}: pool {len(p)} < {n}")
            print(f"purposive {name}/{g}: pool {len(p)}, drawn {len(picked)}")
            rows += [dict(test_id=i, half="purposive", stratum=f"{name}_{g.split('-')[0]}", weight=np.nan)
                     for i in sorted(picked, key=int)]
            taken |= set(picked)
    key = pd.DataFrame(rows).join(sig.drop(columns="gold_idx"), on="test_id")
    assert key.test_id.is_unique and not key.test_id.isin(pilot).any()

    # audit ids: one shuffled order over all 120, so ids reveal nothing about the half
    order = rng.permutation(len(key))
    key = key.iloc[order].reset_index(drop=True)
    key.insert(0, "audit_id", [f"R{k + 1:03d}" for k in range(len(key))])
    key["original_label"] = key.pop("gold")
    key["source_image"] = [meta[i]["image"] for i in key.test_id]
    key["review_image"] = [f"images/{r}.jpg" for r in key.audit_id]

    print("\nby half/stratum:\n", key.groupby(["half", "stratum"]).size().to_string())
    print("\nstratified half, LA swap flip rate by gold:\n",
          key[key.half == "stratified"].groupby("original_label").la_swap_flip_rate.agg(["size", "mean"]).round(3).to_string())
    if not a.write:
        print("\nDry run: nothing written. Add --write after the team approves guideline v2.")
        return

    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "coordinator_key.csv").exists():
        raise SystemExit(f"{a.out}/coordinator_key.csv exists; refusing to overwrite a drawn sample")
    cap = {r.audit_id: meta[r.test_id]["caption"] for r in key.itertuples()}
    passes = {
        "01_caption.csv": lambda r: dict(audit_id=r, caption=cap[r], sarcasm="", confidence="", evidence=""),
        "02_image.csv": lambda r: dict(audit_id=r, image=f"images/{r}.jpg", sarcasm="", confidence="", evidence=""),
        "03_pair.csv": lambda r: dict(audit_id=r, caption=cap[r], image=f"images/{r}.jpg", sarcasm="", confidence="",
                                      evidence="", image_contribution=""),
    }
    # batch1 = stratified half (Gate A; rated first), batch2 = purposive half. Raters are not told what a batch is.
    for b, half in [("batch1", "stratified"), ("batch2", "purposive")]:
        (a.out / b).mkdir(exist_ok=True)
        sub = key[key.half == half].audit_id.to_numpy()
        for fname, make in passes.items():  # each pass in its own order, as in the pilot
            ids = sub[rng.permutation(len(sub))]
            pd.DataFrame([make(r) for r in ids]).to_csv(a.out / b / fname, index=False, encoding="utf-8-sig")
    key.to_csv(a.out / "coordinator_key.csv", index=False, encoding="utf-8-sig")
    if a.copy_images:
        (a.out / "images").mkdir(exist_ok=True)
        src_dir = RAW / "private-test-images" / "test-images"
        for r in key.itertuples():
            shutil.copyfile(src_dir / r.source_image, a.out / r.review_image)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
