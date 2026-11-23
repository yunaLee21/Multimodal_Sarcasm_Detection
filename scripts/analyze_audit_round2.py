"""WS-C round-2 audit analysis, fixed before rating in reports/plan/ws_c_round2.md §5.

    python scripts/analyze_audit_round2.py [--dir reports/results/phase11_label_audit_r2] [--n-perm 10000]

Reads <dir>/coordinator_key.csv and the returned sheets <dir>/batch{1,2}/{01_caption,02_image,03_pair}_<rater>.csv
(<rater> = person_1, person_2; ai_rater_{1,2} are used only for the "all four" sensitivity set, if present).
Runs whatever is complete: a batch is analysed when both humans returned all three sheets for it.

  checks       values in the allowed sets, missing answers, v2 rule (image_contribution = NA unless pair = co)
  gate_a       batch 1 only. Image-needed (caption != co, pair = co, image_contribution = can) per rater; sets
               both humans (main), either human, all four (if AI sheets exist). Outcomes: per-item flip rate from the
               WS-A probs.npz, xattn-LA / xattn-CE x random swap (3 seeds x 5 swaps) / mean mask (3 seeds).
               Statistic: mean flip (needed) - mean flip (other). p: two-sided permutation, labels permuted within
               gold class. Holm over the 4 outcomes, per set. Strong form of claim 2 iff LA random swap is
               Holm-significant and positive on the main set.
  reliability  batch 1: raw agreement and Cohen's kappa per pass (unweighted and with stratum weights), kappa on image
               role given both raters said pair = co, human-vs-gold disagreement (weighted, stratified bootstrap CI).
  purposive    batch 2: image-needed rate per stratum vs the weighted batch-1 rate. Descriptive, no tests.
  adjudication every item where the two humans differ on any field, without gold or model columns.

Writes <dir>/analysis/{round2_analysis.json, items.csv, adjudication_batch{1,2}.csv}.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

WSA = Path("data/server/outputs/p12_srw/image_dependence/p11_controlled")
HUMANS = ["person_1", "person_2"]
AIS = ["ai_rater_1", "ai_rater_2"]
PARTS = ["01_caption", "02_image", "03_pair"]
SARC = {"co", "khong", "khong_chac"}
CONTRIB = {"can", "bo_sung", "khong", "khong_chac", "NA"}
CONF = {"1", "2", "3"}
OUTCOMES = [("xattn_la", "random_swap"), ("xattn_ce", "random_swap"), ("xattn_la", "mean_mask"), ("xattn_ce", "mean_mask")]


def load(path):
    """Read a sheet keyed by audit_id. Repairs unquoted commas in the free-text `evidence` column (as in the pilot)."""
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    header, out = rows[0], {}
    ev = header.index("evidence") if "evidence" in header else None
    for r in rows[1:]:
        if not any(x.strip() for x in r):
            continue
        if len(r) == len(header) + 3 and not any(r[2:5]) and header[2:] == ["sarcasm", "confidence", "evidence"]:
            r = r[:2] + r[5:8]  # answers shifted 3 columns right (seen in the pilot)
        elif len(r) > len(header) and ev is not None:
            extra = len(r) - len(header)
            r = r[:ev] + [",".join(r[ev:ev + extra + 1])] + r[ev + extra + 1:]
        assert len(r) == len(header), (path, r[:2])
        out[r[0].strip()] = {k: v.strip() for k, v in zip(header, r)}
    return out


def flip_rates(ids):
    """Per-item flip rate under random swap (mean over 3 seeds x 5 swaps) and mean mask (mean over 3 seeds)."""
    out = {}
    for arm, run in (("xattn_la", "ca_la"), ("xattn_ce", "ca_ce")):
        zs = [np.load(WSA / f"{run}_s{s}" / "probs.npz") for s in (42, 1, 2)]
        zid = zs[0]["ids"].astype(str)
        swap = np.mean([[z[k].argmax(1) != z["cond__original"].argmax(1) for k in z.files
                         if k.startswith("cond__swap_random_")] for z in zs], axis=(0, 1))
        mask = np.mean([z["cond__mask_mean"].argmax(1) != z["cond__original"].argmax(1) for z in zs], axis=0)
        s = pd.DataFrame({f"{arm}_random_swap": swap, f"{arm}_mean_mask": mask}, index=zid)
        out[arm] = s.loc[ids]
    return pd.concat(out.values(), axis=1)


def perm_test(y, needed, strata, n_perm, rng):
    """Two-sided permutation p for mean(y | needed) - mean(y | not), permuting `needed` within strata."""
    needed = needed.astype(bool)
    if needed.all() or not needed.any():
        return float("nan"), float("nan")
    obs = y[needed].mean() - y[~needed].mean()
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    perm = needed.copy()
    hits = 0
    for _ in range(n_perm):
        for g in groups:
            perm[g] = rng.permutation(needed[g])
        d = y[perm].mean() - y[~perm].mean()
        hits += abs(d) >= abs(obs) - 1e-12
    return float(obs), (hits + 1) / (n_perm + 1)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def kappa(a, b, w=None):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(set(a) | set(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b, sample_weight=w))


def wmean(x, w):
    return float(np.sum(np.asarray(x, float) * w) / np.sum(w))


def strat_boot_ci(x, strata, w, rng, n_boot=10000):
    x, strata = np.asarray(x, float), np.asarray(strata)
    idx = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    vals = []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(g, size=len(g)) for g in idx])
        vals.append(wmean(x[b], w[b]))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("reports/results/phase11_label_audit_r2"))
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    key = pd.read_csv(a.dir / "coordinator_key.csv", dtype={"test_id": str}, encoding="utf-8-sig").set_index("audit_id")
    out_dir = a.dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    result = {"n_key": len(key), "batches": {}, "checks": {}}

    # ---- load whatever is complete
    sheets, raters = {}, {}
    for b, half in (("batch1", "stratified"), ("batch2", "purposive")):
        want = set(key.index[key.half == half])
        have = [w for w in HUMANS + AIS if all((a.dir / b / f"{p}_{w}.csv").exists() for p in PARTS)]
        result["batches"][b] = {"half": half, "n": len(want), "raters_complete": have}
        if not all(h in have for h in HUMANS):
            continue
        raters[b] = have
        for w in have:
            for p in PARTS:
                s = load(a.dir / b / f"{p}_{w}.csv")
                extra = set(s) - want
                assert not extra, f"{b}/{p}_{w}: ids not in this batch: {sorted(extra)[:5]}"
                sheets[(b, w, p)] = s
    if not raters:
        (out_dir / "round2_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, indent=1, ensure_ascii=False))
        print("\nNo batch has all three sheets from both humans yet.")
        return

    # ---- per-item table
    flips = flip_rates(list(key.test_id))
    flips.index = key.index
    rows = []
    for b, have in raters.items():
        for aid in key.index[key.half == result["batches"][b]["half"]]:
            r = {"audit_id": aid, "batch": b}
            for w in have:
                c = sheets[(b, w, "01_caption")].get(aid, {})
                i = sheets[(b, w, "02_image")].get(aid, {})
                p = sheets[(b, w, "03_pair")].get(aid, {})
                r[f"caption_{w}"] = c.get("sarcasm", "")
                r[f"image_{w}"] = i.get("sarcasm", "")
                r[f"pair_{w}"] = p.get("sarcasm", "")
                r[f"contrib_{w}"] = p.get("image_contribution", "")
                rated = bool(r[f"caption_{w}"] and r[f"pair_{w}"])
                r[f"needed_{w}"] = (r[f"caption_{w}"] != "co" and r[f"pair_{w}"] == "co"
                                    and r[f"contrib_{w}"] == "can") if rated else None
            rows.append(r)
    items = pd.DataFrame(rows).set_index("audit_id").join(key[["test_id", "half", "stratum", "weight", "original_label"]]).join(flips)

    def need(w):
        return items[f"needed_{w}"] if f"needed_{w}" in items else pd.Series(None, index=items.index, dtype=object)
    both_rated = need("person_1").notna() & need("person_2").notna()
    items["needed_both_humans"] = (need("person_1").eq(True) & need("person_2").eq(True)).where(both_rated)
    items["needed_either_human"] = (need("person_1").eq(True) | need("person_2").eq(True)).where(both_rated)
    if all(f"needed_{w}" in items for w in AIS):
        all_rated = both_rated & need("ai_rater_1").notna() & need("ai_rater_2").notna()
        items["needed_all_four"] = pd.concat([need(w).eq(True) for w in HUMANS + AIS], axis=1).all(axis=1).where(all_rated)
    sets = [c for c in ("needed_both_humans", "needed_all_four", "needed_either_human") if c in items]
    items.to_csv(out_dir / "items.csv")

    # ---- checks
    for b, have in raters.items():
        n_b = result["batches"][b]["n"]
        for w in have:
            chk = {}
            for p in PARTS:
                s = sheets[(b, w, p)]
                chk[f"{p}.missing_rows"] = n_b - len(s)
                chk[f"{p}.blank_sarcasm"] = sum(not v["sarcasm"] for v in s.values())
                chk[f"{p}.bad_sarcasm"] = sorted(k for k, v in s.items() if v["sarcasm"] and v["sarcasm"] not in SARC)
                chk[f"{p}.bad_confidence"] = sorted(k for k, v in s.items() if v["confidence"] and v["confidence"] not in CONF)
                chk[f"{p}.thieu_anh"] = sum("thieu_anh" in v.get("evidence", "") for v in s.values())
            pr = sheets[(b, w, "03_pair")]
            chk["03_pair.bad_contribution"] = sorted(k for k, v in pr.items()
                                                     if v["image_contribution"] and v["image_contribution"] not in CONTRIB)
            chk["v2_violations_contrib_not_NA_when_pair_not_co"] = sorted(
                k for k, v in pr.items() if v["sarcasm"] in ("khong", "khong_chac") and v["image_contribution"] not in ("", "NA"))
            chk["v2_missing_contrib_when_pair_co"] = sorted(
                k for k, v in pr.items() if v["sarcasm"] == "co" and v["image_contribution"] in ("", "NA"))
            result["checks"][f"{b}/{w}"] = chk

    # ---- Gate A (batch 1)
    b1 = items[items.batch == "batch1"]
    if len(b1):
        ga = {"n_items": len(b1), "sets": {}}
        for col in sets:
            d = b1[b1[col].notna()]
            needed = d[col].astype(bool).to_numpy()
            strata = d.original_label.to_numpy()
            res = {"n": len(d), "n_needed": int(needed.sum()),
                   "needed_by_gold": d[d[col].astype(bool)].original_label.value_counts().to_dict(), "tests": []}
            for arm, cond in OUTCOMES:
                y = d[f"{arm}_{cond}"].to_numpy()
                diff, p = perm_test(y, needed, strata, a.n_perm, rng)
                by_gold = {g: {"needed": float(y[(strata == g) & needed].mean()) if ((strata == g) & needed).any() else None,
                               "other": float(y[(strata == g) & ~needed].mean()) if ((strata == g) & ~needed).any() else None}
                           for g in np.unique(strata)}
                res["tests"].append({"arm": arm, "condition": cond,
                                     "mean_needed": float(y[needed].mean()) if needed.any() else None,
                                     "mean_other": float(y[~needed].mean()) if (~needed).any() else None,
                                     "diff": diff, "p_perm_within_gold": p, "by_gold": by_gold})
            ps = [t["p_perm_within_gold"] for t in res["tests"]]
            if not np.isnan(ps).any():
                for t, h in zip(res["tests"], holm(ps)):
                    t["p_holm"] = float(h)
            ga["sets"][col] = res
        main_la = next((t for t in ga["sets"].get("needed_both_humans", {}).get("tests", [])
                        if t["arm"] == "xattn_la" and t["condition"] == "random_swap"), None)
        ga["strong_form_supported"] = bool(main_la and main_la.get("p_holm", 1) < .05 and main_la["diff"] > 0)
        ga["rule"] = "strong form iff xattn-LA random swap, both-humans set: Holm p < .05 and diff > 0"
        result["gate_a"] = ga

        # ---- reliability (batch 1, weighted to the test distribution)
        w = b1.weight.to_numpy(float)
        rel = {}
        for part in ("caption", "image", "pair"):
            x, y = b1[f"{part}_person_1"], b1[f"{part}_person_2"]
            ok = (x != "") & (y != "")
            agree = (x[ok] == y[ok]).to_numpy()
            rel[part] = {"n": int(ok.sum()), "raw_agreement": float(agree.mean()),
                         "raw_agreement_weighted": wmean(agree, w[ok.to_numpy()]),
                         "kappa": kappa(x[ok], y[ok]), "kappa_weighted": kappa(x[ok], y[ok], w[ok.to_numpy()]),
                         "uncertain_any": int(((x == "khong_chac") | (y == "khong_chac")).sum())}
        both_co = (b1.pair_person_1 == "co") & (b1.pair_person_2 == "co")
        x, y = b1.contrib_person_1[both_co], b1.contrib_person_2[both_co]
        rel["image_role_given_both_pair_co"] = {"n": int(both_co.sum()), "raw_agreement": float((x == y).mean()) if len(x) else None,
                                                "kappa": kappa(x, y), "crosstab": pd.crosstab(x, y).to_dict()}
        gold_sarc = (b1.original_label != "not-sarcasm").to_numpy()
        hg = {}
        for who in HUMANS:
            ok = b1[f"pair_{who}"].isin(["co", "khong"]).to_numpy()
            dis = ((b1[f"pair_{who}"] == "co").to_numpy() != gold_sarc)[ok]
            hg[who] = {"n_decided": int(ok.sum()), "disagree_rate": float(dis.mean()),
                       "disagree_rate_weighted": wmean(dis, w[ok]),
                       "ci95_weighted": strat_boot_ci(dis, b1.stratum.to_numpy()[ok], w[ok], rng)}
        agreed = (b1.pair_person_1 == b1.pair_person_2) & b1.pair_person_1.isin(["co", "khong"])
        ok = agreed.to_numpy()
        dis = ((b1.pair_person_1 == "co").to_numpy() != gold_sarc)[ok]
        hg["humans_agreed"] = {"n_decided": int(ok.sum()), "disagree_rate": float(dis.mean()) if ok.any() else None,
                               "disagree_rate_weighted": wmean(dis, w[ok]) if ok.any() else None,
                               "ci95_weighted": strat_boot_ci(dis, b1.stratum.to_numpy()[ok], w[ok], rng) if ok.any() else None,
                               "by_gold": {g: {"n": int(((b1.original_label == g) & agreed).sum()),
                                               "human_says_sarcastic": int(((b1.original_label == g) & agreed & (b1.pair_person_1 == "co")).sum())}
                                           for g in sorted(b1.original_label.unique())}}
        rel["pair_vs_gold_sarcastic"] = hg
        nb = b1["needed_both_humans"].dropna()
        rel["needed_both_humans_rate_weighted"] = wmean(nb.astype(float), b1.loc[nb.index, "weight"].to_numpy(float)) if len(nb) else None
        result["reliability_batch1"] = rel

    # ---- purposive (batch 2), descriptive
    b2 = items[items.batch == "batch2"]
    if len(b2):
        pur = {}
        for col in sets:
            d = b2[b2[col].notna()]
            pur[col] = {s: {"n": int((d.stratum == s).sum()), "needed": int(d[d.stratum == s][col].astype(bool).sum())}
                        for s in sorted(b2.stratum.unique())}
        if len(b1):
            pur["batch1_weighted_rate_both_humans"] = result["reliability_batch1"]["needed_both_humans_rate_weighted"]
        result["purposive_batch2"] = pur

    # ---- adjudication tables (no gold, no model signals)
    for b, have in raters.items():
        recs = []
        for aid in items.index[items.batch == b]:
            for p, field, col in (("01_caption", "sarcasm", "caption"), ("02_image", "sarcasm", "image"),
                                  ("03_pair", "sarcasm", "pair"), ("03_pair", "image_contribution", "contrib")):
                v1, v2 = (sheets[(b, w, p)].get(aid, {}).get(field, "") for w in HUMANS)
                if v1 != v2:
                    e1, e2 = (sheets[(b, w, p)].get(aid, {}).get("evidence", "") for w in HUMANS)
                    recs.append({"audit_id": aid, "field": col, "person_1": v1, "person_2": v2,
                                 "evidence_person_1": e1, "evidence_person_2": e2, "decision": ""})
        pd.DataFrame(recs, columns=["audit_id", "field", "person_1", "person_2", "evidence_person_1", "evidence_person_2",
                                    "decision"]).to_csv(out_dir / f"adjudication_{b}.csv", index=False, encoding="utf-8-sig")
        result["batches"][b]["n_disagreements"] = len(recs)

    (out_dir / "round2_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: result[k] for k in result if k in ("batches", "gate_a")}, indent=1, ensure_ascii=False, default=str)[:6000])


if __name__ == "__main__":
    main()
