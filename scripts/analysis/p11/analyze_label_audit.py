"""Phase 11 label audit: rater agreement, the README "cần ảnh" (image needed) set, and model predictions on it.

Run from the repo root: python data/server/outputs/p11_controlled/analyze_label_audit.py
README rule: the agreed judgment says the caption alone is not sarcastic, the pair is sarcastic,
and the image gives the necessary evidence (image_contribution = can).
person_2's caption sheet has 39 of 40 answers shifted 3 columns right; it is repaired here on read,
and the original file is left unchanged.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

A = Path("reports/results/phase11_label_audit")
O = Path("data/server/outputs")
HUMANS = ["person_1", "person_2"]
AIS = ["ai_rater_1", "ai_rater_2"]
SHORT = {"not-sarcasm": "not", "text-sarcasm": "text", "image-sarcasm": "image", "multi-sarcasm": "multi"}


def load(name):
    path = A / name
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    header, out, repaired = rows[0], {}, 0
    for r in rows[1:]:
        if len(r) == len(header) + 3 and not any(r[2:5]) and header[2:] == ["sarcasm", "confidence", "evidence"]:
            r = r[:2] + r[5:8]
            repaired += 1
        elif len(r) > len(header) and header[-1] == "evidence":  # unquoted commas inside the free-text last column
            r = r[:len(header) - 1] + [",".join(r[len(header) - 1:])]
            repaired += 1
        assert len(r) == len(header), (name, r[:2])
        out[r[0]] = {k: v.strip() for k, v in zip(header, r)}
    return out, repaired


sheets, repairs = {}, {}
for who in HUMANS + AIS:
    for part in ("01_caption", "02_image", "03_pair"):
        sheets[(who, part)], repairs[f"{part}_{who}"] = load(f"{part}_{who}.csv")
key, _ = load("coordinator_key.csv")
ids = sorted(key)


def needs_image(who, a):
    c, p = sheets[(who, "01_caption")][a], sheets[(who, "03_pair")][a]
    if not p["sarcasm"] or not c["sarcasm"]:
        return None  # not rated
    return c["sarcasm"] != "co" and p["sarcasm"] == "co" and p["image_contribution"] == "can"


# ---- agreement between the two humans
agree = {}
for part, field in (("01_caption", "sarcasm"), ("02_image", "sarcasm"), ("03_pair", "sarcasm"), ("03_pair", "image_contribution")):
    pairs = [(sheets[("person_1", part)][a][field], sheets[("person_2", part)][a][field]) for a in ids]
    rated = [(x, y) for x, y in pairs if x and y]
    agree[f"{part}.{field}"] = {"rated_by_both": len(rated), "agree": sum(x == y for x, y in rated),
                                "raw_agreement": round(sum(x == y for x, y in rated) / len(rated), 3),
                                "uncertain_or_blank": sum(1 for x, y in pairs if "khong_chac" in (x, y) or not x or not y)}
ni = {w: {a: needs_image(w, a) for a in ids} for w in HUMANS + AIS}
both_humans = [a for a in ids if ni["person_1"][a] and ni["person_2"][a]]
either_human = [a for a in ids if ni["person_1"][a] or ni["person_2"][a]]
human_disagree = [a for a in ids if ni["person_1"][a] is not None and ni["person_2"][a] is not None
                  and ni["person_1"][a] != ni["person_2"][a]]

# ---- model predictions on the audit samples
runs = {"caption_la": "p11_caption_la_s{}", "xattn_la": "p11_controlled/ca_la_s{}", "xattn_ce": "p11_controlled/ca_ce_s{}"}
P = {m: [pd.read_csv(O / pat.format(s) / "test_predictions.csv", dtype={"id": str}).set_index("id") for s in (42, 1, 2)]
     for m, pat in runs.items()}
rows = []
for a in ids:
    tid, lab = key[a]["test_id"], SHORT[key[a]["original_label"]]
    r = {"audit_id": a, "test_id": tid, "group": key[a]["sampling_group"], "label": lab,
         **{f"needs_image_{w}": ni[w][a] for w in HUMANS + AIS},
         "needs_image_both_humans": a in both_humans}
    for m, ps in P.items():
        assert all(SHORT[p.loc[tid, "label"]] == lab for p in ps)
        preds = [SHORT[p.loc[tid, "pred"]] for p in ps]
        r[f"{m}_preds"] = "/".join(preds)
        r[f"{m}_correct"] = sum(x == lab for x in preds)
        r[f"{m}_sarcastic"] = sum(x != "not" for x in preds)
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(A / "image_needed_model_comparison.csv", index=False)


def summarize(mask):
    s = df[mask]
    out = {"n": len(s), "labels": dict(Counter(s.label)), "groups": dict(Counter(s.group))}
    for m in runs:
        out[m] = {"correct": int(s[f"{m}_correct"].sum()), "sarcastic": int(s[f"{m}_sarcastic"].sum()), "of": 3 * len(s)}
    d = s["xattn_la_correct"] - s["caption_la_correct"]
    out["xattn_la_vs_caption_la_per_sample"] = {"better": int((d > 0).sum()), "worse": int((d < 0).sum()), "tied": int((d == 0).sum())}
    return out


sets = {"both humans": df.needs_image_both_humans, "either human": df.audit_id.isin(either_human),
        "all four raters": df[[f"needs_image_{w}" for w in HUMANS + AIS]].eq(True).all(axis=1)}
result = {"csv_repairs": {k: v for k, v in repairs.items() if v}, "human_agreement": agree,
          "needs_image_counts": {w: sum(bool(v) for v in ni[w].values()) for w in ni},
          "both_humans_ids": both_humans, "human_disagreements_on_needs_image": human_disagree,
          "subsets": {k: summarize(v) for k, v in sets.items()}}
(O / "p11_controlled" / "label_audit_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
pd.set_option("display.width", 250)
print(df[df.needs_image_both_humans][["audit_id", "group", "label", "caption_la_preds", "xattn_la_preds", "xattn_ce_preds"]].to_string(index=False))
print(json.dumps(result, indent=1, ensure_ascii=False))
