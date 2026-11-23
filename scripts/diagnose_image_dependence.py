"""WS-A image-dependence diagnostics (reports/plan/eacl_srw_run_plan.md §2). Inference only, no retraining.

    VIMMSD_ENV=server python scripts/diagnose_image_dependence.py \
        --runs p11_controlled/ca_ce_s42 ... p11_caption_la_s2 --out data/server/outputs/p12_srw/image_dependence

The caption stays fixed. Conditions: original, random image swap (derangement), same-label swap, different-label
swap (each with several swap seeds), mask with the train-mean image-encoder output, and mask with a grey image.
Results are never used to select a model (reports/plan/protocol.md).

Outputs in --out:
  <run>/probs.npz                  probabilities per condition (key = condition), ids, labels
  <run>/check_original.json        original condition vs the run's own test_predictions.csv
  per_run_condition.csv            one row per run × condition × sample set
  arm_condition.csv                mean over seeds per arm, with paired bootstrap CIs
  flip_matrix.csv                  orig → new prediction counts, by arm, condition family and gold class
  subsets.json                     audit "image needed" subsets (ids) and their rule
"""
import argparse
import csv
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vimmsd.analysis.image_dependence import build_conditions, compare, encode_test_set, image_feature_mean, run_condition
from vimmsd.analysis.shortcut_check import load_trained_run
from vimmsd.data.dataset import build_dataloaders, build_datasets, build_image_transform
from vimmsd.data.image_io import encoder_mean_color
from vimmsd.models.classifier import MODALITY_NEEDS
from vimmsd.training.metrics import compute_metrics
from vimmsd.training.trainer import selection_score

logger = logging.getLogger("image_dependence")
OUTPUTS = Path("data/server/outputs")
AUDIT = Path("reports/results/phase11_label_audit")
SHORT = ["not", "text", "image", "multi"]


# ---------------------------------------------------------------- audit subsets

def _load_sheet(path):
    """Same repairs as scripts/analysis/p11/analyze_label_audit.py (shifted columns, unquoted commas)."""
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    header, out = rows[0], {}
    for r in rows[1:]:
        if len(r) == len(header) + 3 and not any(r[2:5]) and header[2:] == ["sarcasm", "confidence", "evidence"]:
            r = r[:2] + r[5:8]
        elif len(r) > len(header) and header[-1] == "evidence":
            r = r[:len(header) - 1] + [",".join(r[len(header) - 1:])]
        out[r[0]] = {k: v.strip() for k, v in zip(header, r)}
    return out


def audit_subsets(audit_dir=AUDIT):
    """Rule (audit README): caption alone not sarcastic, pair sarcastic, image_contribution = can."""
    key = _load_sheet(audit_dir / "coordinator_key.csv")

    def needs(who):
        cap, pair = _load_sheet(audit_dir / f"01_caption_{who}.csv"), _load_sheet(audit_dir / f"03_pair_{who}.csv")
        return {a for a in key if cap[a]["sarcasm"] and pair[a]["sarcasm"] and cap[a]["sarcasm"] != "co"
                and pair[a]["sarcasm"] == "co" and pair[a]["image_contribution"] == "can"}

    h1, h2 = needs("person_1"), needs("person_2")
    a1, a2 = needs("ai_rater_1"), needs("ai_rater_2")
    sets = {"audit_both_humans": h1 & h2, "audit_all_four": h1 & h2 & a1 & a2, "audit_either_human": h1 | h2,
            "audit_all40": set(key)}
    prior = audit_dir / "image_needed_human_model_comparison.csv"
    if prior.exists():  # the 11-sample set used in phase11_la_controlled.md tables
        sets["audit_prior11"] = set(pd.read_csv(prior)["audit_id"])
    return {name: {"audit_ids": sorted(s), "test_ids": sorted(str(key[a]["test_id"]) for a in s)}
            for name, s in sets.items()}


# ---------------------------------------------------------------- constants (mean / grey image)

def mask_constants(cfg, model, collator, device, use_amp, cache_dir, dtype):
    from PIL import Image

    name = cfg.model.image_encoder.name
    path = cache_dir / f"train_mean_image_emb_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.pt"
    fill = encoder_mean_color(name)
    eval_tf = build_image_transform(cfg, False, fill)
    if path.exists():
        mean = torch.load(path)
    else:
        train = build_datasets(cfg, False, True, splits=("train",))["train"]
        train.image_transform = eval_tf  # no train-time augmentation for the statistic
        loader = torch.utils.data.DataLoader(train, batch_size=cfg.train.eval_batch_size, shuffle=False,
                                             num_workers=0, collate_fn=collator)
        from tqdm import tqdm
        mean = image_feature_mean(model.image_encoder, (b["pixel_values"] for b in tqdm(loader, desc="train mean")),
                                  device, use_amp, torch.float32)
        mean["n_bad_images"] = train._bad_images
        torch.save(mean, path)
        logger.info("train-mean image embedding over %d images saved to %s", mean["n"], path)
    grey_px = collator.image_processor(images=[eval_tf(Image.new("RGB", (224, 224), (128, 128, 128)))],
                                       return_tensors="pt")["pixel_values"]
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        g = model.image_encoder(grey_px.to(device))
    return {"mean": {"tokens": mean["tokens"].to(dtype), "pooled": mean["pooled"].to(dtype)},
            "grey": {"tokens": g["tokens"][0].cpu().to(dtype), "pooled": g["pooled"][0].cpu().to(dtype)}}


# ---------------------------------------------------------------- per run

def diagnose_run(run, args, out_dir):
    run_dir = OUTPUTS / run
    cfg, model, collator, device, use_amp = load_trained_run(run_dir)
    needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
    ds = build_datasets(cfg, needs_text, needs_image, splits=(args.split,))[args.split]
    loader = build_dataloaders(cfg, {args.split: ds}, collator)[args.split]
    enc = encode_test_set(model, loader, device, use_amp)
    labels = enc["labels"]
    conds = build_conditions(labels, args.swap_seeds)
    constants = None
    if enc["image"] is not None:
        constants = mask_constants(cfg, model, collator, device, use_amp, args.out, enc["image"]["tokens"].dtype)
    probs = {name: run_condition(model, enc, src, device, use_amp, constants) for name, src in conds.items()}

    # the original condition must reproduce the run's saved test predictions
    saved = pd.read_csv(run_dir / f"{args.split}_predictions.csv", dtype={"id": str}).set_index("id")
    label_names = list(cfg.data.labels)
    pred = pd.Series([label_names[i] for i in probs["original"].argmax(1)], index=[str(i) for i in enc["ids"]])
    pcols = [f"prob_{c}" for c in label_names]
    max_abs = float(np.abs(saved.loc[pred.index, pcols].to_numpy() - probs["original"]).max())
    check = {"n": len(pred), "pred_mismatches": int((saved.loc[pred.index, "pred"] != pred).sum()),
             "max_abs_prob_diff": max_abs, "modality": cfg.model.modality,
             "n_bad_test_images": getattr(ds, "_bad_images", 0)}
    (out_dir / run).mkdir(parents=True, exist_ok=True)
    (out_dir / run / "check_original.json").write_text(json.dumps(check, indent=2))
    np.savez_compressed(out_dir / run / "probs.npz", ids=np.asarray(enc["ids"], dtype=str), labels=labels,
                        **{f"cond__{k}": v for k, v in probs.items()},
                        **{f"src__{k}": v for k, v in conds.items() if not isinstance(v, str)})
    logger.info("%s: original check %s", run, check)
    if check["pred_mismatches"] > args.max_mismatch:
        raise SystemExit(f"{run}: original condition does not reproduce {args.split}_predictions.csv ({check})")
    return {"ids": [str(i) for i in enc["ids"]], "labels": labels, "probs": probs, "label_names": label_names,
            "exclude": list(cfg.train.get("select_exclude_labels") or []) or ["text-sarcasm"]}


# ---------------------------------------------------------------- summaries

def arm_of(run):
    name = run.split("/")[-1]
    return re.sub(r"_s\d+$", "", name), int(re.search(r"_s(\d+)$", name).group(1))


def family(cond):
    return re.sub(r"_s\d+$", "", cond)


def summarize(results, subsets, args):
    any_r = next(iter(results.values()))
    ids = any_r["ids"]
    label_names = any_r["label_names"]
    C = len(label_names)
    pos = {i: k for k, i in enumerate(ids)}
    masks = {args.set_name: np.ones(len(ids), bool)}
    for name, s in subsets.items():
        m = np.zeros(len(ids), bool)
        m[[pos[t] for t in s["test_ids"] if t in pos]] = True
        masks[name] = m
    same_order = all(r["ids"] == ids for r in results.values())
    if not same_order:  # fold runs: each has its own held-out ids; arm-level bootstrap needs a shared order
        return summarize_concat(results, args)

    rows, flips = [], []
    for run, r in results.items():
        arm, seed = arm_of(run)
        y, p0 = r["labels"], r["probs"]["original"]
        for cond, pc in r["probs"].items():
            cmp = compare(y, p0, pc, C)
            for set_name, m in masks.items():
                met = compute_metrics(y[m], pc[m].argmax(1), label_names)
                row = {"run": run, "arm": arm, "seed": seed, "condition": cond, "family": family(cond),
                       "set": set_name, "n": int(m.sum()),
                       "flips": int(cmp["flip"][m].sum()), "flip_rate": float(cmp["flip"][m].mean()),
                       "mean_d_p_gold": float(cmp["d_p_gold"][m].mean()),
                       "macro_f1_4": met["macro_f1"], "macro_f1_3": selection_score(met, "macro_f1", ["text-sarcasm"]),
                       "accuracy": met["accuracy"], "pred_sarcastic": int((pc[m].argmax(1) != 0).sum()),
                       "correct": int((pc[m].argmax(1) == y[m]).sum())}
                for k, n in enumerate(label_names):
                    row[f"f1_{SHORT[k]}"] = met["per_class"][n]["f1"]
                    row[f"flip_rate_gold_{SHORT[k]}"] = (float(cmp["flip"][m & (y == k)].mean())
                                                         if (m & (y == k)).any() else np.nan)
                rows.append(row)
                if set_name in (args.set_name, "audit_both_humans") and cond != "original":
                    for g in range(C):
                        gm = m & (y == g)
                        fm = np.bincount(cmp["pred_orig"][gm] * C + cmp["pred_cond"][gm], minlength=C * C)
                        for a in range(C):
                            for b in range(C):
                                if fm[a * C + b]:
                                    flips.append({"arm": arm, "seed": seed, "family": family(cond), "condition": cond,
                                                  "set": set_name, "gold": SHORT[g], "from": SHORT[a],
                                                  "to": SHORT[b], "count": int(fm[a * C + b])})
    per = pd.DataFrame(rows)
    per.to_csv(args.out / "per_run_condition.csv", index=False)
    pd.DataFrame(flips).to_csv(args.out / "flip_matrix.csv", index=False)

    # arm level: average over model seeds and swap seeds; paired bootstrap over sample ids (same ids for all arms)
    rng = np.random.default_rng(0)
    arm_rows = []
    by_arm = {}
    for run, r in results.items():
        by_arm.setdefault(arm_of(run)[0], []).append(r)
    for set_name, m in masks.items():
        idx = np.flatnonzero(m)
        boot = rng.integers(0, len(idx), size=(args.n_boot, len(idx)))
        for arm, rs in by_arm.items():
            for fam in sorted({family(c) for c in rs[0]["probs"]} - {"original"}):
                flip_mat, dp_mat = [], []
                for r in rs:
                    for cond, pc in r["probs"].items():
                        if family(cond) == fam:
                            cmp = compare(r["labels"], r["probs"]["original"], pc, C)
                            flip_mat.append(cmp["flip"][idx])
                            dp_mat.append(cmp["d_p_gold"][idx])
                flip_s, dp_s = np.mean(flip_mat, 0), np.mean(dp_mat, 0)  # per sample, averaged over seeds
                fb, db = flip_s[boot].mean(1), dp_s[boot].mean(1)
                sub = per[(per.arm == arm) & (per.family == fam) & (per.set == set_name)]
                orig = per[(per.arm == arm) & (per.condition == "original") & (per.set == set_name)]
                arm_rows.append({
                    "set": set_name, "n": len(idx), "arm": arm, "family": fam, "n_model_seeds": len(rs),
                    "n_condition_draws": len(flip_mat),
                    "flip_rate": flip_s.mean(), "flip_ci_lo": np.quantile(fb, .025), "flip_ci_hi": np.quantile(fb, .975),
                    "flip_rate_sd_over_draws": float(np.std([f.mean() for f in flip_mat], ddof=1)) if len(flip_mat) > 1 else 0.0,
                    "flips_total": int(np.sum(flip_mat)),
                    "mean_d_p_gold": dp_s.mean(), "d_p_gold_ci_lo": np.quantile(db, .025),
                    "d_p_gold_ci_hi": np.quantile(db, .975),
                    "macro_f1_4_orig": orig.macro_f1_4.mean(), "macro_f1_4_cond": sub.macro_f1_4.mean(),
                    "accuracy_orig": orig.accuracy.mean(), "accuracy_cond": sub.accuracy.mean(),
                    "f1_multi_orig": orig.f1_multi.mean(), "f1_multi_cond": sub.f1_multi.mean(),
                    "f1_image_orig": orig.f1_image.mean(), "f1_image_cond": sub.f1_image.mean(),
                    "f1_not_orig": orig.f1_not.mean(), "f1_not_cond": sub.f1_not.mean(),
                    "pred_sarcastic_orig": orig.pred_sarcastic.mean(), "pred_sarcastic_cond": sub.pred_sarcastic.mean(),
                })
    arm = pd.DataFrame(arm_rows)
    arm.to_csv(args.out / "arm_condition.csv", index=False)
    return per, arm


def summarize_concat(results, args):
    """Fold runs (one model per held-out fold): concatenate the folds into one OOF set and summarize it as one
    'run' per arm, so per_run_condition / arm_condition keep the same columns."""
    merged = {}
    for run, r in results.items():
        arm = re.sub(r"_f\d+_s\d+$", "", run.split("/")[-1])
        m = merged.setdefault(f"{arm}_oof_s42", {"ids": [], "labels": [], "probs": {}, "label_names": r["label_names"]})
        m["ids"] += r["ids"]
        m["labels"] = np.concatenate([m["labels"], r["labels"]]).astype(int)
        for c, p in r["probs"].items():
            m["probs"][c] = np.concatenate([m["probs"][c], p]) if c in m["probs"] else p
    return summarize(merged, {}, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs relative to data/server/outputs")
    ap.add_argument("--out", type=Path, default=OUTPUTS / "p12_srw/image_dependence")
    ap.add_argument("--swap-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="test = private test (official split); val = held-out fold for fold-manifest runs")
    ap.add_argument("--max-mismatch", type=int, default=0,
                    help="allowed prediction mismatches vs test_predictions.csv in the original condition")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    args.set_name = "private_test" if args.split == "test" else "oof_train"
    # audit subsets are private-test ids; they do not apply to held-out train folds (ids could collide)
    subsets = audit_subsets() if args.split == "test" else {}
    (args.out / "subsets.json").write_text(json.dumps(
        {"rule": "caption alone not sarcastic (sarcasm != co), pair sarcastic (co), image_contribution = can",
         **subsets}, indent=2))
    logger.info("audit subsets: %s", {k: len(v["test_ids"]) for k, v in subsets.items()})

    results = {run: diagnose_run(run, args, args.out) for run in args.runs}
    _, arm = summarize(results, subsets, args)
    cols = ["set", "n", "arm", "family", "flip_rate", "flip_ci_lo", "flip_ci_hi", "mean_d_p_gold",
            "macro_f1_4_orig", "macro_f1_4_cond", "accuracy_orig", "accuracy_cond"]
    print(arm[arm.set.isin([args.set_name, "audit_both_humans"])][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
