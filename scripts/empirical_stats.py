"""Thống kê cho nghiên cứu thực nghiệm (docs/empirical_analysis_plan.md mục 6, 8.4, 11), đọc <run>/val_predictions.csv.

    # Experiment A: họ so sánh khai báo trước, bootstrap ghép cặp 2.000 lần, Holm trong họ
    python scripts/empirical_stats.py contrasts --family A_modality \\
        --pairs empA_C:empA_T empA_O:empA_T empA_D:empA_T empA_OD:empA_O empA_OD:empA_D empA_M:empA_OD
    # Experiment C: macro-F1 out-of-fold gộp 3 fold cho từng treatment
    python scripts/empirical_stats.py pooled --treatments empC_B0 empC_B1 empC_B2 empC_B3
    # subset cố định trên val (mô tả, kèm support)
    python scripts/empirical_stats.py subsets --models empA_T empA_C empA_M

Run có tên <model>_s<seed> (contrasts, subsets) hoặc <treatment>_f<fold>_s<seed> (pooled), trong --runs-dir
(mặc định paths.output_dir). Kết quả ghi vào <runs-dir>/_empirical/statistics/. contrasts dùng macro-F1 bốn lớp
(--exclude để bỏ lớp, vd text-sarcasm) và thêm bootstrap theo nhóm (caption hoặc ảnh trùng) nếu --group-bootstrap."""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from vimmsd.analysis import stats as S
from vimmsd.utils.config import load_config
from vimmsd.utils.env import git_commit


def load_preds(run_dir, labels, split="val"):
    df = pd.read_csv(Path(run_dir) / f"{split}_predictions.csv", dtype={"id": str})
    idx = {lab: i for i, lab in enumerate(labels)}
    df = df.assign(y=df["label"].map(idx), p=df["pred"].map(idx)).set_index("id")
    if df[["y", "p"]].isna().any().any():
        raise ValueError(f"{run_dir}: nhãn lạ trong {split}_predictions.csv")
    return df


def aligned(runs_dir, names, seeds, labels):
    """{model: [pred theo seed]}, y, ids — cùng thứ tự id; báo lỗi nếu các run không cùng tập mẫu."""
    preds, ids, y = {}, None, None
    for name in names:
        preds[name] = []
        for s in seeds:
            df = load_preds(Path(runs_dir) / f"{name}_s{s}", labels)
            if ids is None:
                ids = sorted(df.index, key=lambda x: (len(x), x))
                y = df.loc[ids, "y"].to_numpy(int)
            elif set(df.index) != set(ids) or not np.array_equal(df.loc[ids, "y"].to_numpy(int), y):
                raise ValueError(f"{name}_s{s}: tập mẫu hoặc nhãn khác các run khác")
            preds[name].append(df.loc[ids, "p"].to_numpy(int))
    return preds, y, ids


def val_groups(cfg, ids):
    """Nhóm của các mẫu val (public test): caption chuẩn hoá hoặc md5 ảnh trùng, như fold nội bộ (mục 4.2)."""
    from vimmsd.data.denoise import file_md5
    from vimmsd.data.splits import build_groups

    src = Path(cfg.paths.data_dir)
    raw = json.loads((src / cfg.data.public_test_json).read_text(encoding="utf-8"))
    image_dir = src / cfg.data.public_test_image_dir
    groups = build_groups({k: r["caption"] for k, r in raw.items()}, {k: file_md5(image_dir / r["image"]) for k, r in raw.items()})
    return np.array([groups[i] for i in ids])


def out_dir(runs_dir):
    d = Path(runs_dir) / "_empirical" / "statistics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_contrasts(args, cfg, labels):
    pairs = [p.split(":") for p in args.pairs]
    names = sorted({n for p in pairs for n in p})
    preds, y, ids = aligned(args.runs_dir, names, args.seeds, labels)
    exclude = tuple(labels.index(x) for x in args.exclude)
    groups = val_groups(cfg, ids) if args.group_bootstrap else None
    rows = []
    for a, b in pairs:
        r = {"contrast": f"{a} - {b}",
             "a_mean": float(np.mean([S.macro_f1(y, p, len(labels), exclude) for p in preds[a]])),
             "b_mean": float(np.mean([S.macro_f1(y, p, len(labels), exclude) for p in preds[b]])),
             **S.paired_contrast(y, preds[a], preds[b], len(labels), exclude, args.n_boot, args.seed)}
        if groups is not None:
            g = S.paired_contrast(y, preds[a], preds[b], len(labels), exclude, args.n_boot, args.seed, groups)
            r["group_ci95"], r["group_p_boot"] = g["ci95"], g["p_boot"]
        rows.append(r)
    for r, adj in zip(rows, S.holm([r["p_boot"] for r in rows])):
        r["p_holm"] = adj
    per_class = {n: [S.per_class(y, p, len(labels)) for p in preds[n]] for n in names}
    result = {"family": args.family, "metric": "macro_f1" + (f" excluding {args.exclude}" if args.exclude else " (4 classes)"),
              "seeds": args.seeds, "n_examples": len(y), "n_boot": args.n_boot, "boot_seed": args.seed,
              "uncertainty": "bootstrap over evaluation examples with fixed trained models; not training randomness",
              "contrasts": rows, "per_class": per_class, **meta()}
    write(args.runs_dir, f"contrasts_{args.family}", result, contrasts_md(result, labels))


def contrasts_md(r, labels):
    lines = [f"# {r['family']}: {r['metric']}, seeds {r['seeds']}, {r['n_examples']} examples", "",
             "| contrast | A | B | A − B (mean over seeds) | per seed | 95% CI | p | p (Holm) | group 95% CI |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in r["contrasts"]:
        g = c.get("group_ci95")
        lines.append(f"| {c['contrast']} | {c['a_mean']:.4f} | {c['b_mean']:.4f} | {c['mean']:+.4f} | "
                     + " ".join(f"{x:+.3f}" for x in c["per_seed"])
                     + f" | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['p_boot']:.3f} | {c['p_holm']:.3f} | "
                     + (f"[{g[0]:+.4f}, {g[1]:+.4f}]" if g else "—") + " |")
    lines += ["", f"Uncertainty: {r['uncertainty']}."]
    return "\n".join(lines) + "\n"


def cmd_pooled(args, cfg, labels):
    rows, pooled, ids = [], {}, None
    for t in args.treatments:
        frames = [load_preds(Path(args.runs_dir) / f"{t}_f{f}_s{args.train_seed}", labels) for f in range(args.folds)]
        df = pd.concat(frames)
        if df.index.duplicated().any():
            raise ValueError(f"{t}: một mẫu xuất hiện ở nhiều fold")
        ids = ids or sorted(df.index, key=lambda x: (len(x), x))
        if set(df.index) != set(ids):
            raise ValueError(f"{t}: tập mẫu out-of-fold khác treatment đầu")
        pooled[t] = df.loc[ids]
        y, p = df["y"].to_numpy(int), df["p"].to_numpy(int)
        rows.append({"treatment": t, "n": len(df), "macro_f1": S.macro_f1(y, p, len(labels)),
                     "per_fold_macro_f1": [S.macro_f1(f["y"].to_numpy(int), f["p"].to_numpy(int), len(labels)) for f in frames],
                     "per_class": dict(zip(labels, S.per_class(y, p, len(labels))))})
    md = ["# Pooled out-of-fold (internal grouped folds)", "", "| treatment | n | macro-F1 | per fold | "
          + " | ".join(f"{lab} P/R/F1" for lab in labels) + " |", "|---" * (4 + len(labels)) + "|"]
    for r in rows:
        md.append(f"| {r['treatment']} | {r['n']} | {r['macro_f1']:.4f} | " + " ".join(f"{x:.3f}" for x in r["per_fold_macro_f1"])
                  + " | " + " | ".join(f"{c['precision']:.2f}/{c['recall']:.2f}/{c['f1']:.3f} ({c['support']})"
                                       for c in r["per_class"].values()) + " |")
    contrasts = []
    if args.baseline:
        groups = None
        if args.splits_manifest:  # bootstrap theo nhóm của fold nội bộ (caption hoặc ảnh trùng)
            g = json.loads(Path(args.splits_manifest).read_text(encoding="utf-8"))["groups"]
            groups = np.array([g[i] for i in ids])
        y = pooled[args.baseline]["y"].to_numpy(int)
        for t in args.treatments:
            if t == args.baseline:
                continue
            a_p, b_p = [pooled[t]["p"].to_numpy(int)], [pooled[args.baseline]["p"].to_numpy(int)]
            c = {"contrast": f"{t} - {args.baseline}", **S.paired_contrast(y, a_p, b_p, len(labels), (), args.n_boot, args.seed)}
            if groups is not None:
                gc = S.paired_contrast(y, a_p, b_p, len(labels), (), args.n_boot, args.seed, groups)
                c["group_ci95"], c["group_p_boot"] = gc["ci95"], gc["p_boot"]
            contrasts.append(c)
        md += ["", f"Paired bootstrap against {args.baseline} ({args.n_boot} replicates over out-of-fold examples"
               + (", groups resampled" if groups is not None else "") + "; one training seed, so training randomness is not "
               "included):", "", "| contrast | Δ macro-F1 | 95% CI | p | group 95% CI | group p |", "|---|---|---|---|---|---|"]
        for c in contrasts:
            g = c.get("group_ci95")
            md.append(f"| {c['contrast']} | {c['mean']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['p_boot']:.3f} | "
                      + (f"[{g[0]:+.4f}, {g[1]:+.4f}] | {c['group_p_boot']:.3f}" if g else "— | —") + " |")
    write(args.runs_dir, "pooled_" + "_".join(args.treatments), {"rows": rows, "contrasts": contrasts, **meta()},
          "\n".join(md) + "\n")


def subset_masks(cfg, ids):
    """Subset cố định trên val từ dữ liệu gốc (mục 6). Ngưỡng độ dài lấy từ train."""
    from vimmsd.data.denoise import normalize_caption
    from vimmsd.data.preprocessing import SOURCE_TAG_RE

    src = Path(cfg.paths.data_dir)
    d = cfg.data
    train = json.loads((src / d.train_json).read_text(encoding="utf-8"))
    val = json.loads((src / d.public_test_json).read_text(encoding="utf-8"))
    refs = json.loads((src / d.public_test_labels).read_text(encoding="utf-8"))["results"]
    image_text = json.loads((Path(cfg.paths.cache_dir) / d.image_text.file).read_text(encoding="utf-8"))["images"]
    image_dir = Path(d.public_test_image_dir).name
    cap = {i: normalize_caption(val[i]["caption"]) for i in ids}
    train_caps = {normalize_caption(r["caption"]) for r in train.values()}
    q1, q2 = np.quantile([len(r["caption"].split()) for r in train.values()], [1 / 3, 2 / 3])
    length = {i: len(val[i]["caption"].split()) for i in ids}
    labels_by_cap = {}
    for i in ids:
        labels_by_cap.setdefault(cap[i], set()).add(refs[i])
    ocr = {i: bool((image_text.get(f"{image_dir}/{val[i]['image']}") or {}).get("ocr")) for i in ids}
    m = lambda f: np.array([f(i) for i in ids])
    return {
        "all": m(lambda i: True),
        "duplicate caption with differing labels (within val)": m(lambda i: len(labels_by_cap[cap[i]]) > 1),
        f"caption length short (<= {q1:.0f} words, train tertile)": m(lambda i: length[i] <= q1),
        f"caption length medium": m(lambda i: q1 < length[i] <= q2),
        f"caption length long (> {q2:.0f} words)": m(lambda i: length[i] > q2),
        "source tag present": m(lambda i: bool(SOURCE_TAG_RE.search(val[i]["caption"]))),
        "source tag absent": m(lambda i: not SOURCE_TAG_RE.search(val[i]["caption"])),
        "OCR text present": m(lambda i: ocr[i]),
        "OCR text absent": m(lambda i: not ocr[i]),
        "caption seen in train": m(lambda i: cap[i] in train_caps),
        "caption not seen in train": m(lambda i: cap[i] not in train_caps),
    }


def cmd_subsets(args, cfg, labels):
    preds, y, ids = aligned(args.runs_dir, args.models, args.seeds, labels)
    exclude = tuple(labels.index(x) for x in args.exclude)
    rows = S.subset_table(y, preds, subset_masks(cfg, ids), len(labels), exclude)
    md = ["# Fixed val subsets (descriptive; small subsets are not evidence)", "",
          "Macro-F1 over the classes present in each subset (absent classes are not scored as 0).", "",
          "| subset | support | labels " + "/".join(x[:5] for x in labels) + " | " + " | ".join(args.models) + " |",
          "|---" * (3 + len(args.models)) + "|"]
    for r in rows:
        md.append(f"| {r['subset']} | {r['support']} | {'/'.join(map(str, r['labels']))} | "
                  + " | ".join("—" if r[m] is None else f"{r[m]:.4f}" for m in args.models) + " |")
    write(args.runs_dir, "subsets_" + "_".join(args.models), {"rows": rows, "seeds": args.seeds, **meta()}, "\n".join(md) + "\n")


def meta():
    return {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(short=False)}


def write(runs_dir, name, result, md):
    name = re.sub(r"[^\w.-]+", "_", name)[:120]
    d = out_dir(runs_dir)
    (d / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (d / f"{name}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"-> {d / name}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("contrasts", "pooled", "subsets"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/base.yaml")
        p.add_argument("--runs-dir", default=None)
        p.add_argument("--exclude", nargs="*", default=[])
        if name in ("contrasts", "subsets"):
            p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 42])
        if name == "contrasts":
            p.add_argument("--family", required=True)
            p.add_argument("--pairs", nargs="+", required=True, help="A:B, hiệu = A − B")
            p.add_argument("--n-boot", type=int, default=2000)
            p.add_argument("--seed", type=int, default=0, help="seed của bootstrap")
            p.add_argument("--group-bootstrap", action="store_true")
        if name == "pooled":
            p.add_argument("--treatments", nargs="+", required=True)
            p.add_argument("--folds", type=int, default=3)
            p.add_argument("--train-seed", type=int, default=42)
            p.add_argument("--baseline", default=None, help="treatment làm mốc cho bootstrap ghép cặp, vd empC_B0")
            p.add_argument("--splits-manifest", default="data/processed/empirical/splits/folds_k3_s42.json")
            p.add_argument("--n-boot", type=int, default=2000)
            p.add_argument("--seed", type=int, default=0)
        if name == "subsets":
            p.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.runs_dir = args.runs_dir or cfg.paths.output_dir
    labels = list(cfg.data.labels)
    {"contrasts": cmd_contrasts, "pooled": cmd_pooled, "subsets": cmd_subsets}[args.cmd](args, cfg, labels)


if __name__ == "__main__":
    main()
