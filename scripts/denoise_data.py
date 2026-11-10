"""Lọc nhiễu tập train ViMMSD và ghi bản sạch, tự chứa, cùng bố cục với ViMMSD gốc (src/vimmsd/data/denoise.py).

    VIMMSD_ENV=server python scripts/denoise_data.py --config configs/base.yaml --out-dir /data/Phat/vimmsd-clear
    ... --override denoise.knn_agree_max=0.1     # tham số khác xem denoise.DEFAULTS, ghi hết vào manifest.json
    ... --drop-test-overlap                      # bỏ khỏi train mẫu trùng chính xác với public/private test (mặc định chỉ gắn cờ)

Nguồn là paths.data_dir của config (chỉ đọc). Chỉ lọc train; public/private test và file *-ref.json chép nguyên.
Bước 1: trùng chính xác (md5 ảnh + caption chuẩn hoá). Bước 2: gần trùng (pHash và CLIP cho ảnh, e5 cho caption, cần cả
ảnh lẫn caption gần) và group_id cho StratifiedGroupKFold. Bước 3: nghi sai nhãn (cleanlab trên xác suất out-of-fold của
logistic regression, cộng đồng thuận kNN). Ảnh được hard link (cùng filesystem) hoặc chép, không dùng symlink.
Embedding lưu ở <out>/noise_report/embeddings/ và được dùng lại khi chạy lại với cùng model."""
import argparse
import json
import logging
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vimmsd.data import denoise as D
from vimmsd.data.dataset import load_label_file
from vimmsd.utils.config import load_config
from vimmsd.utils.env import git_commit

logger = logging.getLogger("denoise")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--drop-test-overlap", action="store_true")
    parser.add_argument("--device", default=None, help="mặc định cuda nếu có")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-write", action="store_true", help="chỉ ghi noise_report, không dựng raw/")
    return parser.parse_args()


# ---------------------------------------------------------------- dữ liệu nguồn

def load_split(cfg, src, split):
    d = cfg.data
    json_key, img_key, ref_key = {
        "train": ("train_json", "train_image_dir", None),
        "public_test": ("public_test_json", "public_test_image_dir", "public_test_labels"),
        "private_test": ("private_test_json", "private_test_image_dir", "private_test_labels"),
    }[split]
    raw = json.loads((src / d[json_key]).read_text(encoding="utf-8"))
    refs = load_label_file(src / d[ref_key]) if ref_key and d.get(ref_key) else {}
    image_dir = src / d[img_key]
    return raw, refs, image_dir


def check_out_dir(src, out):
    src, out = src.resolve(), out.resolve()
    if out == src or src in out.parents or out in src.parents:
        raise SystemExit(f"--out-dir {out} trùng hoặc lồng với thư mục gốc {src}: không ghi đè dữ liệu gốc")


# ---------------------------------------------------------------- băm và nhúng (có cache)

def cached_json(path, compute):
    if path.exists():
        return json.loads(path.read_text())
    value = compute()
    path.write_text(json.dumps(value))
    return value


def compute_hashes(emb_dir, splits):
    """md5 cho ảnh của mọi split và pHash cho ảnh train, theo image_key (<thư mục ảnh>/<tên file>)."""
    def md5_all():
        out = {}
        for name, (raw, _, image_dir) in splits.items():
            for r in tqdm(raw.values(), desc=f"md5 {name}", leave=False):
                out[f"{image_dir.name}/{r['image']}"] = D.file_md5(image_dir / r["image"])
        return out

    def phash_train():
        raw, _, image_dir = splits["train"]
        return {f"{image_dir.name}/{r['image']}": format(D.phash_int(image_dir / r["image"]), "016x")
                for r in tqdm(raw.values(), desc="phash", leave=False)}

    return cached_json(emb_dir / "md5.json", md5_all), cached_json(emb_dir / "phash_train.json", phash_train)


def compute_embeddings(emb_dir, ids, paths, captions, p, device, batch_size):
    meta_path = emb_dir / "embeddings_meta.json"
    meta = {"ids_file": "train_ids.json", "image_model": p["image_model"], "text_model": p["text_model"],
            "text_prefix": p["text_prefix"], "text_max_length": p["text_max_length"], "image_resize": "pad"}
    ids_path = emb_dir / "train_ids.json"
    img_path, txt_path = emb_dir / "clip_image.npy", emb_dir / "e5_caption.npy"
    if (meta_path.exists() and ids_path.exists() and img_path.exists() and txt_path.exists()
            and json.loads(meta_path.read_text()) == meta and json.loads(ids_path.read_text()) == ids):
        logger.info("dùng lại embedding trong %s", emb_dir)
        return np.load(img_path), np.load(txt_path)
    img = D.clip_image_embeddings(paths, p["image_model"], device, batch_size)
    txt = D.text_embeddings(captions, p["text_model"], device, p["text_prefix"], p["text_max_length"], batch_size)
    np.save(img_path, img.astype(np.float32))
    np.save(txt_path, txt.astype(np.float32))
    ids_path.write_text(json.dumps(ids))
    meta_path.write_text(json.dumps(meta, indent=2))
    return img, txt


# ---------------------------------------------------------------- bước 3

def mislabel_signals(feats, y, group_codes, n_labels, p):
    from cleanlab.filter import find_label_issues
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=p["cv_folds"], shuffle=True, random_state=p["seed"])
    probs = np.zeros((len(y), n_labels))
    fold_sizes = []
    for train_idx, test_idx in cv.split(feats, y, group_codes):
        clf = LogisticRegression(C=p["lr_C"], class_weight="balanced", max_iter=p["lr_max_iter"])
        clf.fit(feats[train_idx], y[train_idx])
        probs[np.ix_(test_idx, clf.classes_)] = clf.predict_proba(feats[test_idx])
        fold_sizes.append({"n": int(len(test_idx)), "labels": np.bincount(y[test_idx], minlength=n_labels).tolist()})
    pred = probs.argmax(1)
    macro = float(f1_score(y, pred, average="macro"))
    per_class = f1_score(y, pred, average=None, labels=list(range(n_labels))).tolist()
    issues = find_label_issues(labels=y, pred_probs=probs, filter_by="prune_by_noise_rate")
    self_conf = probs[np.arange(len(y)), y]
    knn = D.knn_agreement(feats, y, group_codes, p["knn_k"])
    return {"probs": probs, "pred": pred, "macro_f1": macro, "per_class_f1": per_class, "issues": issues,
            "self_confidence": self_conf, "knn_agree": knn, "folds": fold_sizes}


# ---------------------------------------------------------------- ghi bản sạch

def link_or_copy(src, dst):
    """Hard link nếu cùng filesystem, không thì copy2. Không bao giờ symlink. Trả về 'link' | 'copy' | 'exists'."""
    if dst.is_symlink():
        raise RuntimeError(f"{dst} là symlink, xoá tay trước")
    if dst.exists():
        if os.path.samefile(src, dst) or dst.stat().st_size == src.stat().st_size:
            return "exists"
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_clean(cfg, src, out, splits, clean_train):
    d = cfg.data
    raw_out = out / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    (raw_out / d.train_json).write_text(json.dumps(clean_train, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(src / d.train_json, raw_out / "vimmsd-train.original.json")
    for key in ("public_test_json", "public_test_labels", "private_test_json", "private_test_labels"):
        shutil.copy2(src / d[key], raw_out / d[key])
    plan = {
        d.train_image_dir: {r["image"] for r in clean_train.values()},
        d.public_test_image_dir: None,   # None = toàn bộ thư mục
        d.private_test_image_dir: None,
    }
    for rel, wanted in plan.items():
        s_dir, o_dir = src / rel, raw_out / rel
        o_dir.mkdir(parents=True, exist_ok=True)
        names = sorted(wanted) if wanted is not None else sorted(x.name for x in s_dir.iterdir() if x.is_file())
        for name in tqdm(names, desc=f"ảnh {rel}", leave=False):
            stats[link_or_copy(s_dir / name, o_dir / name)] += 1
        if wanted is not None:  # chạy lại với bộ lọc khác: xoá ảnh train không còn dùng (chỉ trong bản sạch)
            for x in o_dir.iterdir():
                if x.name not in wanted:
                    x.unlink()
                    stats["removed_stale"] += 1
    return dict(stats)


def verify_clean(cfg, src, out, labels):
    d = cfg.data
    raw_out = out / "raw"
    problems = []
    for p in raw_out.rglob("*"):
        if p.is_symlink():
            problems.append(f"symlink {p}")
    orig_train = json.loads((src / d.train_json).read_text(encoding="utf-8"))
    for json_key, img_key, ref_key in (("train_json", "train_image_dir", None),
                                       ("public_test_json", "public_test_image_dir", "public_test_labels"),
                                       ("private_test_json", "private_test_image_dir", "private_test_labels")):
        new = json.loads((raw_out / d[json_key]).read_text(encoding="utf-8"))
        old = json.loads((src / d[json_key]).read_text(encoding="utf-8"))
        missing = [r["image"] for r in new.values() if not (raw_out / d[img_key] / r["image"]).is_file()]
        if missing:
            problems.append(f"{json_key}: thiếu {len(missing)} ảnh, vd {missing[:3]}")
        if ref_key:
            if new != old:
                problems.append(f"{json_key} khác bản gốc")
            if (raw_out / d[ref_key]).read_bytes() != (src / d[ref_key]).read_bytes():
                problems.append(f"{ref_key} khác bản gốc")
        else:
            extra = set(new) - set(orig_train)
            if extra:
                problems.append(f"train có {len(extra)} id không có trong bản gốc")
            bad = {r["label"] for r in new.values()} - set(labels)
            if bad:
                problems.append(f"train có nhãn lạ {bad}")
            changed_fields = [k for k, r in new.items() if {f: v for f, v in r.items() if f != "label"}
                              != {f: v for f, v in orig_train[k].items() if f != "label"}]
            if changed_fields:
                problems.append(f"{len(changed_fields)} mẫu train đổi image/caption")
    return problems


# ---------------------------------------------------------------- báo cáo

def dataset_report_check(train, splits, md5):
    """Tính lại số liệu trùng lặp trong reports/dataset_report.md mục 5–6 trên dữ liệu gốc."""
    out = {}
    for name, key in (("raw caption", lambda c: c), ("normalized caption", D.normalize_caption)):
        groups = D.group_by_key({k: key(r["caption"]) for k, r in train.items()})
        conflict = sum(len({train[i]["label"] for i in g}) > 1 for g in groups)
        out[name] = {"samples": sum(map(len, groups)), "groups": len(groups), "conflicting_groups": conflict}
    by_md5 = {}
    for split, (raw, _, image_dir) in splits.items():
        for k, r in raw.items():
            by_md5.setdefault(md5[f"{image_dir.name}/{r['image']}"], []).append((split, k))
    dup = [v for v in by_md5.values() if len(v) > 1]
    train_conflict = sum(len({train[k]["label"] for s, k in v if s == "train"}) > 1 for v in dup)
    cross = sum(len({s for s, _ in v}) > 1 for v in dup)
    out["md5"] = {"files": sum(map(len, dup)), "groups": len(dup), "cross_split_groups": cross,
                  "train_conflicting_groups": train_conflict}
    return out


def write_summary(path, df, labels, n_orig, clean_train, check, manifest):
    lines = ["# Lọc nhiễu tập train ViMMSD", "",
             f"Sinh bởi `scripts/denoise_data.py` lúc {manifest['timestamp']} (commit `{manifest['git_commit']}`). "
             "Tham số đầy đủ ở `manifest.json`. Chỉ lọc train; public/private test giữ nguyên.", "",
             "## Số mẫu theo action × nhãn gốc", ""]
    table = pd.crosstab(df["action"], df["label"], margins=True, margins_name="tổng").reindex(columns=labels + ["tổng"],
                                                                                                fill_value=0)
    order = [a for a in D.ACTIONS if a in table.index] + ["tổng"]
    table = table.reindex(order)
    lines += ["| action | " + " | ".join(table.columns) + " |", "|---" * (len(table.columns) + 1) + "|"]
    lines += [f"| {a} | " + " | ".join(str(int(v)) for v in row) + " |" for a, row in table.iterrows()]
    before = Counter(df["label"])
    after = Counter(r["label"] for r in clean_train.values())
    lines += ["", f"## Phân bố nhãn train: {n_orig} → {len(clean_train)} mẫu", "",
              "| nhãn | trước | % | sau | % | chênh |", "|---|---|---|---|---|---|"]
    for lab in labels:
        lines.append(f"| {lab} | {before[lab]} | {100 * before[lab] / n_orig:.1f} | {after[lab]} | "
                     f"{100 * after[lab] / len(clean_train):.1f} | {after[lab] - before[lab]:+d} |")
    s3 = manifest["step3"]
    lines += ["", "## Bước 3", "",
              f"- Out-of-fold macro-F1 (logistic regression trên CLIP ảnh ⊕ e5 caption, {manifest['params']['cv_folds']} fold "
              f"StratifiedGroupKFold): **{s3['oof_macro_f1']:.3f}**; F1 từng lớp: "
              + ", ".join(f"{lab} {v:.3f}" for lab, v in zip(labels, s3["oof_per_class_f1"])) + ".",
              f"- Luật drop_mislabel {'được áp dụng' if s3['drop_rule_applied'] else '**không áp dụng** (macro-F1 dưới ngưỡng)'}"
              f" (ngưỡng macro-F1 {manifest['params']['min_oof_macro_f1']}). cleanlab gắn cờ {s3['cleanlab_issues']} mẫu; "
              f"{s3['drop_candidates']} mẫu đủ điều kiện drop, trần {s3['max_drop']} (5% train gốc), cắt {s3['capped']}.",
              "", "## Đối chiếu với `reports/dataset_report.md`", "",
              "| số liệu | báo cáo | tính lại (caption gốc) | tính lại (caption chuẩn hoá) |", "|---|---|---|---|"]
    raw_c, norm_c, m = check["raw caption"], check["normalized caption"], check["md5"]
    lines += [f"| mẫu train trùng caption | 2.860 | {raw_c['samples']} | {norm_c['samples']} |",
              f"| nhóm caption trùng | 697 | {raw_c['groups']} | {norm_c['groups']} |",
              f"| nhóm caption trùng khác nhãn | 123 | {raw_c['conflicting_groups']} | {norm_c['conflicting_groups']} |",
              f"| nhóm ảnh trùng md5 (mọi split) | 42 (84 file) | {m['groups']} ({m['files']} file) | |",
              f"| nhóm md5 xuất hiện ở nhiều split | 20 | {m['cross_split_groups']} | |",
              f"| nhóm md5 trong train khác nhãn | 9 | {m['train_conflicting_groups']} | |", ""]
    g = manifest["groups"]
    lines += ["## Nhóm cho StratifiedGroupKFold (`groups.json`)", "",
              f"{g['n_groups']} nhóm trên {g['n_ids']} mẫu còn lại; {g['n_multi']} nhóm có ≥ 2 mẫu; nhóm lớn nhất "
              f"{g['largest']} mẫu. Cạnh: {', '.join(f'{k} {v}' for k, v in manifest['step2']['edges'].items())}.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.override)
    p = {**D.DEFAULTS, **(cfg.get("denoise") or {})}
    labels = list(cfg.data.labels)
    label2id = {lab: i for i, lab in enumerate(labels)}
    minority, droppable = set(p["minority_labels"]), set(p["droppable_labels"])
    np.random.seed(p["seed"])
    import torch

    torch.manual_seed(p["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    src = Path(cfg.paths.data_dir).resolve()
    out = Path(args.out_dir).resolve()
    check_out_dir(src, out)
    report_dir, emb_dir = out / "noise_report", out / "noise_report" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    logger.info("nguồn %s -> %s", src, out)

    splits = {s: load_split(cfg, src, s) for s in ("train", "public_test", "private_test")}
    train, _, train_dir = splits["train"]
    ids = sorted(train, key=D.id_sort_key)
    n_orig = len(ids)
    lab = {i: train[i]["label"] for i in ids}
    norm = {i: D.normalize_caption(train[i]["caption"]) for i in ids}
    ikey = {i: f"{train_dir.name}/{train[i]['image']}" for i in ids}

    md5, phash = compute_hashes(emb_dir, splits)
    rows = {i: {"id": i, "image": train[i]["image"], "label": lab[i], "new_label": lab[i], "group_id": "",
                "exact_group": "", "near_group": "", "leak_test": "", "cl_issue": False, "self_confidence": np.nan,
                "knn_agree": np.nan, "pred_label": "", "action": "keep", "reason": []} for i in ids}

    def apply(i, action, new_label, reason):
        r = rows[i]
        if reason:
            r["reason"].append(reason)
        if action == "keep":
            return
        if r["action"] in D.DROP_ACTIONS:
            return
        r["action"], r["new_label"] = action, new_label if action not in D.DROP_ACTIONS else ""

    # ---- bước 1: trùng chính xác
    exact_groups = D.group_by_key({i: (md5[ikey[i]], norm[i]) for i in ids})
    for g in exact_groups:
        for i in g:
            rows[i]["exact_group"] = f"e{g[0]}"
        for i, (action, new_label, reason) in D.resolve_group(g, lab, minority, p["majority_fraction"], "exact").items():
            apply(i, action, new_label, reason)
    test_keys = {}
    for split in ("public_test", "private_test"):
        raw, _, image_dir = splits[split]
        for k, r in raw.items():
            test_keys.setdefault((md5[f"{image_dir.name}/{r['image']}"], D.normalize_caption(r["caption"])), []).append(
                f"{split}:{k}")
    for i in ids:
        hits = test_keys.get((md5[ikey[i]], norm[i]))
        if hits:
            rows[i]["leak_test"] = ";".join(hits)
            if args.drop_test_overlap:
                apply(i, "drop_leak_test", "", f"exact duplicate of {', '.join(hits)}")
            else:
                rows[i]["reason"].append(f"exact duplicate of {', '.join(hits)} (flag only)")
    step1 = Counter(r["action"] for r in rows.values())
    logger.info("bước 1: %d nhóm trùng chính xác, %s", len(exact_groups), dict(step1))

    # ---- bước 2: gần trùng
    paths = [train_dir / train[i]["image"] for i in ids]
    img, txt = compute_embeddings(emb_dir, ids, paths, [train[i]["caption"] for i in ids], p, device, args.batch_size)
    hashes = np.array([int(phash[ikey[i]], 16) for i in ids], dtype=np.uint64)
    ph_pairs = D.hamming_pairs(hashes, p["phash_max_distance"])
    clip_pairs = D.cosine_pairs(img, p["image_cosine"])
    cap_pairs = D.cosine_pairs(txt, p["caption_cosine"])
    pos = {i: n for n, i in enumerate(ids)}
    norm_cap_groups = D.group_by_key(norm)
    md5_groups = D.group_by_key({i: md5[ikey[i]] for i in ids})
    chain = lambda groups: [(pos[g[a]], pos[g[a + 1]]) for g in groups for a in range(len(g) - 1)]

    def image_close(a, b):
        if md5[ikey[ids[a]]] == md5[ikey[ids[b]]]:
            return True
        ham = int(np.bitwise_count(hashes[a] ^ hashes[b]))
        clip = float(img[a] @ img[b])
        if p["near_image_rule"] == "phash_and_clip":
            return ham <= p["phash_max_distance"] and clip >= p["image_cosine"]
        return ham <= p["phash_max_distance"] or clip >= p["image_cosine"]

    def caption_close(a, b):
        return norm[ids[a]] == norm[ids[b]] or float(txt[a] @ txt[b]) >= p["caption_cosine"]

    candidates = {tuple(sorted(map(int, e))) for e in np.concatenate([ph_pairs, clip_pairs])} | {
        tuple(sorted(e)) for e in chain(md5_groups)}
    alive = {i for i in ids if rows[i]["action"] not in D.DROP_ACTIONS}
    near_edges = []
    for a, b in sorted(candidates):
        ia, ib = ids[a], ids[b]
        if ia not in alive or ib not in alive:
            continue
        if rows[ia]["exact_group"] and rows[ia]["exact_group"] == rows[ib]["exact_group"]:
            continue  # đã xử lý ở bước 1 (vd mẫu lớp hiếm được giữ cạnh mẫu đại diện)
        if image_close(a, b) and caption_close(a, b):
            near_edges.append((ia, ib))
    near_groups = D.components_from_pairs(sorted(alive, key=D.id_sort_key), near_edges)
    cur = {i: rows[i]["new_label"] for i in ids}
    for g in near_groups:
        for i in g:
            rows[i]["near_group"] = f"n{g[0]}"
        for i, (action, new_label, reason) in D.resolve_group(g, cur, minority, p["majority_fraction"], "near").items():
            apply(i, action, new_label, reason)
    step2 = Counter(r["action"] for r in rows.values())
    logger.info("bước 2: %d cạnh gần trùng, %d nhóm, %s", len(near_edges), len(near_groups), dict(step2))

    uf = D.UnionFind(ids)
    edge_sets = {"phash": ph_pairs.tolist(), "clip_image": clip_pairs.tolist(), "e5_caption": cap_pairs.tolist(),
                 "same_caption": chain(norm_cap_groups), "same_md5": chain(md5_groups)}
    for edges in edge_sets.values():
        for a, b in edges:
            uf.union(ids[a], ids[b])
    group_of = uf.groups()
    for i in ids:
        rows[i]["group_id"] = group_of[i]

    # ---- bước 3: nghi sai nhãn
    kept = [i for i in ids if rows[i]["action"] not in D.DROP_ACTIONS]
    kpos = np.array([pos[i] for i in kept])
    feats = np.concatenate([img[kpos], txt[kpos]], 1)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    y = np.array([label2id[rows[i]["new_label"]] for i in kept])
    gcodes = pd.factorize(pd.Series([group_of[i] for i in kept]))[0]
    sig = mislabel_signals(feats, y, gcodes, len(labels), p)
    logger.info("bước 3: OOF macro-F1 %.3f, per-class %s, cleanlab %d", sig["macro_f1"],
                np.round(sig["per_class_f1"], 3).tolist(), int(sig["issues"].sum()))
    for n, i in enumerate(kept):
        rows[i].update(cl_issue=bool(sig["issues"][n]), self_confidence=float(sig["self_confidence"][n]),
                       knn_agree=float(sig["knn_agree"][n]), pred_label=labels[sig["pred"][n]])
    apply_drop = sig["macro_f1"] >= p["min_oof_macro_f1"]
    max_drop = math.floor(p["max_drop_fraction"] * n_orig)
    knn_max = p["knn_agree_max"] if apply_drop else -1.0  # dưới ngưỡng: không mẫu nào đủ điều kiện drop
    decisions, capped = D.decide_mislabel(kept, [rows[i]["new_label"] for i in kept], sig["issues"], sig["knn_agree"],
                                          sig["self_confidence"], droppable, knn_max, max_drop)
    candidates3 = sum(1 for n in range(len(kept)) if sig["issues"][n] and sig["knn_agree"][n] <= p["knn_agree_max"]
                      and rows[kept[n]]["new_label"] in droppable)
    for i, action in decisions.items():
        r = rows[i]
        note = (f"cleanlab label issue (self_confidence {r['self_confidence']:.3f}, knn_agree {r['knn_agree']:.1f}, "
                f"pred {r['pred_label']})")
        if action == "drop_mislabel":
            apply(i, "drop_mislabel", "", note)
        elif r["action"] == "keep":
            r["action"] = "flag_mislabel"
            r["reason"].append(note)
        else:
            r["reason"].append(note + ", flag_mislabel")

    # ---- báo cáo
    df = pd.DataFrame([{**r, "reason": "; ".join(r["reason"])} for r in rows.values()])
    cols = ["id", "image", "label", "new_label", "group_id", "exact_group", "near_group", "leak_test", "cl_issue",
            "self_confidence", "knn_agree", "action", "reason", "pred_label"]
    df[cols].to_csv(report_dir / "train_noise_flags.csv", index=False)
    clean_train = {i: {**train[i], "label": rows[i]["new_label"]} for i in ids if rows[i]["action"] not in D.DROP_ACTIONS}
    groups_json = {i: group_of[i] for i in clean_train}
    (report_dir / "groups.json").write_text(json.dumps(groups_json, indent=0))

    flagged = [i for i in ids if rows[i]["action"] in ("drop_mislabel", "flag_mislabel")]
    review_ids = D.stratified_sample(flagged, lab, p["review_size"], p["seed"])
    review = df.set_index("id").loc[review_ids].reset_index()
    review.insert(2, "caption", [train[i]["caption"] for i in review_ids])
    review.insert(3, "image_path", [str(train_dir / train[i]["image"]) for i in review_ids])
    review = review[["id", "image", "image_path", "caption", "label", "pred_label", "self_confidence", "knn_agree",
                     "action", "reason"]]
    review["reviewer_label"] = ""
    review["is_mislabel"] = ""   # y / n / unsure
    review["notes"] = ""
    review.to_csv(report_dir / "review_sample.csv", index=False)

    sizes = Counter(groups_json.values())
    check = dataset_report_check(train, splits, md5)
    import cleanlab
    import imagehash
    import sklearn
    import transformers

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(short=False),
        "command": " ".join(sys.argv),
        "source": {"data_dir": str(src), "config": args.config, "overrides": args.override},
        "out_dir": str(out),
        "drop_test_overlap": args.drop_test_overlap,
        "device": device,
        "params": p,
        "versions": {"cleanlab": cleanlab.__version__, "imagehash": imagehash.__version__,
                     "scikit-learn": sklearn.__version__, "torch": torch.__version__,
                     "transformers": transformers.__version__, "numpy": np.__version__},
        "counts": {"train_before": n_orig, "train_after": len(clean_train),
                   "public_test": len(splits["public_test"][0]), "private_test": len(splits["private_test"][0]),
                   "actions": dict(Counter(df["action"])),
                   "actions_by_label": {a: dict(Counter(df.loc[df["action"] == a, "label"])) for a in df["action"].unique()},
                   "relabelled": int(sum(1 for i in clean_train if clean_train[i]["label"] != lab[i])),
                   "leak_test_flagged": int((df["leak_test"] != "").sum()),
                   "label_before": dict(Counter(lab.values())),
                   "label_after": dict(Counter(r["label"] for r in clean_train.values()))},
        "step1": {"exact_groups": len(exact_groups), "samples_in_groups": sum(map(len, exact_groups)),
                  "actions_after_step1": dict(step1)},
        "step2": {"edges": {k: len(v) for k, v in edge_sets.items()}, "near_edges": len(near_edges),
                  "near_groups": len(near_groups), "samples_in_near_groups": sum(map(len, near_groups)),
                  "actions_after_step2": dict(step2)},
        "step3": {"n_samples": len(kept), "oof_macro_f1": sig["macro_f1"],
                  "oof_per_class_f1": sig["per_class_f1"], "folds": sig["folds"],
                  "drop_rule_applied": bool(apply_drop), "cleanlab_issues": int(sig["issues"].sum()),
                  "drop_candidates": candidates3 if apply_drop else 0, "max_drop": max_drop, "capped": capped,
                  "cap_note": (f"{capped} drop candidates beyond the 5% cap were kept as flag_mislabel "
                               "(lowest self_confidence dropped first)") if capped else ""},
        "groups": {"n_ids": len(groups_json), "n_groups": len(sizes), "n_multi": sum(v > 1 for v in sizes.values()),
                   "largest": max(sizes.values())},
        "dataset_report_check": check,
        "review_sample": {"n": len(review_ids), "seed": p["seed"], "by_label": dict(Counter(lab[i] for i in review_ids))},
    }
    if not args.no_write:
        manifest["write"] = write_clean(cfg, src, out, splits, clean_train)
        problems = verify_clean(cfg, src, out, labels)
        manifest["verify"] = problems or "ok"
        if problems:
            logger.error("kiểm tra bản sạch lỗi: %s", problems)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_summary(report_dir / "summary.md", df, labels, n_orig, clean_train, check, manifest)
    logger.info("train %d -> %d; %s", n_orig, len(clean_train), dict(Counter(df["action"])))
    if not args.no_write and manifest["verify"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
