"""Phase 0 của docs/empirical_analysis_plan.md (mục 4.1–4.2): nhóm, fold nội bộ có nhóm, checksum và độ trùng giữa các split.

    VIMMSD_ENV=server python scripts/build_empirical_splits.py --config configs/base.yaml
    ... --out-dir data/processed/empirical/splits --k 3 --seed 42

Đọc ViMMSD gốc theo paths.data_dir (chỉ đọc). Ghi <out-dir>/folds_k<k>_s<seed>.json:
{"meta": {...checksum, quy tắc nhóm, số mẫu từng lớp trong từng fold, kết quả kiểm tra...},
 "folds": {id: fold}, "groups": {id: group_id}} và overlap.json (trùng caption/ảnh giữa train, public, private).
Thoát mã 1 nếu fold không hợp lệ: không tự đổi sang cách chia không nhóm."""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from vimmsd.data import splits as S
from vimmsd.data.dataset import load_label_file
from vimmsd.data.denoise import file_md5, id_sort_key
from vimmsd.utils.config import load_config
from vimmsd.utils.env import git_commit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--out-dir", default="data/processed/empirical/splits")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-support", nargs="*", default=["text-sarcasm=15", "image-sarcasm=100"],
                        help="nhãn=số mẫu tối thiểu mỗi fold; lớp không nêu cần >= 1")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    d = cfg.data
    src = Path(cfg.paths.data_dir).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources = {"train": ("train_json", "train_image_dir", None),
               "public_test": ("public_test_json", "public_test_image_dir", "public_test_labels"),
               "private_test": ("private_test_json", "private_test_image_dir", "private_test_labels")}
    data, checksums = {}, {}
    for name, (jk, ik, rk) in sources.items():
        raw = json.loads((src / d[jk]).read_text(encoding="utf-8"))
        refs = load_label_file(src / d[rk]) if rk else {}
        image_dir = src / d[ik]
        md5s = {k: file_md5(image_dir / r["image"]) for k, r in tqdm(raw.items(), desc=f"md5 {name}", leave=False)}
        labels = {k: (r.get("label") or refs.get(k)) for k, r in raw.items()}
        data[name] = {"captions": {k: r.get("caption", "") for k, r in raw.items()}, "md5s": md5s, "labels": labels}
        checksums[name] = {"json_sha256": S.file_sha256(src / d[jk]),
                           "labels_sha256": S.file_sha256(src / d[rk]) if rk else None,
                           "images_md5_digest": S.digest(md5s), "n": len(raw),
                           "label_counts": dict(Counter(labels.values()))}

    train = data["train"]
    groups = S.build_groups(train["captions"], train["md5s"])
    folds = S.make_group_folds(train["labels"], groups, args.k, args.seed)
    min_support = {lab: 1 for lab in d.labels}
    min_support.update({kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.min_support})
    problems = S.validate_folds(train["labels"], groups, folds, args.k, min_support)
    support = S.fold_support(train["labels"], folds, args.k)
    sizes = Counter(groups.values())

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(short=False),
        "command": " ".join(sys.argv),
        "source": str(src),
        "grouping": "union-find over train: same normalized caption (html.unescape, NFC, collapse whitespace, strip, "
                    "lowercase) OR same image md5, transitive",
        "splitter": f"sklearn StratifiedGroupKFold(n_splits={args.k}, shuffle=True, random_state={args.seed})",
        "k": args.k, "seed": args.seed, "min_support": min_support,
        "checksums": checksums,
        "groups": {"n_ids": len(groups), "n_groups": len(sizes), "n_multi": sum(v > 1 for v in sizes.values()),
                   "largest": sizes.most_common(1)[0][1],
                   "largest_groups": [{"group": g, "size": n, "labels": dict(Counter(train["labels"][i] for i in groups
                                                                                     if groups[i] == g))}
                                      for g, n in sizes.most_common(5)]},
        "fold_support": [{lab: s.get(lab, 0) for lab in d.labels} | {"total": sum(s.values())} for s in support],
        "validation": problems or "ok",
    }
    name = f"folds_k{args.k}_s{args.seed}.json"
    (out / name).write_text(json.dumps({"meta": meta, "folds": dict(sorted(folds.items(), key=lambda x: id_sort_key(x[0]))),
                                        "groups": dict(sorted(groups.items(), key=lambda x: id_sort_key(x[0])))},
                                       indent=1, ensure_ascii=False), encoding="utf-8")
    overlap = S.overlap_stats({k: {"captions": v["captions"], "md5s": v["md5s"]} for k, v in data.items()})
    (out / "overlap.json").write_text(json.dumps(overlap, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(groups)} mẫu train, {len(sizes)} nhóm ({meta['groups']['n_multi']} nhóm >= 2 mẫu, lớn nhất "
          f"{meta['groups']['largest']})")
    for f, s in enumerate(meta["fold_support"]):
        print(f"fold {f}: {s}")
    for pair, o in overlap.items():
        print(f"{pair}: {o}")
    print("kiểm tra:", meta["validation"], "->", out / name)
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
