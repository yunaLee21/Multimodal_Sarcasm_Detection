"""Phân nhóm, rationalization, xuất file chấm, đo κ và gộp thành rationale_dataset.json
(docs/experiment_design.md mục 4, docs/experiment_plan.md giai đoạn 3-5).

    # 1. phân nhóm theo số lần teacher đoán đúng (chỉ train)
    python scripts/build_rationale_dataset.py group
    # 2. danh sách id nhóm 3 để chạy rationalization
    python scripts/build_rationale_dataset.py rationalize
    #    rồi: python scripts/generate_rationales.py --split train --ids-file <rationales>/ids/group3.txt --k 1 --with-label
    # 3. xuất file chấm: calibration.csv (50 mẫu, 2 người cùng chấm) và review/<lô>.csv
    python scripts/build_rationale_dataset.py export
    # 4. độ đồng thuận của 2 người chấm trên calibration
    python scripts/build_rationale_dataset.py kappa --a review/calibration_A.csv --b review/calibration_B.csv
    # 5. gộp kết quả chấm
    python scripts/build_rationale_dataset.py merge --reviews review/*.csv

Thư mục làm việc: <data.rationale.dir hoặc <cache_dir>/rationales>. Cột rubric (c1_text, c2_image, c3_combined,
c4_conclusion) ghi 1 = đạt, 0 = không đạt; edited_rationale để sửa reasoning; text_sarcastic / image_sarcastic sửa được."""
import argparse
import json
import random
from pathlib import Path

from vimmsd.data.dataset import rationale_dir
from vimmsd.data.rationale import (LABELS, PROMPT_VERSION, group_samples, group_stats, load_raw, merge_dataset,
                                   read_csv, review_rows, rubric_kappa, select_for_review, split_records, write_csv,
                                   write_ids)
from vimmsd.utils.config import load_config
from vimmsd.utils.env import git_commit


def _paths(cfg, args):
    root = rationale_dir(cfg)
    return {
        "root": root,
        "raw": Path(args.raw_dir) if getattr(args, "raw_dir", None) else root / "raw" / PROMPT_VERSION,
        "rationalized": root / "raw" / f"{PROMPT_VERSION}_label",
        "grouped": root / "grouped.json",
        "review": root / "review",
        "dataset": root / cfg.data.get("rationale", {}).get("file", "rationale_dataset.json"),
    }


def _load_grouped(p):
    if not p["grouped"].exists():
        raise SystemExit(f"chưa có {p['grouped']}: chạy lệnh group trước")
    return json.loads(p["grouped"].read_text(encoding="utf-8"))


def _rationalized(p):
    return load_raw(p["rationalized"], "train") if any(p["rationalized"].glob("*.jsonl")) else {}


def cmd_group(cfg, args, p):
    labels = {sid: r["label_name"] for sid, r in split_records(cfg, "train", with_ocr=False).items()}
    grouped = group_samples(load_raw(p["raw"], "train"), labels)
    p["grouped"].write_text(json.dumps(grouped, ensure_ascii=False), encoding="utf-8")
    stats = group_stats(grouped)
    (p["root"] / "group_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for label, s in stats.items():
        print(f"{label:<14} " + " ".join(f"{k}={v}" for k, v in sorted(s.items())))
    print(f"đã ghi {p['grouped']}")


def cmd_rationalize(cfg, args, p):
    grouped = _load_grouped(p)
    ids = sorted(sid for sid, g in grouped.items() if g["group"] == 3)
    path = p["root"] / "ids" / "group3.txt"
    write_ids(path, ids)
    print(f"{len(ids)} mẫu nhóm 3 -> {path}\nchạy tiếp:\n  python scripts/generate_rationales.py --split train "
          f"--ids-file {path} --k 1 --with-label --temperature 0")


def cmd_export(cfg, args, p):
    grouped = _load_grouped(p)
    rationalized = _rationalized(p)
    group3 = [sid for sid, g in grouped.items() if g["group"] == 3]
    if group3 and not rationalized:
        print(f"CẢNH BÁO: {len(group3)} mẫu nhóm 3 chưa có rationalization, cột rationale sẽ trống")
    records = split_records(cfg, "train")
    review_ids = select_for_review(grouped, args.group1_fraction, args.seed)
    rows = review_rows(grouped, review_ids, rationalized, records)
    # calibration: 50 mẫu trải đều 4 lớp và 3 nhóm, 2 người chấm độc lập để đo κ trước khi chia việc
    rng = random.Random(args.seed)
    buckets = {}
    for r in rows:
        buckets.setdefault((r["label"], r["group"]), []).append(r)
    calibration = []
    while len(calibration) < min(args.calibration, len(rows)):
        for key in sorted(buckets):
            if buckets[key] and len(calibration) < args.calibration:
                calibration.append(buckets[key].pop(rng.randrange(len(buckets[key]))))
    write_csv(p["review"] / "calibration_A.csv", calibration)
    write_csv(p["review"] / "calibration_B.csv", calibration)
    # các lô còn lại: lớp hiếm trước (chấm trước), mỗi file tối đa --batch-rows dòng
    cal_ids = {r["id"] for r in calibration}
    rest = [r for r in rows if r["id"] not in cal_ids]
    rest.sort(key=lambda r: (LABELS.index(r["label"]) if r["label"] != "not-sarcasm" else 9, r["group"], r["id"]))
    for n, start in enumerate(range(0, len(rest), args.batch_rows), 1):
        write_csv(p["review"] / f"batch{n:02d}.csv", rest[start:start + args.batch_rows])
    print(f"{len(rows)} mẫu cần chấm ({len(calibration)} calibration + {len(rest)} chia lô) -> {p['review']}")


def cmd_kappa(cfg, args, p):
    result = rubric_kappa(read_csv(args.a), read_csv(args.b))
    for c, r in result.items():
        k = "–" if r["kappa"] is None else f"{r['kappa']:.3f}"
        print(f"{c:<14} κ = {k} (n={r['n']})" + ("   < 0,6: sửa rubric" if r["kappa"] is not None and r["kappa"] < 0.6 else ""))


def cmd_merge(cfg, args, p):
    grouped = _load_grouped(p)
    rows = [row for path in sorted(args.reviews) for row in read_csv(path)]
    meta = {"prompt_version": PROMPT_VERSION, "rubric_version": args.rubric_version, "commit": git_commit(),
            "reviews": [Path(x).name for x in sorted(args.reviews)]}
    data = merge_dataset(grouped, _rationalized(p), rows, meta)
    p["dataset"].write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    by = {}
    for s in data["samples"].values():
        by.setdefault(s["label"], {}).setdefault(s["review"], 0)
        by[s["label"]][s["review"]] += 1
    for label in LABELS:
        print(f"{label:<14} " + " ".join(f"{k}={v}" for k, v in sorted(by.get(label, {}).items())))
    print(f"đã ghi {p['dataset']} (loader chỉ dùng review = pass / auto)")


def main():
    # --config / --override đặt sau tên lệnh (vd `merge --override ... --reviews ...`), vì --override nhận nhiều giá trị
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/base.yaml")
    common.add_argument("--override", nargs="*", default=[])
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("group", parents=[common])
    g.add_argument("--raw-dir", default=None, help="mặc định <rationales>/raw/<prompt_version>")
    sub.add_parser("rationalize", parents=[common])
    e = sub.add_parser("export", parents=[common])
    e.add_argument("--group1-fraction", type=float, default=0.15)
    e.add_argument("--calibration", type=int, default=50)
    e.add_argument("--batch-rows", type=int, default=200)
    e.add_argument("--seed", type=int, default=42)
    k = sub.add_parser("kappa", parents=[common])
    k.add_argument("--a", required=True)
    k.add_argument("--b", required=True)
    m = sub.add_parser("merge", parents=[common])
    m.add_argument("--reviews", nargs="+", required=True, help="các file CSV đã chấm (calibration + batch)")
    m.add_argument("--rubric-version", default="v1")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    p = _paths(cfg, args)
    p["root"].mkdir(parents=True, exist_ok=True)
    {"group": cmd_group, "rationalize": cmd_rationalize, "export": cmd_export, "kappa": cmd_kappa,
     "merge": cmd_merge}[args.cmd](cfg, args, p)


if __name__ == "__main__":
    main()
