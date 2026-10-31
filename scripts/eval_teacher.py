"""Đánh giá teacher: so nhãn teacher đoán (đa số các lần sinh) với nhãn gốc (cổng G1, docs/experiment_plan.md).

    python scripts/eval_teacher.py --raw-dir <rationales>/raw/v1 --split train --ids-file <rationales>/ids/pilot_eval.txt
    python scripts/eval_teacher.py --raw-dir <rationales>/raw/v1 --split public_test --name val_zero_shot

Ghi teacher_metrics.json (F1 từng lớp, tỉ lệ parse lỗi, độ đồng thuận giữa các lần sinh, tốc độ) và confusion matrix
vào <paths.output_dir>/cot_distill/teacher_eval/<name>/. Mẫu không đọc được output nào tính là đoán sai (không gán
mặc định not-sarcasm, vì như vậy sẽ cộng điểm cho lớp đông nhất), số mẫu này báo riêng ở unparsable_samples."""
import argparse
import json
from collections import Counter
from pathlib import Path

from vimmsd.data.rationale import load_raw, predicted_label, read_ids
from vimmsd.training.metrics import compute_metrics, format_metrics, plot_confusion_matrix
from vimmsd.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "public_test", "private_test"])
    parser.add_argument("--ids-file", default=None, help="chỉ đánh giá các id này (vd pilot_eval.txt)")
    parser.add_argument("--name", default=None, help="tên thư mục kết quả, mặc định <split>[_<tên ids-file>]")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    labels = list(cfg.data.labels)
    raw = load_raw(args.raw_dir, args.split)
    if args.ids_file:
        wanted = set(read_ids(args.ids_file))
        raw = {sid: rows for sid, rows in raw.items() if sid in wanted}
        missing = len(wanted - set(raw))
        if missing:
            print(f"CẢNH BÁO: {missing} id trong {args.ids_file} chưa có output")
    rows_all = [r for rows in raw.values() for r in rows]
    if not rows_all:
        raise SystemExit("không có output nào để đánh giá")
    gold = {sid: rows[0]["label"] for sid, rows in raw.items()}
    if any(v is None for v in gold.values()):
        raise SystemExit(f"split {args.split} không có nhãn trong output, không đánh giá được")

    # mẫu không đọc được: dự đoán = lớp "không tồn tại" để tính là sai ở mọi lớp
    preds = {sid: predicted_label(rows) for sid, rows in raw.items()}
    ids = sorted(raw)
    y_true = [labels.index(gold[s]) for s in ids]
    y_pred = [labels.index(preds[s]) if preds[s] else (labels.index(gold[s]) + 1) % len(labels) for s in ids]
    metrics = compute_metrics(y_true, y_pred, labels)

    parse_errors = sum(r.get("parsed") is None for r in rows_all)
    k = Counter(len(rows) for rows in raw.values())
    agreement = [max(Counter(r["parsed"]["label"] for r in rows if r.get("parsed")).values(), default=0) / len(rows)
                 for rows in raw.values() if len(rows) > 1]
    seconds = [r["seconds"] for r in rows_all if r.get("seconds") is not None]
    per_sample = sum(seconds) / len(seconds) if seconds else None
    summary = {
        **metrics,
        "n_samples": len(ids), "n_generations": len(rows_all), "k_distribution": dict(k),
        "parse_error_rate": parse_errors / len(rows_all),
        "unparsable_samples": sum(p is None for p in preds.values()),
        "mean_agreement": sum(agreement) / len(agreement) if agreement else None,
        "seconds_per_generation": per_sample,
        "generations_per_hour": 3600 / per_sample if per_sample else None,
        "teacher": rows_all[0].get("teacher"), "prompt_version": rows_all[0].get("prompt_version"),
    }

    name = args.name or args.split + (f"_{Path(args.ids_file).stem}" if args.ids_file else "")
    out = Path(cfg.paths.output_dir) / "cot_distill" / "teacher_eval" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "teacher_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_confusion_matrix(metrics["confusion_matrix"], labels, out / "confusion_matrix.png", title=f"teacher {name}")
    print(format_metrics(metrics))
    print(f"parse lỗi {summary['parse_error_rate']:.1%} số lần sinh | {summary['unparsable_samples']} mẫu không đọc được"
          f" | đồng thuận giữa các lần sinh {summary['mean_agreement'] or float('nan'):.2f}"
          f" | {summary['generations_per_hour'] or float('nan'):.0f} lần sinh/giờ (mỗi tiến trình)")
    print(f"đã ghi vào {out}")


if __name__ == "__main__":
    main()
