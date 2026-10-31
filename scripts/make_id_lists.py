"""Tạo danh sách id cho tập thử teacher và các lô sinh reasoning (docs/experiment_plan.md giai đoạn 1-2).

    python scripts/make_id_lists.py --config configs/base.yaml

Ghi vào <data.rationale.dir hoặc <cache_dir>/rationales>/ids/:
    pilot_dev.txt   40 mẫu train mỗi lớp, để sửa prompt
    pilot_eval.txt  30 mẫu train mỗi lớp khác pilot_dev, chạy một lần với prompt đã chốt
    batch1.txt      toàn bộ text-sarcasm + image-sarcasm (k = 5)
    batch2.txt      toàn bộ multi-sarcasm
    batch3.txt      2000 mẫu not-sarcasm ngẫu nhiên
    batch4.txt      phần not-sarcasm còn lại (chỉ chạy nếu còn thời gian)
    val.txt         toàn bộ public test (zero-shot teacher, so với baseline)
Teacher chỉ được đánh giá trên tập thử chứ không học, nên lấy từ train không gây rò rỉ."""
import argparse
import json

from vimmsd.data.dataset import rationale_dir
from vimmsd.data.rationale import make_id_lists, split_records, write_ids
from vimmsd.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-dev", type=int, default=40)
    parser.add_argument("--pilot-eval", type=int, default=30)
    parser.add_argument("--not-batch", type=int, default=2000)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    train = split_records(cfg, "train", with_ocr=False)
    lists = make_id_lists({sid: r["label_name"] for sid, r in train.items()}, args.seed,
                          args.pilot_dev, args.pilot_eval, args.not_batch)
    lists["val"] = sorted(split_records(cfg, "public_test", with_ocr=False))
    out = rationale_dir(cfg) / "ids"
    for name, ids in lists.items():
        write_ids(out / f"{name}.txt", ids)
    (out / "meta.json").write_text(json.dumps({"seed": args.seed, **{k: len(v) for k, v in lists.items()}}, indent=2))
    print("\n".join(f"{name:<11} {len(ids):>5} id" for name, ids in lists.items()))
    print(f"đã ghi vào {out}")


if __name__ == "__main__":
    main()
