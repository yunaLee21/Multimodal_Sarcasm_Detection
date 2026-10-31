"""Teacher (VLM) sinh reasoning CoT 4 bước cho từng mẫu (docs/experiment_design.md mục 3).

    # thử prompt, 1 lần sinh mỗi mẫu (greedy)
    python scripts/generate_rationales.py --split train --ids-file <rationales>/ids/pilot_dev.txt --k 1 --temperature 0
    # zero-shot trên val để so với baseline
    python scripts/generate_rationales.py --split public_test --ids-file <rationales>/ids/val.txt --k 1 --temperature 0
    # cả lô, mỗi GPU một shard (chạy trong tmux)
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_rationales.py --split train --ids-file .../batch2.txt --k 3 \\
        --num-shards 2 --shard 0
    # GPU 12 GB: thêm --load-in-4bit; k=5 thì cần --batch-size 2 (batch 4 hết VRAM)
    # rationalization nhóm 3: đưa nhãn đúng, yêu cầu giải thích
    python scripts/generate_rationales.py --split train --ids-file <rationales>/ids/group3.txt --k 1 --with-label

Output: <rationales>/raw/<prompt_version>[_label]/<split>.shard<i>of<n>.jsonl, mỗi dòng một lần sinh
{id, split, sample_idx, label (nhãn gốc, chỉ để đánh giá), with_label, output, parsed, teacher, load_in_4bit,
prompt_version, temperature, seconds}. Chạy tiếp được: mẫu đã đủ k lần sinh thì bỏ qua. Không bao giờ dùng output này để lọc
mẫu val/test (chỉ đánh giá zero-shot)."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

from vimmsd.data.dataset import rationale_dir
from vimmsd.data.image_io import open_image_rgb
from vimmsd.data.rationale import (PROMPT_VERSION, TeacherVLM, build_prompt, load_raw, parse_teacher_output,
                                   read_ids, split_records)
from vimmsd.utils.config import load_config

log = logging.getLogger("generate_rationales")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="train", choices=["train", "public_test", "private_test"])
    parser.add_argument("--ids-file", default=None, help="chỉ chạy các id trong file (mỗi dòng một id)")
    parser.add_argument("--k", type=int, default=1, help="số lần sinh mỗi mẫu (k > 1 cần --temperature > 0)")
    parser.add_argument("--temperature", type=float, default=0.7, help="0 = greedy")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--with-label", action="store_true", help="rationalization: đưa nhãn đúng vào prompt")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-pixels", type=int, default=640 * 28 * 28)
    parser.add_argument("--device-map", default="cuda", help='"auto" để chia model lớn ra nhiều GPU')
    parser.add_argument("--load-in-4bit", action="store_true", help="lượng tử hoá 4-bit (nf4) cho GPU ít VRAM")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N mẫu đầu (thử)")
    parser.add_argument("--out-dir", default=None, help="mặc định <rationales>/raw/<prompt_version>[_label]")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.k > 1 and args.temperature <= 0:
        sys.exit("k > 1 cần --temperature > 0 (greedy cho k lần sinh giống hệt nhau)")
    if args.with_label and args.split != "train":
        sys.exit("--with-label chỉ dùng cho train: đưa nhãn val/test vào prompt là rò rỉ nhãn")

    cfg = load_config(args.config, args.override)
    records = split_records(cfg, args.split)
    ids = read_ids(args.ids_file) if args.ids_file else sorted(records)
    unknown = [i for i in ids if i not in records]
    if unknown:
        sys.exit(f"{len(unknown)} id không có trong split {args.split}, vd {unknown[:3]}")
    if args.with_label and any(records[i]["label_name"] is None for i in ids):
        sys.exit("--with-label cần nhãn cho mọi mẫu")
    ids = ids[args.shard::args.num_shards]
    if args.limit:
        ids = ids[:args.limit]

    version = PROMPT_VERSION + ("_label" if args.with_label else "")
    out_dir = Path(args.out_dir) if args.out_dir else rationale_dir(cfg) / "raw" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.shard{args.shard}of{args.num_shards}.jsonl"
    done = {}
    if any(out_dir.glob("*.jsonl")):
        done = {sid: len(rows) for sid, rows in load_raw(out_dir, args.split).items()}
    todo = [i for i in ids if done.get(i, 0) < args.k]
    log.info("%s: %d mẫu, %d đã đủ %d lần sinh, còn %d | output %s",
             args.split, len(ids), len(ids) - len(todo), args.k, len(todo), out_path)
    if not todo:
        return

    teacher = TeacherVLM(args.model, max_pixels=args.max_pixels, device_map=args.device_map,
                         load_in_4bit=args.load_in_4bit)
    failed, streak = [], 0
    with open(out_path, "a", encoding="utf-8") as f, tqdm(total=len(todo), desc="teacher") as bar:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            recs = [records[i] for i in chunk]
            t0 = time.time()
            try:
                images = [open_image_rgb(r["image_path"]) for r in recs]
                prompts = [build_prompt(r["caption"], r.get("ocr", ""), r["label_name"] if args.with_label else None)
                           for r in recs]
                outputs = teacher.generate(images, prompts, k=args.k, temperature=args.temperature,
                                           top_p=args.top_p, max_new_tokens=args.max_new_tokens)
                streak = 0
            except Exception as e:  # noqa: BLE001 - một lô lỗi (ảnh hỏng, hết VRAM) không nên dừng cả lần chạy
                log.warning("lô %d mẫu lỗi: %r", len(chunk), e)
                failed += chunk
                streak += 1
                if streak >= 5:
                    sys.exit(f"5 lô lỗi liên tiếp, dừng lại (lỗi cuối {e!r}). Giảm --batch-size hoặc kiểm tra model.")
                bar.update(len(chunk))
                continue
            seconds = (time.time() - t0) / len(chunk)
            for sid, rec, outs in zip(chunk, recs, outputs):
                for j, text in enumerate(outs[:args.k - done.get(sid, 0)]):
                    f.write(json.dumps({
                        "id": sid, "split": args.split, "sample_idx": done.get(sid, 0) + j, "label": rec["label_name"],
                        "with_label": args.with_label, "output": text, "parsed": parse_teacher_output(text),
                        "teacher": args.model, "load_in_4bit": args.load_in_4bit, "prompt_version": version, "temperature": args.temperature,
                        "seconds": round(seconds, 2)}, ensure_ascii=False) + "\n")
            f.flush()
            bar.update(len(chunk))
    if failed:
        sys.exit(f"{len(failed)} mẫu lỗi, chạy lại đúng lệnh này để thử lại (mẫu đã xong được bỏ qua)")


if __name__ == "__main__":
    main()
