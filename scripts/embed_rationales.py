"""Mã hoá reasoning đạt rubric bằng PhoBERT đóng băng (docs/experiment_plan.md giai đoạn 5).

    python scripts/embed_rationales.py --config configs/distill_flat_d.yaml

Đọc <rationales>/rationale_dataset.json, tách từ reasoning giống caption (data.text), mã hoá bằng
model.text_encoder.name, lấy trung bình các token (bỏ padding), ghi rationale_embeddings.npy (float32, N × hidden)
và rationale_embeddings_ids.json cùng thư mục. Loss λ_rat kéo phép chiếu vector fusion về các vector này."""
import argparse
import json

import numpy as np
import torch
from tqdm.auto import tqdm

from vimmsd.data.dataset import rationale_dir
from vimmsd.data.preprocessing import TextPreprocessor
from vimmsd.data.rationale import load_rationale_dataset
from vimmsd.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distill_flat_d.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    cfg = load_config(args.config, args.override)
    r = cfg.data.get("rationale") or {}
    root = rationale_dir(cfg)
    samples, meta = load_rationale_dataset(root / r.get("file", "rationale_dataset.json"))  # mọi nhóm đạt rubric
    ids = sorted(samples)
    texts = [samples[s]["rationale"] for s in ids]
    # không truyền cache_dir: không ghi reasoning vào file cache caption của nhóm
    texts = TextPreprocessor.from_config(cfg.data.text).process_all(texts)

    name = cfg.model.text_encoder.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().to(device)
    out = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), args.batch_size), desc="embed"):
            enc = tokenizer(texts[start:start + args.batch_size], padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy())
    matrix = np.concatenate(out) if out else np.zeros((0, model.config.hidden_size), dtype=np.float32)

    emb_path = root / r.get("embeddings", "rationale_embeddings.npy")
    np.save(emb_path, matrix.astype(np.float32))
    emb_path.with_name(emb_path.stem + "_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (root / (emb_path.stem + "_meta.json")).write_text(json.dumps(
        {"model": name, "pooling": "mean", "max_length": args.max_length, "n": len(ids), "dim": matrix.shape[1],
         "dataset_meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(ids)} reasoning, {matrix.shape[1]} chiều -> {emb_path}")


if __name__ == "__main__":
    main()
