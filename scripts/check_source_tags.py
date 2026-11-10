"""Model có dựa vào thẻ của trang nguồn trong caption không? Xoá thẻ khỏi caption val rồi đo lại, không train lại.

    python scripts/check_source_tags.py --runs distill_flat_a_s42 distill_flat_a_s1 distill_flat_a_s2

Cách xoá: source_tags (#interpool, #WMO, #voz_f33, vozer...), all_tags (thêm mọi hashtag và dòng chữ ký như "- Elsa").
Ghi <output_dir>/diagnostics/<run>/source_tags.json. Đo trên train, kèm bảng hashtag phổ biến, ở source_tags_train.csv.
Model vẫn đã học với thẻ, nên đây là mức phụ thuộc lúc suy luận, chưa phải kết quả khi train không có thẻ."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from vimmsd.analysis.experiment_report import refresh_report
from vimmsd.analysis.shortcut_check import (HASHTAG_RE, compare_predictions, edit_captions, load_trained_run, strip_hashtags_and_signatures,
                                            strip_source_tags)
from vimmsd.data.dataset import build_dataloaders, build_datasets, load_all_records
from vimmsd.data.preprocessing import TextPreprocessor
from vimmsd.models.classifier import MODALITY_NEEDS
from vimmsd.training.metrics import compute_metrics
from vimmsd.training.trainer import predict, selection_score
from vimmsd.utils.config import load_config

VARIANTS = {"source_tags": strip_source_tags, "all_tags": strip_hashtags_and_signatures}
TAGS = {"interpool": re.compile(r"interpool", re.I), "voz": re.compile(r"(?<![\w.])#?voz(?:er|_f\d+)?(?![\w.])", re.I)}


def hashtag_table(cfg):
    """Hashtag phổ biến trên train và val: số bài và tỉ lệ từng nhãn."""
    records = load_all_records(cfg)
    labels = list(cfg.data.labels)
    rows = defaultdict(lambda: {"train": Counter(), "val": Counter()})
    for split in ("train", "val"):
        for rec in records[split]:
            for tag in {t.lower() for t in HASHTAG_RE.findall(rec["caption"] or "")}:
                rows[tag][split][labels[rec["label"]]] += 1
    out = []
    for tag, c in rows.items():
        row = {"tag": tag}
        for split in ("train", "val"):
            n = sum(c[split].values())
            row[f"{split}_n"] = n
            row.update({f"{split}_{label}": c[split][label] / n if n else None for label in labels})
        out.append(row)
    return pd.DataFrame(out).sort_values("train_n", ascending=False).head(50)


def share(preds, labels):
    return {name: float((preds == i).mean()) for i, name in enumerate(labels)}


def check(run_dir):
    cfg, model, collator, device, use_amp = load_trained_run(run_dir)
    needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
    dataset = build_datasets(cfg, needs_text, needs_image, splits=("val",))["val"]
    labels = list(cfg.data.labels)
    exclude = cfg.train.get("select_exclude_labels") or []
    preprocess = TextPreprocessor.from_config(cfg.data.text)

    def run(ds):
        return predict(model, build_dataloaders(cfg, {"val": ds}, collator)["val"], device, use_amp=use_amp)

    def scores(metrics):
        return {"macro_f1": metrics["macro_f1"], "macro_f1_no_text": selection_score(metrics, "macro_f1", exclude),
                "accuracy": metrics["accuracy"], "f1": {k: v["f1"] for k, v in metrics["per_class"].items()}}

    def compared(base, other, mask=None):
        out = compare_predictions(base, other, labels, mask)
        out["metrics"] = scores(out.pop("metrics"))
        return out

    base = run(dataset)
    y, p_base = np.array(base["labels"]), np.array(base["preds"])
    result = {"run": Path(run_dir).name, "n": len(dataset), "baseline": scores(compute_metrics(y, p_base, labels)), "variants": {}, "tag_subsets": {}}
    source_preds = None
    for name, edit in VARIANTS.items():
        edited, changed = edit_captions(dataset, edit, preprocess)
        out = run(edited)
        result["variants"][name] = {"n_changed": int(changed.sum()), "all": compared(base, out), "changed_only": compared(base, out, changed)}
        if name == "source_tags":
            source_preds = np.array(out["preds"])
    captions = [r["caption"] or "" for r in dataset.records]
    for tag, pattern in TAGS.items():
        mask = np.array([bool(pattern.search(c)) for c in captions])
        if mask.any():
            result["tag_subsets"][tag] = {
                "n": int(mask.sum()), "gold_share": share(y[mask], labels), "pred_share": share(p_base[mask], labels),
                "pred_share_after": share(source_preds[mask], labels), "accuracy": float((p_base[mask] == y[mask]).mean()),
                "accuracy_after": float((source_preds[mask] == y[mask]).mean())}
    return cfg, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    root = Path(load_config("configs/base.yaml").paths.output_dir)
    cfg = None
    for name in args.runs:
        cfg, result = check(root / name)
        out = root / "diagnostics" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "source_tags.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n== {name}: bỏ text {result['baseline']['macro_f1_no_text']:.4f}, accuracy {result['baseline']['accuracy']:.4f}", flush=True)
        for variant, v in result["variants"].items():
            print(f"{variant}: {v['n_changed']} caption đổi | bỏ text {v['all']['metrics']['macro_f1_no_text']:.4f} | "
                  f"accuracy {v['all']['metrics']['accuracy']:.4f} | đổi nhãn trên mẫu bị đổi {v['changed_only']['flip_rate']:.3f} | "
                  f"dịch xác suất {v['changed_only']['mean_prob_shift']:.3f}")
        print(json.dumps(result["tag_subsets"], ensure_ascii=False))
    table = hashtag_table(cfg)
    table.to_csv(root / "diagnostics" / "source_tags_train.csv", index=False)
    print(table.head(15).round(2).to_string(index=False))
    refresh_report()


if __name__ == "__main__":
    main()
