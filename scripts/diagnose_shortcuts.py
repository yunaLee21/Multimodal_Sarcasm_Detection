"""D1: kiểm tra shortcut trên val cho các run đã train (docs/experiment_design.md mục 6.6). Không train lại.

    python scripts/diagnose_shortcuts.py --runs distill_flat_a_s42 distill_flat_a_s1 distill_flat_a_s2

Gồm: đổi ảnh (cùng lúc ảnh, OCR, mô tả), xoá emoji và dấu câu lặp khỏi caption, Integrated Gradients trên caption.
So sánh text-only lấy từ kết quả plan phase4_baselines, script này không train. Ghi vào
<output_dir>/diagnostics/<run>/ (diagnostics.json, top_words.csv) và <output_dir>/diagnostics/summary.md.
Không dùng kết quả để chọn model, và không đụng private test."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from vimmsd.analysis.shortcut_check import (compare_predictions, derangement, integrated_gradients, rank_words,
                                            stratified_indices, swap_image_views, without_surface_signals)
from vimmsd.data.dataset import ViMMSDCollator, build_dataloaders, build_datasets
from vimmsd.data.preprocessing import TextPreprocessor
from vimmsd.models.classifier import MODALITY_NEEDS, build_model
from vimmsd.training.metrics import compute_metrics
from vimmsd.training.trainer import move_to_device, predict, selection_score
from vimmsd.utils.checkpoint import load_checkpoint
from vimmsd.utils.config import _to_config, load_config

logger = logging.getLogger(__name__)


def diagnose(run_dir, args):
    cfg = _to_config(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    load_checkpoint(run_dir / "best.pt", model, map_location=device)
    model.to(device).eval()

    needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
    dataset = build_datasets(cfg, needs_text, needs_image, splits=("val",))["val"]
    aux_max_length = (cfg.data.get("image_text") or {}).get("aux_max_length") or 96
    collator = ViMMSDCollator(model.tokenizer, model.image_processor, cfg.data.max_length,
                              cfg.data.get("max_caption_length"), aux_max_length)
    use_amp = cfg.train.amp and device.type == "cuda"
    labels = list(cfg.data.labels)
    exclude = cfg.train.get("select_exclude_labels") or []

    def run(ds):
        return predict(model, build_dataloaders(cfg, {"val": ds}, collator)["val"], device, use_amp=use_amp)

    def scores(metrics):
        return {"macro_f1": metrics["macro_f1"], "macro_f1_no_text": selection_score(metrics, "macro_f1", exclude),
                "accuracy": metrics["accuracy"], "weighted_f1": metrics["weighted_f1"],
                "f1": {k: v["f1"] for k, v in metrics["per_class"].items()}}

    def compared(base, other, mask=None):
        out = compare_predictions(base, other, labels, mask)
        out["metrics"] = scores(out.pop("metrics"))
        return out

    base = run(dataset)
    base_metrics = compute_metrics(base["labels"], base["preds"], labels)
    # kiểm tra đã nạp đúng checkpoint: phải ra đúng số evaluate.py đã ghi lúc train
    saved = json.loads((run_dir / "val_metrics.json").read_text())["macro_f1"]
    if abs(base_metrics["macro_f1"] - saved) > 1e-3:
        raise RuntimeError(f"{run_dir.name}: macro-F1 {base_metrics['macro_f1']:.4f} khác val_metrics.json {saved:.4f}")
    result = {"run": run_dir.name, "n": len(dataset), "baseline": scores(base_metrics)}
    pred_counts = np.bincount(base["preds"], minlength=len(labels))
    result["baseline"]["pred_counts"] = {name: int(n) for name, n in zip(labels, pred_counts)}

    if needs_image:
        result["image_swap"] = []
        for seed in args.swap_seeds:
            perm = derangement(len(dataset), np.random.default_rng(seed))
            result["image_swap"].append({"seed": seed, **compared(base, run(swap_image_views(dataset, perm)))})

    preprocess = TextPreprocessor.from_config(cfg.data.text)
    stripped, changed = without_surface_signals(dataset, preprocess)
    stripped_out = run(stripped)
    result["surface_removed"] = {"n_changed": int(changed.sum()), "all": compared(base, stripped_out),
                                 "changed_only": compared(base, stripped_out, changed)}

    if args.ig_per_class:
        idx = stratified_indices(base["preds"], args.ig_per_class)
        attributions, errors = [], []
        loader = DataLoader(Subset(dataset, idx), batch_size=args.ig_batch_size, collate_fn=collator)
        for batch in tqdm(loader, desc="integrated gradients", leave=False):
            batch = move_to_device(batch, device)
            attr, target, error = integrated_gradients(model, batch, args.ig_steps)
            errors.extend(error.tolist())
            attributions.extend(zip(batch["input_ids"].tolist(), attr.tolist(), target.tolist()))
        ranked = rank_words(model.tokenizer, attributions, labels)
        result["integrated_gradients"] = {"n": len(idx), "steps": args.ig_steps,
                                          "mean_completeness_error": float(np.mean(errors)), "per_class": ranked}
    return result


def write_outputs(result, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    ig = result.get("integrated_gradients")
    if ig:
        rows = [{"predicted": name, **row} for name, c in ig["per_class"].items() for row in c["top_words"]]
        pd.DataFrame(rows).to_csv(out_dir / "top_words.csv", index=False)


def summary_markdown(results):
    lines = ["| Run | Macro-F1 | Bỏ text | Đổi ảnh: tỉ lệ đổi nhãn | Đổi ảnh: dịch xác suất | Đổi ảnh: bỏ text | "
             "Bỏ emoji/dấu câu: bỏ text | Bỏ emoji/dấu câu: tỉ lệ đổi nhãn (mẫu bị đổi) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        swaps = r.get("image_swap") or []
        def swap(fn):
            return f"{np.mean([fn(s) for s in swaps]):.3f}" if swaps else "–"
        surface = r["surface_removed"]
        lines.append(f"| `{r['run']}` | {r['baseline']['macro_f1']:.3f} | {r['baseline']['macro_f1_no_text']:.3f} | "
                     f"{swap(lambda s: s['flip_rate'])} | {swap(lambda s: s['mean_prob_shift'])} | "
                     f"{swap(lambda s: s['metrics']['macro_f1_no_text'])} | "
                     f"{surface['all']['metrics']['macro_f1_no_text']:.3f} | {surface['changed_only']['flip_rate']:.3f} |")
    return "\n".join(lines) + "\n"


def print_report(r):
    print(f"\n== {r['run']} (n={r['n']})")
    print(f"gốc: macro-F1 {r['baseline']['macro_f1']:.4f} | bỏ text {r['baseline']['macro_f1_no_text']:.4f} | "
          f"dự đoán {r['baseline']['pred_counts']}")
    for s in r.get("image_swap") or []:
        per_class = {k: round(v["flip_rate"], 3) for k, v in s["per_class"].items()}
        print(f"đổi ảnh seed {s['seed']}: đổi nhãn {s['flip_rate']:.3f} | dịch xác suất {s['mean_prob_shift']:.3f} | "
              f"bỏ text {s['metrics']['macro_f1_no_text']:.4f} | theo lớp {per_class}")
    sur = r["surface_removed"]
    print(f"bỏ emoji/dấu câu: {sur['n_changed']} caption đổi | bỏ text {sur['all']['metrics']['macro_f1_no_text']:.4f} | "
          f"đổi nhãn trên mẫu bị đổi {sur['changed_only']['flip_rate']:.3f}")
    ig = r.get("integrated_gradients")
    if ig:
        print(f"IG: {ig['n']} mẫu, sai số completeness {ig['mean_completeness_error']:.4f}")
        for name, c in ig["per_class"].items():
            words = ", ".join(w["word"] for w in c["top_words"][:12])
            print(f"  {name}: dấu câu {c['punct_share']:.1%} | {words}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="tên thư mục run trong <output_dir>")
    parser.add_argument("--swap-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ig-per-class", type=int, default=100, help="số mẫu mỗi lớp dự đoán; 0 = bỏ qua IG")
    parser.add_argument("--ig-steps", type=int, default=300)
    parser.add_argument("--ig-batch-size", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root = Path(load_config("configs/base.yaml").paths.output_dir)
    results = []
    for name in args.runs:
        result = diagnose(root / name, args)
        write_outputs(result, root / "diagnostics" / name)
        print_report(result)
        results.append(result)
    table = summary_markdown(results)
    (root / "diagnostics" / "summary.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"đã ghi vào {root / 'diagnostics'}")


if __name__ == "__main__":
    main()
