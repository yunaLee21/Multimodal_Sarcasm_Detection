"""Entrypoint train: python scripts/train.py --config configs/text_only.yaml [--override train.lr_head=5e-4 ...]"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from vimmsd.data.dataset import (ViMMSDCollator, build_dataloaders, build_datasets, class_counts,
                                 stage_preprocessed_cache)
from vimmsd.models.classifier import MODALITY_NEEDS, build_model
from vimmsd.training.losses import build_training_loss
from vimmsd.training.metrics import format_metrics, plot_confusion_matrix
from vimmsd.training.trainer import Trainer
from vimmsd.utils.config import find_project_root, load_config
from vimmsd.utils.env import git_commit
from vimmsd.utils.seed import set_seed

RESULTS_HEADER = (
    "| Ngày | Config | Commit | Môi trường | Val macro-F1 | Test macro-F1 | Thời gian train (phút) | Ghi chú |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def append_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        path.write_text("# Kết quả thí nghiệm\n\n" + RESULTS_HEADER, encoding="utf-8")
    with open(path, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[], help='ví dụ: train.epochs=3 loss.name=focal')
    parser.add_argument("--note", default="", help="ghi chú thêm vào reports/results.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("train")

    cfg = load_config(args.config, args.override)
    stage_preprocessed_cache(cfg)
    set_seed(cfg.seed)
    commit = git_commit()
    output_dir = Path(cfg.paths.output_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "commit": commit}, indent=2))
    log.info("config %s | env %s | commit %s | output %s", cfg.config_path, cfg.env, commit, output_dir)

    model = build_model(cfg)
    needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
    datasets = build_datasets(cfg, needs_text, needs_image)
    aux_max_length = (cfg.data.get("image_text") or {}).get("aux_max_length") or 96
    collator = ViMMSDCollator(model.tokenizer, model.image_processor, cfg.data.max_length,
                              cfg.data.get("max_caption_length"), aux_max_length)
    loaders = build_dataloaders(cfg, datasets, collator)
    counts = class_counts(datasets["train"], len(cfg.data.labels))
    log.info("train %d | val %d | test %d | phân bố lớp train %s",
             len(datasets["train"]), len(datasets["val"]), len(datasets["test"]),
             dict(zip(cfg.data.labels, counts)))

    trainer = Trainer(model, build_training_loss(cfg, counts), loaders, cfg, output_dir)
    summary = trainer.fit()

    trainer.load_best()
    aux_note = ""
    if hasattr(trainer.model.fusion, "weights"):
        weights = trainer.model.fusion.weights()
        (output_dir / "aux_weights.json").write_text(json.dumps(weights, indent=2))
        aux_note = (f"caption_weight={weights['caption']:.1f} ocr_weight={weights['ocr']:.3f} "
                    f"description_weight={weights['description']:.3f}")
        log.info("trọng số feature: %s", aux_note)
    test_metrics = None
    # train.eval_test: false (các config CoT distillation): không chấm private test sau mỗi run, chỉ chấm một lần
    # ở bước cuối bằng run_experiments.py --extra-eval test (docs/experiment_plan.md giai đoạn 6)
    if cfg.train.get("eval_test", True):
        test_metrics, _ = trainer.evaluate("test")
        log.info("test (best checkpoint)\n%s", format_metrics(test_metrics))
        (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        plot_confusion_matrix(test_metrics["confusion_matrix"], cfg.data.labels,
                              output_dir / "confusion_matrix_test.png", title=cfg.name)

    note = args.note
    if aux_note:
        note = f"{note} | {aux_note}" if note else aux_note
    if summary.get("peak_vram_gb") is not None:
        vram = f"peak VRAM {summary['peak_vram_gb']:.2f} GB"
        note = f"{note} | {vram}" if note else vram
    row = (f"| {datetime.now():%Y-%m-%d} | `{cfg.name}` | `{commit}` | {cfg.env} | "
           f"{summary['best_val_macro_f1']:.4f} | {test_metrics['macro_f1'] if test_metrics else float('nan'):.4f} | "
           f"{summary['train_minutes']:.1f} | {note} |")
    # trên Kaggle/Colab repo là bản clone tạm, nên ghi thêm vào output_dir để copy dòng này về reports/results.md
    append_result(output_dir / "results_row.md", row)
    if cfg.env in ("local", "server"):  # repo là bản clone lâu dài, ghi thẳng vào reports/results.md
        append_result(find_project_root(args.config) / "reports" / "results.md", row)
    print("\nDòng kết quả cho reports/results.md:\n" + row)


if __name__ == "__main__":
    main()
