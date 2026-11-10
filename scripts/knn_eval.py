"""E1: kNN retrieval trên val cho các run đã train (docs/experiment_design.md mục 5.5, docs/experiment_plan.md).

    python scripts/knn_eval.py --runs distill_flat_a_s42 distill_flat_a_s1 distill_flat_a_s2

Theo thiết kế, model và kNN cùng hiệu chỉnh prior theo val. --no-prior: không hiệu chỉnh ở cả hai, mốc là model gốc
(ghi knn_noprior.json, grid_noprior.csv). Ghi <output_dir>/knn/<run>/knn.json và grid.csv. `best` là
điểm dò trên chính val nên lạc quan; `cv` là ước lượng ngoài mẫu bằng kiểm định chéo trong val. Không đụng private test."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from vimmsd.analysis.experiment_report import refresh_report
from vimmsd.analysis.knn import _best, collect_outputs, grid_search, grid_search_cv, nearest_neighbors, similarity_stats
from vimmsd.analysis.shortcut_check import load_trained_run
from vimmsd.data.dataset import build_datasets
from vimmsd.models.classifier import MODALITY_NEEDS
from vimmsd.models.heads import compute_prior_shift
from vimmsd.utils.config import load_config

GRID = {"k": [5, 10, 20, 50], "tau": [0.05, 0.1, 0.2], "lambda": [0.5, 0.7, 0.9, 1.0]}


def evaluate_run(run_dir, prior=True):
    cfg, model, collator, device, use_amp = load_trained_run(run_dir)
    needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
    datasets = build_datasets(cfg, needs_text, needs_image, splits=("train", "val"))
    labels = list(cfg.data.labels)
    exclude = cfg.train.get("select_exclude_labels") or []
    metric = cfg.train.select_metric

    def loader(split):  # không dùng build_dataloaders: train ở đó bị xáo
        return DataLoader(datasets[split], batch_size=cfg.train.eval_batch_size, shuffle=False, collate_fn=collator,
                          num_workers=cfg.train.num_workers, pin_memory=device.type == "cuda")

    def counts(split):
        c = Counter(datasets[split].labels)
        return [c.get(i, 0) for i in range(len(labels))]

    train_counts, val_counts = counts("train"), counts("val")
    memory = collect_outputs(model, loader("train"), device, use_amp)
    uncorrected = collect_outputs(model, loader("val"), device, use_amp)
    val = uncorrected
    if prior:
        model.set_prior_shift(compute_prior_shift(model.head_type, train_counts, val_counts, labels))
        val = collect_outputs(model, loader("val"), device, use_amp)
        model.set_prior_shift(None)
    target_counts = val_counts if prior else train_counts  # tỉ lệ prior = 1: láng giềng giữ phân bố của train

    sims, idx = nearest_neighbors(val["z"], memory["z"], max(GRID["k"]))
    neighbor_labels = memory["labels"][idx]
    args = (val["labels"], val["probs"], sims, neighbor_labels, train_counts, target_counts, labels, GRID, metric, exclude)
    rows = grid_search(*args)
    unc = grid_search(val["labels"], uncorrected["probs"], sims, neighbor_labels, train_counts, val_counts, labels,
                      {"k": [], "tau": [], "lambda": []}, metric, exclude)[0]
    return {"run": Path(run_dir).name, "prior_correction": prior, "metric": metric, "exclude": list(exclude), "uncorrected": unc, "no_retrieval": rows[0],
            "best": _best(rows), "cv": grid_search_cv(*args), "similarity": similarity_stats(sims, neighbor_labels, val["labels"])}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--no-prior", action="store_true", help="không hiệu chỉnh prior ở model và kNN")
    args = parser.parse_args()
    suffix = "_noprior" if args.no_prior else ""
    root = Path(load_config("configs/base.yaml").paths.output_dir)
    gains = []
    for name in args.runs:
        result, rows = evaluate_run(root / name, prior=not args.no_prior)
        out = root / "knn" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / f"knn{suffix}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame([{k: v for k, v in r.items() if k != "f1"} for r in rows]).to_csv(out / f"grid{suffix}.csv", index=False)
        print(f"{name}: chưa prior {result['uncorrected']['score']:.4f} | prior {result['no_retrieval']['score']:.4f} | "
              f"kNN {result['best']['score']:.4f} (k={result['best']['k']}, τ={result['best']['tau']}, λ={result['best']['lambda']}) | "
              f"CV {result['cv']['score']:.4f} | sim {result['similarity']['top1_mean']:.3f}", flush=True)
        gains.append(result["cv"]["score"] - result["no_retrieval"]["score"])
    if len(gains) > 1:
        print(f"chênh lệch CV so với λ = 1: {statistics.fmean(gains):+.4f} ± {statistics.stdev(gains):.4f}")
    refresh_report()


if __name__ == "__main__":
    main()
