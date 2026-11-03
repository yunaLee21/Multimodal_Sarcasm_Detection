"""B2: dự đoán out-of-fold của các model T, I, M rồi gộp bằng logistic regression (docs/experiment_plan.md 7B).

    python scripts/stack_predictions.py

Cần đã train plan configs/experiments/b2_stacking.yaml. Ghi <output_dir>/b2/stacking.json. Classifier gộp học trên
dự đoán out-of-fold của train, chọn C, class_weight và việc dịch prior theo val; `cv` là ước lượng ngoài mẫu bằng
kiểm định chéo trong val. Không đụng private test."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

from vimmsd.analysis.experiment_report import refresh_report
from vimmsd.analysis.shortcut_check import load_trained_run
from vimmsd.analysis.stacking import assemble_oof, cross_validated, features, fit_candidates, prior_ratio, score
from vimmsd.data.dataset import build_datasets, fold_of
from vimmsd.models.classifier import MODALITY_NEEDS
from vimmsd.training.trainer import predict
from vimmsd.utils.config import load_config

GRID = {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}


def run_predictions(run_dir, split, fold=None):
    """Xác suất (chưa hiệu chỉnh prior) của run trên `split`; với fold != None chỉ các mẫu train thuộc fold đó."""
    cfg, model, collator, device, use_amp = load_trained_run(run_dir)
    cfg = copy.deepcopy(cfg)
    fold_cfg = cfg.data.get("fold") or {}
    cfg.data["fold"] = None  # lấy lại đủ train để chọn ra đúng fold bị giữ lại
    needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
    dataset = build_datasets(cfg, needs_text, needs_image, splits=(split,))[split]
    folds = None
    if fold is not None:
        folds = fold_of(dataset.records, fold_cfg.get("k", 5), fold_cfg.get("seed", 42))
        dataset = Subset(dataset, [i for i, r in enumerate(dataset.records) if folds[r["id"]] == fold])
    loader = DataLoader(dataset, batch_size=cfg.train.eval_batch_size, shuffle=False, collate_fn=collator,
                        num_workers=cfg.train.num_workers)
    out = predict(model, loader, device, use_amp=use_amp)
    return {"ids": out["ids"], "labels": out["labels"], "probs": out["probs"].numpy()}, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["text", "image", "multi"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="seed train của các run (hậu tố _s<seed>)")
    args = parser.parse_args()
    root = Path(load_config("configs/base.yaml").paths.output_dir)

    oof, val, cfg = {}, {}, None
    for name in args.models:
        per_fold = {}
        for f in range(args.folds):
            per_fold[f], cfg = run_predictions(root / f"b2_{name}_f{f}_s{args.seed}", "train", fold=f)
            print(f"{name} fold {f}: {len(per_fold[f]['ids'])} mẫu out-of-fold", flush=True)
        oof[name] = per_fold
        val[name], _ = run_predictions(root / f"b2_{name}_full_s{args.seed}", "val")

    labels_names = list(cfg.data.labels)
    exclude = cfg.train.get("select_exclude_labels") or []
    metric = cfg.train.select_metric
    first = oof[args.models[0]]
    train_ids = sorted(sid for pred in first.values() for sid in pred["ids"])
    train_labels = dict(zip([s for p in first.values() for s in p["ids"]], [y for p in first.values() for y in p["labels"]]))
    folds = {}
    for f, pred in first.items():
        folds.update({sid: f for sid in pred["ids"]})
    oof_probs = {m: assemble_oof({f: {"ids": p["ids"], "probs": p["probs"]} for f, p in oof[m].items()}, folds, train_ids)
                 for m in args.models}
    y_train = np.array([train_labels[s] for s in train_ids])
    val_ids = val[args.models[0]]["ids"]
    if any(val[m]["ids"] != val_ids for m in args.models):
        raise ValueError("dự đoán val của các model không cùng thứ tự")
    y_val = np.array(val[args.models[0]]["labels"])

    counts_train = Counter(y_train.tolist())
    counts_val = Counter(y_val.tolist())
    shift = prior_ratio([counts_train.get(i, 0) for i in range(len(labels_names))],
                        [counts_val.get(i, 0) for i in range(len(labels_names))])

    def shift_probs(p):
        z = np.log(np.clip(p, 1e-6, 1)) + shift
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    singles = {}
    for m in args.models:
        singles[m] = {"oof": score(y_train, oof_probs[m], labels_names, metric, exclude),
                      "val": score(y_val, val[m]["probs"], labels_names, metric, exclude),
                      "val_prior": score(y_val, shift_probs(val[m]["probs"]), labels_names, metric, exclude)}
    mean_probs = np.mean([val[m]["probs"] for m in args.models], axis=0)
    candidates = fit_candidates(features([oof_probs[m] for m in args.models]), y_train,
                                features([val[m]["probs"] for m in args.models]), len(labels_names), shift, GRID)
    best_key = max(candidates, key=lambda k: score(y_val, candidates[k], labels_names, metric, exclude)["score"])
    result = {
        "metric": metric, "exclude": list(exclude), "models": args.models, "n_train": len(train_ids), "n_val": len(val_ids),
        "single": singles,
        "average": {"val": score(y_val, mean_probs, labels_names, metric, exclude),
                    "val_prior": score(y_val, shift_probs(mean_probs), labels_names, metric, exclude)},
        "stacking_best": {"params": list(map(str, best_key)), **score(y_val, candidates[best_key], labels_names, metric, exclude)},
        "stacking_cv": cross_validated(y_val, candidates, labels_names, metric, exclude),
    }
    out = root / "b2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stacking.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for m, s in singles.items():
        print(f"{m}: out-of-fold {s['oof']['score']:.4f} | val {s['val']['score']:.4f} | val prior {s['val_prior']['score']:.4f}")
    print(f"trung bình 3 model: val {result['average']['val']['score']:.4f} | prior {result['average']['val_prior']['score']:.4f}")
    print(f"stacking: tốt nhất {result['stacking_best']['score']:.4f} {result['stacking_best']['params']} | "
          f"kiểm định chéo {result['stacking_cv']['score']:.4f}")
    print(f"đã ghi {out / 'stacking.json'}")
    refresh_report()


if __name__ == "__main__":
    main()
