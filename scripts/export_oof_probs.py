"""Xuất xác suất out-of-fold trên train của các run B2 (configs/experiments/b2_stacking.yaml) ra một file .npz,
để scripts/denoise_data.py --oof-probs dùng thay cho logistic regression ở bước 3.

    VIMMSD_ENV=server python scripts/export_oof_probs.py --runs-dir /data/Phat/vimmsd_outputs \
        --data-dir /data/Phat/vimmsd-data --cache-dir /data/Phat/vimmsd_cache \
        --out /data/Phat/vimmsd-clear/noise_report/oof_b2.npz

--data-dir / --cache-dir thay đường dẫn trong config.json của run (thường là symlink data/server/*), để chia fold
đúng trên tập train mà run đã học. Ghi ids, labels (chỉ số lớp), label_names, probs_<model> (n, C) theo thứ tự ids,
và macro-F1 out-of-fold của từng model trong meta (JSON)."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from vimmsd.analysis.shortcut_check import load_trained_run
from vimmsd.analysis.stacking import assemble_oof
from vimmsd.data.dataset import build_datasets, fold_of
from vimmsd.models.classifier import MODALITY_NEEDS
from vimmsd.training.trainer import predict


def fold_predictions(run_dir, fold, data_dir, cache_dir):
    cfg, model, collator, device, use_amp = load_trained_run(run_dir)
    cfg = copy.deepcopy(cfg)
    fold_cfg = cfg.data.get("fold") or {}
    if fold_cfg.get("index") != fold:
        raise ValueError(f"{run_dir}: data.fold.index={fold_cfg.get('index')}, cần {fold}")
    cfg.data["fold"] = None  # lấy đủ train để chọn đúng fold bị giữ lại
    cfg.paths["data_dir"], cfg.paths["cache_dir"] = str(data_dir), str(cache_dir)
    needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
    dataset = build_datasets(cfg, needs_text, needs_image, splits=("train",))["train"]
    folds = fold_of(dataset.records, fold_cfg.get("k", 5), fold_cfg.get("seed", 42))
    subset = Subset(dataset, [i for i, r in enumerate(dataset.records) if folds[r["id"]] == fold])
    loader = DataLoader(subset, batch_size=cfg.train.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    out = predict(model, loader, device, use_amp=use_amp)
    return {"ids": out["ids"], "labels": out["labels"], "probs": out["probs"].numpy()}, folds, list(cfg.data.labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="+", default=["text", "image", "multi"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arrays, meta, ids, labels, names = {}, {"runs_dir": args.runs_dir, "data_dir": args.data_dir, "models": {}}, None, None, None
    for m in args.models:
        per_fold, folds = {}, None
        for f in range(args.folds):
            run = Path(args.runs_dir) / f"b2_{m}_f{f}_s{args.seed}"
            per_fold[f], folds, names = fold_predictions(run, f, args.data_dir, args.cache_dir)
            print(f"{m} fold {f}: {len(per_fold[f]['ids'])} mẫu", flush=True)
        run_ids = sorted((s for p in per_fold.values() for s in p["ids"]), key=lambda s: (len(s), s))
        y = dict(zip([s for p in per_fold.values() for s in p["ids"]], [v for p in per_fold.values() for v in p["labels"]]))
        if ids is None:
            ids, labels = run_ids, np.array([y[s] for s in run_ids])
        elif run_ids != ids:
            raise ValueError(f"{m}: tập id khác model đầu")
        probs = assemble_oof(per_fold, folds, ids)
        arrays[f"probs_{m}"] = probs
        macro = float(f1_score(labels, probs.argmax(1), average="macro"))
        meta["models"][m] = {"oof_macro_f1": macro}
        print(f"{m}: out-of-fold macro-F1 {macro:.4f}", flush=True)
    mean = np.mean([arrays[f"probs_{m}"] for m in args.models], axis=0)
    meta["mean_oof_macro_f1"] = float(f1_score(labels, mean.argmax(1), average="macro"))
    print(f"trung bình {args.models}: out-of-fold macro-F1 {meta['mean_oof_macro_f1']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=np.array(ids), labels=labels, label_names=np.array(names), meta=json.dumps(meta), **arrays)
    print(f"đã ghi {args.out}")


if __name__ == "__main__":
    main()
