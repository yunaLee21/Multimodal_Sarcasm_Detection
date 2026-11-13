"""Experiment D, phần suy luận (docs/empirical_analysis_plan.md mục 9): sửa caption val, không train lại.

    VIMMSD_ENV=server python scripts/diagnose_cues.py --runs empA_T_s1 empA_C_s1 empA_M_s1 ...

Ba cách sửa, áp dụng riêng rẽ lên caption thô: source_tags (thẻ trang nguồn), all_tags (thẻ nguồn, mọi hashtag và dòng
chữ ký), surface (emoji và chuỗi dấu câu lặp). OCR và mô tả không đổi. Caption đã sửa được tiền xử lý lại (không ghi
vào text cache dùng chung). Báo cáo trên toàn bộ val và riêng các mẫu bị sửa, kèm support. Bản caption trước/sau ghi ở
data/processed/empirical/diagnostics/cue_edits_val.csv. Kết quả: <runs-dir>/_empirical/diagnostics/<run>/cues.json và
cues_summary.md."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from vimmsd.analysis.interventions import disruption
from vimmsd.analysis.shortcut_check import edit_captions, load_trained_run, strip_surface_signals
from vimmsd.data.dataset import build_datasets
from vimmsd.data.preprocessing import TextPreprocessor, strip_hashtags_and_signatures, strip_source_tags
from vimmsd.models.classifier import MODALITY_NEEDS
from vimmsd.training.trainer import predict
from vimmsd.utils.config import load_config
from vimmsd.utils.env import git_commit

EDITS = {"source_tags": strip_source_tags, "all_tags": strip_hashtags_and_signatures, "surface": strip_surface_signals}
EDITS_CSV = Path("data/processed/empirical/diagnostics/cue_edits_val.csv")


def probs_of(model, dataset, collator, cfg, device, use_amp):
    loader = DataLoader(dataset, batch_size=cfg.train.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    return predict(model, loader, device, use_amp=use_amp)["probs"].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runs-dir", default=None)
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir or load_config("configs/base.yaml").paths.output_dir)
    labels = list(load_config("configs/base.yaml").data.labels)
    rows = []
    for run in args.runs:
        cfg, model, collator, device, use_amp = load_trained_run(runs_dir / run)
        if cfg.data.get("caption_filter"):
            print(f"{run}: train với data.caption_filter={cfg.data.caption_filter}, val đã được sửa sẵn; vẫn chạy để đối chiếu")
        needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
        val = build_datasets(cfg, needs_text, needs_image, splits=("val",))["val"]
        y = np.array(val.labels)
        pre = TextPreprocessor.from_config(cfg.data.text)
        base = probs_of(model, val, collator, cfg, device, use_amp)
        result = {"run": run, "n": len(y), "git_commit": git_commit(short=False),
                  "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "edits": {}}
        edited_texts = {}
        for name, fn in EDITS.items():
            ds, mask = edit_captions(val, fn, pre)
            edited_texts[name] = [fn(r["caption"]) for r in val.records]
            probs = probs_of(model, ds, collator, cfg, device, use_amp)
            entry = {"n_edited": int(mask.sum()), "edited_by_class": np.bincount(y[mask], minlength=len(labels)).tolist(),
                     "all": disruption(base, probs, y, len(labels))}
            if mask.any():
                entry["edited_only"] = disruption(base[mask], probs[mask], y[mask], len(labels))
            result["edits"][name] = entry
            o, e = entry["all"]["overall"], (entry.get("edited_only") or {}).get("overall")
            rows.append((run, name, entry["n_edited"], o, e))
            print(f"{run} {name}: sửa {entry['n_edited']} mẫu; flip toàn bộ {o['flip_rate']:.3f}, trên mẫu bị sửa "
                  f"{e['flip_rate'] if e else float('nan'):.3f}; macro-F1 {o['macro_f1_before']:.4f} -> {o['macro_f1_after']:.4f}",
                  flush=True)
        out = runs_dir / "_empirical" / "diagnostics" / run
        out.mkdir(parents=True, exist_ok=True)
        (out / "cues.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        if not EDITS_CSV.exists():
            EDITS_CSV.parent.mkdir(parents=True, exist_ok=True)
            with open(EDITS_CSV, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "label", "caption", *[f"caption_{k}" for k in EDITS]])
                for n, rec in enumerate(val.records):
                    w.writerow([rec["id"], labels[rec["label"]], rec["caption"], *[edited_texts[k][n] for k in EDITS]])

    md = ["# Caption cue edits on val (inference only; OCR and descriptions unchanged)", "",
          "| run | edit | edited samples | flip rate (all) | flip rate (edited) | macro-F1 before → after (all) |",
          "|---|---|---|---|---|---|"]
    for run, name, n, o, e in rows:
        md.append(f"| {run} | {name} | {n} | {o['flip_rate']:.3f} | {e['flip_rate'] if e else float('nan'):.3f} | "
                  f"{o['macro_f1_before']:.4f} → {o['macro_f1_after']:.4f} |")
    path = runs_dir / "_empirical" / "diagnostics" / "cues_summary.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
