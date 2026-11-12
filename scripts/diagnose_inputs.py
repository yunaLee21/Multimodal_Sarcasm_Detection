"""Experiment B (docs/empirical_analysis_plan.md mục 7): thay hoặc bỏ từng nguồn input trên val, không train lại.

    VIMMSD_ENV=server python scripts/diagnose_inputs.py --runs empA_C_s42 empA_M_s42 [--policy random] [--n-perm 5]

Mẫu cho lấy từ donor manifest cố định (tạo nếu chưa có, dùng chung cho mọi model và seed):
data/processed/empirical/diagnostics/donors_val_<policy>_s<seed>_p<n>.json. Mỗi run chạy các can thiệp áp dụng được
(C: pixels, joint; M: cả sáu), ghi <runs-dir>/_empirical/diagnostics/<run>/inputs_<policy>.json (theo từng hoán vị và
trung bình) và bảng tổng hợp inputs_<policy>_summary.md. Không đụng private test, không dùng để chọn model."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from vimmsd.analysis import interventions as I
from vimmsd.analysis.shortcut_check import load_trained_run
from vimmsd.data.dataset import build_datasets
from vimmsd.models.classifier import MODALITY_NEEDS
from vimmsd.training.trainer import predict
from vimmsd.utils.config import load_config
from vimmsd.utils.env import git_commit

DONOR_DIR = Path("data/processed/empirical/diagnostics")


def probs_of(model, dataset, collator, cfg, device, use_amp):
    loader = DataLoader(dataset, batch_size=cfg.train.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    out = predict(model, loader, device, use_amp=use_amp)
    return out["probs"].numpy(), list(out["ids"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--policy", choices=["random", "same_label"], default="random")
    parser.add_argument("--n-perm", type=int, default=5)
    parser.add_argument("--donor-seed", type=int, default=0)
    parser.add_argument("--interventions", nargs="*", default=list(I.INTERVENTIONS))
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir or load_config("configs/base.yaml").paths.output_dir)
    donor_path = DONOR_DIR / f"donors_val_{args.policy}_s{args.donor_seed}_p{args.n_perm}.json"
    summary_rows = []

    for run in args.runs:
        cfg, model, collator, device, use_amp = load_trained_run(runs_dir / run)
        needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
        val = build_datasets(cfg, needs_text, needs_image, splits=("val",))["val"]
        ids, y = [r["id"] for r in val.records], np.array(val.labels)
        if donor_path.exists():
            donors = json.loads(donor_path.read_text())
            if donors["ids"] != ids:
                raise ValueError(f"{run}: thứ tự mẫu val khác donor manifest {donor_path}")
        else:
            donors = I.make_donor_manifest(ids, y, args.n_perm, args.donor_seed, args.policy)
            DONOR_DIR.mkdir(parents=True, exist_ok=True)
            donor_path.write_text(json.dumps(donors))
            print(f"tạo donor manifest {donor_path}")
        n_classes = len(cfg.data.labels)
        base, base_ids = probs_of(model, val, collator, cfg, device, use_amp)
        assert base_ids == ids
        result = {"run": run, "policy": args.policy, "donor_manifest": str(donor_path), "n": len(ids),
                  "git_commit": git_commit(short=False), "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "ocr_policy": "donor OCR as cleaned against the donor's own caption", "interventions": {}}
        for kind in args.interventions:
            if not I.applicable(val, kind) or (kind in ("pixels",) and not val.load_image):
                continue
            perms = [None] if kind.startswith("remove_") else donors["permutations"]
            per_perm = []
            for perm in perms:
                probs, _ = probs_of(model, I.intervene(val, kind, perm), collator, cfg, device, use_amp)
                per_perm.append(I.disruption(base, probs, y, n_classes))
            result["interventions"][kind] = {"mean": I.average_over_permutations(per_perm), "per_permutation": per_perm}
            o = result["interventions"][kind]["mean"]["overall"]
            by = result["interventions"][kind]["mean"]["by_true_class"]
            summary_rows.append((run, kind, o, {c: (by[c] or {}).get("flip_rate") for c in by}))
            print(f"{run} {kind}: flip {o['flip_rate']:.3f} tv {o['tv_shift']:.3f} macro-F1 "
                  f"{o['macro_f1_before']:.4f} -> {o['macro_f1_after']:.4f}", flush=True)
        if not result["interventions"]:
            print(f"{run}: không có can thiệp nào áp dụng được (modality {cfg.model.modality}), bỏ qua")
            continue
        out = runs_dir / "_empirical" / "diagnostics" / run
        out.mkdir(parents=True, exist_ok=True)
        (out / f"inputs_{args.policy}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    labels = list(load_config("configs/base.yaml").data.labels)
    md = [f"# Input interventions on val ({args.policy} donors, {args.n_perm} permutations)", "",
          "Original labels need not remain valid after substitution: score changes measure disruption, not accuracy.", "",
          "| run | intervention | flip rate | TV shift | macro-F1 before → after | "
          + " | ".join(f"flip {lab}" for lab in labels) + " |", "|---" * (4 + len(labels)) + "|"]
    for run, kind, o, by in summary_rows:
        md.append(f"| {run} | {kind} | {o['flip_rate']:.3f} | {o['tv_shift']:.3f} | {o['macro_f1_before']:.4f} → "
                  f"{o['macro_f1_after']:.4f} | " + " | ".join("—" if by.get(c) is None else f"{by[c]:.3f}"
                                                              for c in range(len(labels))) + " |")
    path = runs_dir / "_empirical" / "diagnostics" / f"inputs_{args.policy}_summary.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
