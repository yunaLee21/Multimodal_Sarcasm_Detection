"""Chạy ma trận config × seed trong một file plan, mỗi GPU một run (thay cho cell chạy song song của notebook 04).

    python scripts/run_experiments.py --plan configs/experiments/phase4_baselines.yaml
    python scripts/run_experiments.py --plan configs/experiments/smoke.yaml --gpu-ids 0
    python scripts/run_experiments.py --plan ... --only fusion_concat --dry-run

Định dạng plan: xem vimmsd/utils/experiments.py. Run đã có đủ kết quả thì bỏ qua (--rerun để chạy lại).
--extra-eval test: chỉ chạy evaluate.py --split test cho các run đã train (bước chấm private test cuối cùng).
Log mỗi run ở <output_dir>/<tên run>/run.log; trạng thái cả plan ở <output_dir>/_plans/<plan>/status.json.
Chạy xong tự tổng hợp kết quả trên val (scripts/summarize_results.py để tổng hợp lại, kể cả test).
Chạy trong tmux trên GPU server (CODEBASE.md mục 7.3)."""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from vimmsd.utils.env import git_commit
from vimmsd.utils.experiments import build_job, expand_runs, is_done, load_plan, plan_output_dir, run_output_dir, summarize
from vimmsd.utils.gpu_queue import run_on_gpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--gpus", type=int, default=None, help="số GPU dùng (mặc định: tất cả GPU thấy được)")
    parser.add_argument("--gpu-ids", default=None, help="chỉ dùng các GPU này, vd 0,2 (server dùng chung)")
    parser.add_argument("--only", nargs="*", default=[], help="chỉ chạy run có tên chứa một trong các chuỗi này")
    parser.add_argument("--rerun", action="store_true", help="chạy lại cả run đã có kết quả")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in lệnh sẽ chạy")
    parser.add_argument("--wandb", action="store_true", help="bật wandb cho các run trong plan (wandb.enabled=true)")
    parser.add_argument("--extra-eval", nargs="*", default=[], choices=["val", "test"],
                        help="chạy thêm evaluate.py cho run đã train xong mà chưa có split này "
                             "(vd --extra-eval test ở bước đánh giá cuối)")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    os.chdir(plan["root"])  # lệnh của job gọi scripts/train.py theo đường dẫn tương đối từ gốc repo
    specs = expand_runs(plan)
    if args.wandb:
        for spec in specs:
            spec.overrides.append("wandb.enabled=true")
    out_dirs = {s.name: run_output_dir(s) for s in specs}
    selected = [s for s in specs if not args.only or any(k in s.name for k in args.only)]

    jobs, skipped = [], []
    for spec in selected:
        out_dir = out_dirs[spec.name]
        if not args.rerun and is_done(out_dir, plan["evaluate"]):
            extra = [s for s in args.extra_eval if not (out_dir / f"{s}_predictions.csv").exists()]
            if extra:
                jobs.append(build_job(spec, out_dir, extra, train=False))
            else:
                skipped.append(spec.name)
            continue
        jobs.append(build_job(spec, out_dir, [*plan["evaluate"], *args.extra_eval], note=f"plan {plan['name']}"))
    print(f"plan {plan['name']}: {len(selected)} run, bỏ qua {len(skipped)} run đã xong, chạy {len(jobs)} run")
    for name in skipped:
        print("  bỏ qua", name)

    if args.dry_run:
        for job in jobs:
            print(f"\n# {job.name} (log {job.log})")
            for cmd in job.commands:
                print("  " + " ".join(map(str, cmd)))
        return

    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else None
    if args.gpus is None and not gpu_ids:
        import torch

        args.gpus = torch.cuda.device_count()
    plans_dir = plan_output_dir(out_dirs)
    plans_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(plans_dir).free / 1e9
    print(f"GPU: {gpu_ids or args.gpus} | đĩa còn trống: {free_gb:.1f} GB | output: {plans_dir.parent}")
    if jobs and free_gb < 1:
        sys.exit("đĩa gần đầy (< 1 GB), dọn bớt outputs/ trước khi train")

    started = datetime.now()
    finished = run_on_gpus(jobs, args.gpus or 0, summary_pattern=r"INFO epoch|early stopping|^\| \d{4}-",
                           gpu_ids=gpu_ids) if jobs else []
    status = {
        "plan": plan["name"], "commit": git_commit(), "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"), "skipped": skipped,
        "runs": {j.name: {"error": j.error, "minutes": round(j.minutes or 0, 1), "log": str(j.log)} for j in finished},
    }
    status_path = plans_dir / plan["name"] / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    out, md = summarize(plan, specs, out_dirs)
    print(f"\nTổng hợp trên val ({out}):\n{md}")
    failed = [j for j in finished if j.error]
    if failed:
        sys.exit("run lỗi: " + ", ".join(f"{j.name} ({j.error}, log {j.log})" for j in failed))


if __name__ == "__main__":
    main()
