"""Tổng hợp kết quả các seed của một plan thành bảng trung bình ± độ lệch chuẩn.

    python scripts/summarize_results.py --plan configs/experiments/phase4_baselines.yaml
    python scripts/summarize_results.py --plan ... --with-test   # chỉ khi đã chốt checkpoint mọi run

Mặc định chỉ đọc val (best_val_metrics.json). Private test chỉ mở một lần ở bước đánh giá cuối
(docs/experiment_plan.md giai đoạn 6), nên --with-test phải bật tay.
Ghi runs.csv (mỗi run một dòng), summary.csv và summary.md vào <output_dir>/_plans/<plan>/."""
import argparse

from vimmsd.utils.experiments import expand_runs, load_plan, run_output_dir, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--with-test", action="store_true", help="thêm cột test (private test)")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    specs = expand_runs(plan)
    out_dirs = {s.name: run_output_dir(s) for s in specs}
    missing = [s.name for s in specs if not (out_dirs[s.name] / "best_val_metrics.json").exists()]
    out, md = summarize(plan, specs, out_dirs, with_test=args.with_test)
    print(md)
    print(f"\nđã ghi vào {out}")
    if missing:
        print(f"chưa có kết quả ({len(missing)} run): {', '.join(missing)}")


if __name__ == "__main__":
    main()
