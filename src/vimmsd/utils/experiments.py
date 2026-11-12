"""Ma trận thí nghiệm config × seed: dựng job cho gpu_queue và tổng hợp kết quả (thay cho notebook 04 trên GPU server).

File plan (configs/experiments/*.yaml), đường dẫn config tính từ gốc repo:

    name: phase4_baselines          # thư mục tổng hợp: <paths.output_dir>/_plans/<name>/
    seeds: [42, 1, 2]
    overrides: [data.split_seed=42]  # áp cho mọi run, trước override riêng của run
    evaluate: [test]                 # split chạy scripts/evaluate.py sau khi train ([] = không chạy)
    refresh_report: false            # mặc định true: sinh lại reports/experiment_report.md (báo cáo CoT) khi xong
    runs:
      - configs/text_only_frozen.yaml
      - config: configs/fusion_concat_frozen.yaml
        name: concat_frozen_long     # mặc định = tên file config
        overrides: [train.epochs=30]

Mỗi run tên `<name>_s<seed>`, output ở `<paths.output_dir>/<tên run>/` giống khi gọi train.py trực tiếp."""
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from vimmsd.utils.config import _yaml_load, find_project_root, load_config
from vimmsd.utils.gpu_queue import Job

RARE = "text-sarcasm"  # 1 mẫu trong val, 7 mẫu trong test: tách riêng khi đọc macro-F1
EVAL_SPLITS = ("val", "test")


@dataclass
class RunSpec:
    name: str          # tên run, vd text_only_frozen_s42
    group: str         # tên dòng trong bảng tổng hợp (gộp các seed), vd text_only_frozen
    config: Path
    seed: int
    overrides: list = field(default_factory=list)


def load_plan(path):
    path = Path(path)
    plan = _yaml_load(path.read_text(encoding="utf-8")) or {}
    if not plan.get("runs"):
        raise ValueError(f"{path}: plan cần ít nhất một mục trong runs")
    evaluate = plan.get("evaluate") or []
    bad = set(evaluate) - set(EVAL_SPLITS)
    if bad:
        raise ValueError(f"{path}: evaluate chỉ nhận {EVAL_SPLITS}, nhận được {sorted(bad)}")
    plan.setdefault("name", path.stem)
    plan["seeds"] = list(plan.get("seeds") or [42])
    plan["overrides"] = list(plan.get("overrides") or [])
    plan["evaluate"] = list(evaluate)
    plan["refresh_report"] = bool(plan.get("refresh_report", True))   # false: không sinh lại reports/experiment_report.md
    plan["root"] = find_project_root(path)
    return plan


def expand_runs(plan):
    """Danh sách RunSpec, vòng ngoài là seed: dừng giữa chừng vẫn có đủ mọi config của seed đầu."""
    entries = []
    for entry in plan["runs"]:
        entry = {"config": entry} if isinstance(entry, str) else dict(entry)
        config = Path(entry["config"])
        config = config if config.is_absolute() else plan["root"] / config
        if not config.exists():
            raise FileNotFoundError(f"plan {plan['name']}: không thấy config {config}")
        entries.append((entry.get("name") or config.stem, config, list(entry.get("overrides") or [])))
    groups = [g for g, _, _ in entries]
    duplicated = {g for g in groups if groups.count(g) > 1}
    if duplicated:
        raise ValueError(f"plan {plan['name']}: trùng tên run {sorted(duplicated)}, đặt name riêng cho từng mục")
    return [RunSpec(f"{group}_s{seed}", group, config, seed,
                    [*plan["overrides"], *overrides, f"seed={seed}", f"name={group}_s{seed}"])
            for seed in plan["seeds"] for group, config, overrides in entries]


def run_output_dir(spec):
    cfg = load_config(spec.config, spec.overrides)
    return Path(cfg.paths.output_dir) / cfg.name


def expected_files(evaluate):
    # results_row.md: train.py ghi ở cuối run, kể cả khi train.eval_test=false (không có test_metrics.json)
    return ["results_row.md", *(f"{split}_predictions.csv" for split in evaluate)]


def is_done(out_dir, evaluate):
    return all((Path(out_dir) / f).exists() for f in expected_files(evaluate))


def build_job(spec, out_dir, evaluate, note="", python=sys.executable, train=True):
    """Job train (+ evaluate cho từng split), log ở <out_dir>/run.log. Xong thì xoá last.pt cho đỡ tốn đĩa.
    train=False: chỉ chạy evaluate (run đã train xong, vd chấm private test ở bước cuối), log ở eval.log."""
    out_dir = Path(out_dir)
    overrides = ["--override", *spec.overrides]
    commands = [[python, "scripts/train.py", "--config", str(spec.config), *overrides,
                 "--note", note or f"{spec.group} seed {spec.seed}"]] if train else []
    commands += [[python, "scripts/evaluate.py", "--config", str(spec.config), "--split", split, *overrides]
                 for split in evaluate]

    def check():
        missing = [f for f in expected_files(evaluate) if not (out_dir / f).exists()]
        if missing:
            return f"thiếu {', '.join(missing)}"
        (out_dir / "last.pt").unlink(missing_ok=True)  # chỉ cần để resume giữa run
        return None

    return Job(spec.name, commands, out_dir / ("run.log" if train else "eval.log"), check=check)


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _split_columns(prefix, metrics):
    if metrics is None:
        return {}
    per_class = metrics.get("per_class") or {}
    cols = {f"{prefix} macro-F1": metrics.get("macro_f1"),
            f"{prefix} weighted-F1": metrics.get("weighted_f1"),
            f"{prefix} acc": metrics.get("accuracy")}
    others = [m["f1"] for c, m in per_class.items() if c != RARE]
    if others:
        cols[f"{prefix} macro-F1 (bỏ {RARE})"] = statistics.fmean(others)
    cols.update({f"{prefix} F1 {c}": m["f1"] for c, m in per_class.items()})
    return cols


def collect_rows(specs, out_dirs, with_test=False):
    """Một dòng mỗi run đã có best_val_metrics.json. Mặc định không đọc test (private test chỉ xem ở bước cuối)."""
    rows = []
    for spec in specs:
        out_dir = Path(out_dirs[spec.name])
        val = _read_json(out_dir / "best_val_metrics.json")
        if val is None:
            continue
        row = {"run": spec.name, "group": spec.group, "seed": spec.seed, **_split_columns("val", val)}
        row.update(_split_columns("val+prior", _read_json(out_dir / "val_metrics_prior.json")))
        if with_test:
            row.update(_split_columns("test", _read_json(out_dir / "test_metrics.json")))
            row.update(_split_columns("test+prior", _read_json(out_dir / "test_metrics_prior.json")))
        rows.append(row)
    return rows


def aggregate(rows):
    """Gộp các seed theo group: n, trung bình và độ lệch chuẩn (mẫu) của từng cột số."""
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    table = []
    for group, items in groups.items():
        out = {"group": group, "n": len(items)}
        metrics = [k for k in items[0] if k not in ("run", "group", "seed")]
        for k in metrics:
            values = [r[k] for r in items if r.get(k) is not None]
            if values:
                out[f"{k} mean"] = statistics.fmean(values)
                out[f"{k} std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        table.append(out)
    return table


def to_markdown(table, columns):
    """Bảng markdown dạng `mean ± std` cho các cột metric trong `columns` (tên không kèm mean/std)."""
    header = "| group | n | " + " | ".join(columns) + " |"
    lines = [header, "|" + "---|" * (len(columns) + 2)]
    for row in table:
        cells = [f"{row[f'{c} mean']:.4f} ± {row[f'{c} std']:.4f}" if f"{c} mean" in row else "–" for c in columns]
        lines.append(f"| {row['group']} | {row['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(plan, specs, out_dirs, with_test=False):
    """Ghi runs.csv, summary.csv, summary.md vào <output_dir>/_plans/<plan>/, trả về (thư mục, bảng markdown)."""
    rows = collect_rows(specs, out_dirs, with_test)
    table = aggregate(rows)
    plan_dir = plan_output_dir(out_dirs)
    prefixes = ["val", "val+prior"] + (["test", "test+prior"] if with_test else [])
    columns = [f"{p} {m}" for p in prefixes for m in ("macro-F1", "weighted-F1", f"macro-F1 (bỏ {RARE})")]
    columns = [c for c in columns if any(f"{c} mean" in row for row in table)]
    md = to_markdown(table, columns)
    if rows:
        write_csv(plan_dir / plan["name"] / "runs.csv", rows)
        write_csv(plan_dir / plan["name"] / "summary.csv", table)
    (plan_dir / plan["name"]).mkdir(parents=True, exist_ok=True)
    (plan_dir / plan["name"] / "summary.md").write_text(md + "\n", encoding="utf-8")
    return plan_dir / plan["name"], md


def plan_output_dir(out_dirs):
    """Thư mục _plans/ cạnh các thư mục run (mọi run trong một plan dùng chung paths.output_dir)."""
    parents = {Path(d).parent for d in out_dirs.values()}
    if len(parents) != 1:
        raise ValueError(f"các run trong plan ghi vào nhiều output_dir khác nhau: {sorted(map(str, parents))}")
    return parents.pop() / "_plans"
