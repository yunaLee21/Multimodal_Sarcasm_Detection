"""Test cho các script chạy trên GPU server: chọn môi trường, plan config × seed, tổng hợp kết quả. Không cần GPU."""
import json
import sys

import pytest

from vimmsd.utils.config import load_config
from vimmsd.utils.env import detect_env
from vimmsd.utils.experiments import (RARE, aggregate, build_job, collect_rows, expand_runs, is_done, load_plan,
                                      run_output_dir, summarize)

LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]


def test_vimmsd_env_selects_server_paths(monkeypatch):
    monkeypatch.setenv("VIMMSD_ENV", "server")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")  # VIMMSD_ENV được ưu tiên
    assert detect_env() == "server"
    cfg = load_config("configs/base.yaml")
    assert cfg.env == "server"
    assert cfg.paths.data_dir.endswith("data/server/raw") and cfg.paths.cache_dir.endswith("data/server/cache")
    monkeypatch.delenv("VIMMSD_ENV")
    assert detect_env() == "kaggle"


@pytest.fixture
def plan_dir(tmp_path, monkeypatch):
    """Repo giả: configs/a.yaml, b.yaml kế thừa base.yaml thật, output vào tmp_path/out."""
    import shutil
    from pathlib import Path

    (tmp_path / "configs").mkdir()
    shutil.copy(Path("configs/base.yaml"), tmp_path / "configs" / "base.yaml")
    for name in ("a", "b"):
        (tmp_path / "configs" / f"{name}.yaml").write_text("inherit: base.yaml\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("")
    monkeypatch.setenv("VIMMSD_ENV", "local")
    return tmp_path


def make_plan(root, body):
    path = root / "configs" / "experiments" / "p.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_expand_runs_seed_outer_and_overrides(plan_dir):
    plan = load_plan(make_plan(plan_dir, """
name: demo
seeds: [42, 1]
overrides: [train.epochs=2]
evaluate: [test]
runs:
  - configs/a.yaml
  - config: configs/b.yaml
    name: b_long
    overrides: [train.epochs=9]
"""))
    specs = expand_runs(plan)
    assert [s.name for s in specs] == ["a_s42", "b_long_s42", "a_s1", "b_long_s1"]
    b = specs[1]
    assert b.group == "b_long" and b.seed == 42
    # override riêng của run đặt sau override chung nên thắng; seed và name do plan quyết định
    cfg = load_config(b.config, b.overrides)
    assert cfg.train.epochs == 9 and cfg.seed == 42 and cfg.name == "b_long_s42"
    assert run_output_dir(b).name == "b_long_s42"


def test_plan_validation(plan_dir):
    with pytest.raises(ValueError, match="evaluate"):
        load_plan(make_plan(plan_dir, "runs: [configs/a.yaml]\nevaluate: [private_test]\n"))
    with pytest.raises(ValueError, match="trùng tên"):
        expand_runs(load_plan(make_plan(plan_dir, "runs: [configs/a.yaml, configs/a.yaml]\n")))
    with pytest.raises(FileNotFoundError):
        expand_runs(load_plan(make_plan(plan_dir, "runs: [configs/missing.yaml]\n")))
    with pytest.raises(ValueError, match="runs"):
        load_plan(make_plan(plan_dir, "name: empty\n"))


def test_repo_plans_are_valid():
    for path in ("configs/experiments/smoke.yaml", "configs/experiments/phase4_baselines.yaml"):
        specs = expand_runs(load_plan(path))
        assert specs and len({s.name for s in specs}) == len(specs)


def test_build_job_commands_and_check(plan_dir, tmp_path):
    plan = load_plan(make_plan(plan_dir, "seeds: [7]\nevaluate: [val, test]\nruns: [configs/a.yaml]\n"))
    spec = expand_runs(plan)[0]
    out = tmp_path / "run"
    job = build_job(spec, out, plan["evaluate"], python="py")
    train, ev_val, ev_test = job.commands
    assert train[:3] == ["py", "scripts/train.py", "--config"] and "seed=7" in train and "name=a_s7" in train
    assert ev_val[ev_val.index("--split") + 1] == "val" and ev_test[ev_test.index("--split") + 1] == "test"
    assert job.log == out / "run.log"

    out.mkdir()
    assert "thiếu" in job.check() and not is_done(out, plan["evaluate"])
    for f in ("results_row.md", "val_predictions.csv", "test_predictions.csv", "last.pt"):
        (out / f).write_text("{}")
    assert job.check() is None and not (out / "last.pt").exists()
    assert is_done(out, plan["evaluate"])

    # chỉ chấm thêm (run đã train xong): không có lệnh train, log riêng
    extra = build_job(spec, out, ["test"], python="py", train=False)
    assert [c[1] for c in extra.commands] == ["scripts/evaluate.py"] and extra.log == out / "eval.log"


def fake_metrics(macro, rare_f1=0.0):
    per_class = {c: {"precision": 0.5, "recall": 0.5, "f1": (rare_f1 if c == RARE else macro), "support": 10}
                 for c in LABELS}
    return {"macro_f1": macro, "weighted_f1": macro + 0.1, "accuracy": macro + 0.2, "per_class": per_class}


def test_collect_aggregate_and_summarize(plan_dir, tmp_path):
    plan = load_plan(make_plan(plan_dir, "name: demo\nseeds: [1, 2]\nruns: [configs/a.yaml, configs/b.yaml]\n"))
    specs = expand_runs(plan)
    out_dirs = {s.name: tmp_path / "out" / s.name for s in specs}
    for s, macro in zip(specs, (0.4, 0.5, 0.6, 0.7)):  # a_s1, b_s1, a_s2, b_s2
        if s.name == "b_s2":
            continue  # run chưa xong: không có dòng, không làm hỏng bảng
        out_dirs[s.name].mkdir(parents=True)
        (out_dirs[s.name] / "best_val_metrics.json").write_text(json.dumps(fake_metrics(macro)))
        (out_dirs[s.name] / "test_metrics.json").write_text(json.dumps(fake_metrics(0.99)))

    rows = collect_rows(specs, out_dirs)
    assert [r["run"] for r in rows] == ["a_s1", "b_s1", "a_s2"]
    assert not any(k.startswith("test") for r in rows for k in r)  # mặc định không đọc private test
    assert rows[0][f"val macro-F1 (bỏ {RARE})"] == pytest.approx(0.4)

    table = {r["group"]: r for r in aggregate(rows)}
    assert table["a"]["n"] == 2 and table["a"]["val macro-F1 mean"] == pytest.approx(0.5)
    assert table["a"]["val macro-F1 std"] == pytest.approx(0.1414, abs=1e-4)
    assert table["b"]["n"] == 1 and table["b"]["val macro-F1 std"] == 0.0

    out, md = summarize(plan, specs, out_dirs)
    assert out == tmp_path / "out" / "_plans" / "demo"
    assert (out / "runs.csv").exists() and (out / "summary.csv").exists()
    assert "| a | 2 | 0.5000 ± 0.1414" in (out / "summary.md").read_text(encoding="utf-8") and "test" not in md
    _, md_test = summarize(plan, specs, out_dirs, with_test=True)
    assert "test macro-F1" in md_test and "0.9900" in md_test


def test_run_on_gpus_uses_given_gpu_ids(tmp_path):
    from vimmsd.utils.gpu_queue import Job, run_on_gpus

    show = "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"
    jobs = [Job(n, [[sys.executable, "-c", show]], tmp_path / f"{n}.log") for n in "abc"]
    done = run_on_gpus(jobs, num_gpus=8, gpu_ids=[3, 5])
    assert all(j.error is None for j in done)
    assert {(tmp_path / f"{n}.log").read_text().split()[-1] for n in "abc"} <= {"3", "5"}


def test_load_plan_refresh_report_switch(tmp_path):
    from vimmsd.utils.experiments import load_plan

    base = "name: p\nruns: [configs/text_only_frozen.yaml]\n"
    (tmp_path / "a.yaml").write_text(base)
    (tmp_path / "b.yaml").write_text(base + "refresh_report: false\n")
    assert load_plan(tmp_path / "a.yaml")["refresh_report"] is True
    assert load_plan(tmp_path / "b.yaml")["refresh_report"] is False
