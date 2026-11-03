"""Test báo cáo thí nghiệm (vimmsd/analysis/experiment_report.py). Không cần GPU, mạng, model thật."""
import json

from vimmsd.analysis import experiment_report as er

DESIGN = """# T
## 1. Động cơ
nội dung thiết kế
### 6.3. Câu hỏi
| So sánh | Câu hỏi |
|---|---|
| A với B | Có hơn không? |
### 6.4. Khác
"""
PLAN = "# P\n## Giai đoạn 0\nnội dung kế hoạch\n"
LABELS = er.LABELS


def metrics(macro=0.4, image=0.1, multi=0.8):
    per_class = {l: {"f1": {"image-sarcasm": image, "multi-sarcasm": multi}.get(l, 0.5)} for l in LABELS}
    return {"macro_f1": macro, "weighted_f1": 0.7, "accuracy": 0.7, "per_class": per_class}


def add_run(root, name, **kw):
    path = root / name
    path.mkdir(parents=True)
    (path / "best_val_metrics.json").write_text(json.dumps(metrics(**kw)))


def add_teacher(root, name, image, multi, parse=0.01):
    path = root / "cot_distill" / "teacher_eval" / name
    path.mkdir(parents=True)
    m = metrics(image=image, multi=multi)
    m.update(parse_error_rate=parse, n_samples=10, teacher="t", prompt_version="v7", seconds_per_generation=2.0)
    (path / "teacher_metrics.json").write_text(json.dumps(m))


def test_no_results_leaves_sections_blank_but_keeps_design(tmp_path):
    text = er.build_report(tmp_path, DESIGN, PLAN)
    assert "nội dung thiết kế" in text and "nội dung kế hoạch" in text
    assert "## 7. Giai đoạn 2: sinh reasoning\n\n## 8." in text
    assert "| A với B | Có hơn không? |  |" in text
    assert "| G0 | Phân cấp mềm không kém phẳng quá 1 điểm macro-F1 trên val |  |  |" in text
    assert "±" not in text and "None" not in text


def test_groups_seeds_and_checks_g0(tmp_path):
    for seed, macro in ((42, 0.40), (1, 0.42)):
        add_run(tmp_path, f"distill_flat_a_s{seed}", macro=macro)
        add_run(tmp_path, f"distill_flat_b_s{seed}", macro=macro)
        add_run(tmp_path, f"distill_hier_a_s{seed}", macro=macro - 0.02)
        add_run(tmp_path, f"distill_hier_b_s{seed}", macro=macro - 0.005)
    add_run(tmp_path, "smoke_x_s42")
    text = er.build_report(tmp_path, DESIGN, PLAN)
    assert "(a) kém 2.0 điểm: không đạt; (b) kém 0.5 điểm: đạt" in text
    assert "0.410 ± 0.014" in text and "smoke" not in text


def test_g1_compares_val_runs_against_best_baseline(tmp_path):
    add_run(tmp_path, "distill_flat_a_s42", image=0.2, multi=0.8)
    add_run(tmp_path, "distill_flat_b_s42", image=0.1, multi=0.9)
    add_teacher(tmp_path, "val_good", image=0.3, multi=0.5)
    add_teacher(tmp_path, "val_low", image=0.1, multi=0.5)
    add_teacher(tmp_path, "val_parse", image=0.3, multi=0.5, parse=0.1)
    add_teacher(tmp_path, "pilot_dev_x", image=0.9, multi=0.9)
    text = er.build_report(tmp_path, DESIGN, PLAN)
    assert "`val_good`: đạt; `val_low`: không đạt; `val_parse`: không đạt" in text
    assert "`pilot_dev_x`:" not in text and "| `pilot_dev_x` | t | v7 |" in text


def test_notes_and_kaggle_rows_are_rendered(tmp_path):
    notes = {"stage4": "kappa 0,8", "gate_G2": "đạt", "questions": {1: "Có"}}
    text = er.build_report(tmp_path, DESIGN, PLAN, notes, kaggle=[["image_only", "abc", "0.33", "0.34", "8", "x"]])
    assert "## 9. Giai đoạn 4: chấm theo rubric\n\nkappa 0,8" in text
    assert "| A với B | Có hơn không? | Có |" in text and "| `image_only`" not in text and "| image_only | abc |" in text


def test_kaggle_rows_read_results_file(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "results.md").write_text(
        "| 2026-10-30 | `a` | `c1` | kaggle | 0.3 | 0.4 | 8 | Task \\| note |\n| 2026-10-30 | `b` | `c1` | server | 0.3 | nan | 8 | y |\n",
        encoding="utf-8")
    rows = er.kaggle_rows(tmp_path)
    assert [r[0] for r in rows] == ["`a`"] and rows[0][5] == "Task \\| note"


def test_refresh_skips_outside_server(monkeypatch):
    monkeypatch.setattr(er, "write_report", lambda: (_ for _ in ()).throw(AssertionError("không được ghi")))
    assert er.refresh_report() is None  # pytest
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("VIMMSD_ENV", "kaggle")
    assert er.refresh_report() is None
