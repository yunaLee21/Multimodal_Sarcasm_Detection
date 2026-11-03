"""Sinh reports/experiment_report.md từ kết quả đã có trên đĩa. Phần chưa có kết quả để trống.

Thiết kế và kế hoạch chép nguyên từ docs/experiment_design.md, docs/experiment_plan.md. Số liệu đọc từ <output_dir>.
Nhận xét, quyết định ở các cổng và kết quả không sinh ra file (chấm rubric...) lấy từ reports/experiment_report_notes.yaml.
File được ghi đè mỗi lần chạy, chỉ sửa tay file notes. Script chạy thí nghiệm gọi refresh_report() khi xong."""
import json
import logging
import os
import re
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import yaml

from vimmsd.utils.config import load_config
from vimmsd.utils.env import detect_env
from vimmsd.utils.experiments import RARE, aggregate, collect_rows

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
LABELS = ("not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm")
MATRIX_ROWS = (("Phẳng 4 lớp", "distill_flat"), ("Phân cấp mềm", "distill_hier"), ("Subclass", "distill_sub"))
MATRIX_COLUMNS = ("a", "b", "c", "d")
ABLATION_GROUPS = ("distill_flat_d_nogroup3", "distill_hier_d_nogroup3", "distill_sub_d_nogroup3")
STAGE0_GROUPS = ("distill_flat_a", "distill_flat_b", "distill_hier_a", "distill_hier_b", "hier_cascade")
TEACHER_SETS = (("pilot_dev", "prompt-dev (`pilot_dev`, 160 mẫu cân bằng)"), ("pilot_eval", "pilot-eval (`pilot_eval`, 120 mẫu)"),
                ("natural_dev", "`natural_dev` (500 mẫu train ngẫu nhiên, ngoài kế hoạch gốc)"),
                ("val", "val (public test, 1 413 mẫu)"))
RUN_NAME_RE = re.compile(r"^(?P<group>.+)_s(?P<seed>\d+)$")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_groups(output_dir):
    """Gộp theo seed mọi run <group>_s<seed> đã train trong output_dir. Trả về {group: dòng tổng hợp}."""
    output_dir = Path(output_dir)
    specs, dirs = [], {}
    for path in sorted(output_dir.iterdir()) if output_dir.exists() else []:
        m = RUN_NAME_RE.match(path.name)
        if m and path.is_dir() and not path.name.startswith("smoke"):
            specs.append(SimpleNamespace(name=path.name, group=m["group"], seed=int(m["seed"])))
            dirs[path.name] = path
    return {row["group"]: row for row in aggregate(collect_rows(specs, dirs, with_test=True))}


def mean_std(row, column):
    if not row or f"{column} mean" not in row:
        return ""
    return f"{row[f'{column} mean']:.3f} ± {row[f'{column} std']:.3f}"


def group_table(groups, names, prefix):
    columns = [f"{prefix} macro-F1", f"{prefix} macro-F1 (bỏ {RARE})", f"{prefix} weighted-F1", f"{prefix} acc",
               *(f"{prefix} F1 {label}" for label in LABELS)]
    header = ["Cấu hình", "Số seed", "Macro-F1", "Macro-F1 bỏ text", "Weighted-F1", "Accuracy", *(f"F1 `{l}`" for l in LABELS)]
    return table(header, [[f"`{name}`", (groups.get(name) or {}).get("n"), *(mean_std(groups.get(name), c) for c in columns)]
                          for name in names])


def matrix_table(groups, prefix):
    column = f"{prefix} macro-F1"
    rows = [[title, *(mean_std(groups.get(f"{stem}_{c}"), column) for c in MATRIX_COLUMNS)] for title, stem in MATRIX_ROWS]
    rows.append(["Phân cấp nối tiếp", mean_std(groups.get("hier_cascade"), column), "", "", ""])
    return table(["Cấu trúc đầu ra", "(a) CE", "(b) + nhãn phụ từ nhãn gốc", "(c) + nhãn phụ từ teacher", "(d) đầy đủ"], rows)


def g0_result(groups):
    """Phân cấp mềm không kém phẳng quá 1 điểm macro-F1 trên val, cột (a) và (b)."""
    parts = []
    for column in ("a", "b"):
        flat, hier = groups.get(f"distill_flat_{column}"), groups.get(f"distill_hier_{column}")
        if not flat or not hier:
            return ""
        gap = (flat["val macro-F1 mean"] - hier["val macro-F1 mean"]) * 100
        parts.append(f"({column}) kém {gap:.1f} điểm: {'không đạt' if gap > 1 else 'đạt'}")
    return "; ".join(parts)


def g1_bars(groups):
    """Mốc của G1: F1 cao nhất của cột (a)/(b) trên val, chưa hiệu chỉnh prior và có hiệu chỉnh prior."""
    bars = {}
    for prefix in ("val", "val+prior"):
        for label in ("image-sarcasm", "multi-sarcasm"):
            values = [(row[f"{prefix} F1 {label} mean"], name) for name, row in groups.items()
                      if name in STAGE0_GROUPS and f"{prefix} F1 {label} mean" in row]
            if values:
                bars[prefix, label] = max(values)
    return bars


def teacher_runs(output_dir):
    root = Path(output_dir) / "cot_distill" / "teacher_eval"
    return {p.name: m for p in sorted(root.iterdir()) if (m := read_json(p / "teacher_metrics.json"))} if root.exists() else {}


def teacher_table(runs):
    header = ["Lần chạy", "Teacher", "Prompt", "Số mẫu", "Accuracy", "Macro-F1", *(f"F1 `{l}`" for l in LABELS),
              "Parse lỗi", "Giây/lần sinh"]
    rows = []
    for name, m in runs.items():
        rows.append([f"`{name}`", m.get("teacher"), m.get("prompt_version"), m.get("n_samples"), fmt(m.get("accuracy")),
                     fmt(m.get("macro_f1")), *(fmt(m["per_class"][l]["f1"]) for l in LABELS),
                     f"{m['parse_error_rate']:.1%}", fmt(m.get("seconds_per_generation"), 1)])
    return table(header, rows)


def g1_rows(runs, bars):
    """Một dòng cho mỗi lần chạy trên val, so với mốc cao nhất của mỗi lớp, kể cả bản có hiệu chỉnh prior."""
    image = max((v for (_, label), v in bars.items() if label == "image-sarcasm"), default=None)
    multi = max((v for (_, label), v in bars.items() if label == "multi-sarcasm"), default=None)
    rows = []
    for name, m in runs.items():
        f_image, f_multi = m["per_class"]["image-sarcasm"]["f1"], m["per_class"]["multi-sarcasm"]["f1"]
        passed = None
        if image and multi:
            passed = (f_image > image[0] or f_multi > multi[0]) and m["parse_error_rate"] < 0.05
        rows.append([f"`{name}`", fmt(f_image), fmt(image[0]) if image else "", fmt(f_multi), fmt(multi[0]) if multi else "",
                     f"{m['parse_error_rate']:.1%}", "" if passed is None else ("đạt" if passed else "không đạt")])
    return rows


def diagnostics(output_dir):
    root = Path(output_dir) / "diagnostics"
    return {p.name: m for p in sorted(root.iterdir()) if (m := read_json(p / "diagnostics.json"))} if root.exists() else {}


def _mean(values):
    return statistics.fmean(values) if values else None


def d1_tables(results):
    if not results:
        return table(["Run", "Macro-F1 bỏ text", "Sau đổi ảnh", "Đổi nhãn khi đổi ảnh", "Dịch xác suất", "Sau khi bỏ emoji/dấu câu",
                      "Đổi nhãn trên caption bị sửa", "Sai số IG"], [])
    rows, per_class, words = [], [], []
    for name, r in results.items():
        swaps = r.get("image_swap") or []
        surface = r["surface_removed"]
        ig = r.get("integrated_gradients") or {}
        rows.append([f"`{name}`", fmt(r["baseline"]["macro_f1_no_text"]),
                     fmt(_mean([s["metrics"]["macro_f1_no_text"] for s in swaps])), fmt(_mean([s["flip_rate"] for s in swaps])),
                     fmt(_mean([s["mean_prob_shift"] for s in swaps])), fmt(surface["all"]["metrics"]["macro_f1_no_text"]),
                     fmt(surface["changed_only"]["flip_rate"]), fmt(ig.get("mean_completeness_error"))])
        for label in LABELS:
            if label == RARE:  # val chỉ có 1 mẫu
                continue
            per_class.append([f"`{name}`", f"`{label}`", fmt(r["baseline"]["f1"][label]),
                              fmt(_mean([s["metrics"]["f1"][label] for s in swaps])),
                              fmt(_mean([s["per_class"][label]["flip_rate"] for s in swaps if label in s["per_class"]])),
                              fmt(surface["all"]["metrics"]["f1"][label])])
        for label, c in (ig.get("per_class") or {}).items():
            top = ", ".join(w["word"].strip() or "(trắng)" for w in c["top_words"][:12]).replace("|", "\\|")
            words.append([f"`{name}`", f"`{label}`", f"{c['punct_share']:.1%}", top])
    parts = [table(["Run", "Macro-F1 bỏ text", "Sau đổi ảnh", "Đổi nhãn khi đổi ảnh", "Dịch xác suất", "Sau khi bỏ emoji/dấu câu",
                    "Đổi nhãn trên caption bị sửa", "Sai số IG"], rows),
             "Theo lớp:", table(["Run", "Lớp", "F1 gốc", "F1 sau đổi ảnh", "Đổi nhãn khi đổi ảnh", "F1 sau khi bỏ emoji/dấu câu"], per_class),
             "Từ có điểm IG cao nhất trong caption, theo lớp dự đoán:",
             table(["Run", "Lớp dự đoán", "Tỉ lệ điểm ở dấu câu", "Từ"], words)]
    return "\n\n".join(parts)


def source_tag_section(output_dir):
    """Kết quả scripts/check_source_tags.py, nếu đã chạy."""
    root = Path(output_dir) / "diagnostics"
    results = {p.name: m for p in sorted(root.iterdir()) if (m := read_json(p / "source_tags.json"))} if root.exists() else {}
    if not results:
        return ""
    rows, tag_rows = [], []
    for name, r in results.items():
        for variant, v in r["variants"].items():
            rows.append([f"`{name}`", variant, v["n_changed"], fmt(r["baseline"]["macro_f1_no_text"]), fmt(v["all"]["metrics"]["macro_f1_no_text"]),
                         fmt(r["baseline"]["accuracy"]), fmt(v["all"]["metrics"]["accuracy"]), fmt(v["changed_only"]["flip_rate"]),
                         fmt(v["changed_only"]["mean_prob_shift"])])
        for tag, t in r["tag_subsets"].items():
            rows_share = lambda d: ", ".join(f"{k.split('-')[0]} {d[k]:.0%}" for k in d)
            tag_rows.append([f"`{name}`", f"`{tag}`", t["n"], rows_share(t["gold_share"]), rows_share(t["pred_share"]),
                             rows_share(t["pred_share_after"]), fmt(t["accuracy"]), fmt(t["accuracy_after"])])
    return "\n\n".join([
        table(["Run", "Cách xoá", "Caption bị đổi", "Macro-F1 bỏ text", "Sau khi xoá", "Accuracy", "Sau khi xoá",
               "Đổi nhãn (caption bị đổi)", "Dịch xác suất (caption bị đổi)"], rows),
        "Theo từng thẻ nguồn, xoá theo cách `source_tags`:",
        table(["Run", "Thẻ", "Số mẫu val", "Nhãn thật", "Dự đoán trước", "Dự đoán sau", "Accuracy trước", "Accuracy sau"], tag_rows)])


def knn_section(output_dir):
    """Kết quả scripts/knn_eval.py, nếu đã chạy. Trả về (bảng baseline, bảng khác): cột (a)/(b) và phần còn lại."""
    root = Path(output_dir) / "knn"
    results = {p.name: m for p in sorted(root.iterdir()) if (m := read_json(p / "knn.json"))} if root.exists() else {}
    def render(items):
        if not items:
            return ""
        rows = []
        for name, r in items.items():
            best = r["best"]
            rows.append([f"`{name}`", fmt(r["uncorrected"]["score"]), fmt(r["no_retrieval"]["score"]), fmt(best["score"]), fmt(r["cv"]["score"]),
                         f"k={best['k']}, τ={best['tau']}, λ={best['lambda']}", fmt(r["no_retrieval"]["f1"]["image-sarcasm"]),
                         fmt(best["f1"]["image-sarcasm"]), fmt(r["similarity"]["top1_mean"]), fmt(r["similarity"]["top1_at_least_0.9"]),
                         fmt(r["similarity"]["top1_label_match"])])
        return table(["Run", "Chưa hiệu chỉnh prior", "Prior, không kNN (λ = 1)", "kNN tốt nhất", "kNN, kiểm định chéo", "Tham số",
                      "F1 `image`, λ = 1", "F1 `image`, kNN", "sim láng giềng gần nhất", "Tỉ lệ sim ≥ 0,9", "Trùng nhãn láng giềng"], rows)
    base = {k: v for k, v in results.items() if RUN_NAME_RE.match(k) and RUN_NAME_RE.match(k)["group"] in STAGE0_GROUPS}
    return render(base), render({k: v for k, v in results.items() if k not in base})


def stacking_section(output_dir):
    """Kết quả scripts/stack_predictions.py, nếu đã chạy."""
    r = read_json(Path(output_dir) / "b2" / "stacking.json")
    if not r:
        return ""
    rows = [[f"Model `{m}` riêng", fmt(s["oof"]["score"]), fmt(s["val"]["score"]), fmt(s["val_prior"]["score"]),
             fmt(s["val"]["f1"]["image-sarcasm"])] for m, s in r["single"].items()]
    rows.append(["Trung bình xác suất 3 model", "", fmt(r["average"]["val"]["score"]), fmt(r["average"]["val_prior"]["score"]),
                 fmt(r["average"]["val"]["f1"]["image-sarcasm"])])
    rows.append([f"Stacking, tốt nhất trên val ({', '.join(r['stacking_best']['params'])})", "", fmt(r["stacking_best"]["score"]), "",
                 fmt(r["stacking_best"]["f1"]["image-sarcasm"])])
    rows.append(["Stacking, kiểm định chéo trong val", "", fmt(r["stacking_cv"]["score"]), "", fmt(r["stacking_cv"]["f1"]["image-sarcasm"])])
    return table(["Cách", "Out-of-fold trên train", "Val", "Val, hiệu chỉnh prior", "F1 `image`, val"], rows)


def kaggle_rows(root):
    """Dòng env kaggle trong reports/results.md của nhánh này và của origin/main (nhóm chạy baseline trên Kaggle)."""
    sources = []
    path = Path(root) / "reports" / "results.md"
    if path.exists():
        sources.append(path.read_text(encoding="utf-8"))
    done = subprocess.run(["git", "show", "origin/main:reports/results.md"], cwd=root, capture_output=True, text=True)
    if done.returncode == 0:
        sources.append(done.stdout)
    rows = {}
    for text in sources:
        for line in text.splitlines():
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
            if len(cells) >= 8 and cells[3] == "kaggle":
                rows[tuple(cells[:2])] = [cells[1], cells[2], cells[4], cells[5], cells[6], " | ".join(cells[7:]).replace(" | ", " \\| ")]
    return list(rows.values())


def split_doc(text):
    """Bỏ tiêu đề #, tách theo mục ##. Trả về list (tiêu đề, nội dung)."""
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    return [(parts[i].strip(), parts[i + 1].strip().rstrip("-").strip()) for i in range(1, len(parts), 2)]


def embed_doc(text):
    """Mỗi mục của tài liệu thành một khối thu gọn; #### trở xuống không đổi vì không nằm trong mục lục báo cáo."""
    out = []
    for title, body in split_doc(text):
        out.append(f"<details>\n<summary>{title}</summary>\n\n{body}\n\n</details>")
    return "\n\n".join(out)


def question_rows(design, answers):
    match = re.search(r"^### 6\.3\..*?(?=^### 6\.4\.)", design, flags=re.M | re.S)
    rows = []
    for line in (match.group(0) if match else "").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if line.startswith("|") and len(cells) >= 2 and cells[0] not in ("So sánh",) and not cells[0].startswith("---"):
            rows.append([cells[0], " | ".join(cells[1:]), answers.get(len(rows) + 1, answers.get(str(len(rows) + 1), ""))])
    return rows


def section(heading, *blocks):
    body = "\n\n".join(str(b).strip() for b in blocks if b and str(b).strip())
    return f"{heading}\n\n{body}" if body else heading


def build_report(output_dir, design, plan, notes=None, kaggle=(), image_meta=None):
    notes = notes or {}
    note = lambda key: notes.get(key) or ""
    gate = lambda key: note(f"gate_{key}")
    groups = run_groups(output_dir)
    bars = g1_bars(groups)
    runs = teacher_runs(output_dir)
    val_runs = {k: v for k, v in runs.items() if k.startswith("val")}
    g1 = g1_rows(val_runs, bars)
    diag = diagnostics(output_dir)
    diag_base = {k: v for k, v in diag.items() if RUN_NAME_RE.match(k) and RUN_NAME_RE.match(k)["group"] in STAGE0_GROUPS}
    diag_other = {k: v for k, v in diag.items() if k not in diag_base}
    knn_base, knn_other = knn_section(output_dir)

    gates = [
        ["G0", "Phân cấp mềm không kém phẳng quá 1 điểm macro-F1 trên val", g0_result(groups), gate("G0")],
        ["G1", "F1 `image-sarcasm` hoặc `multi-sarcasm` của teacher vượt mốc G0 trên val, parse lỗi dưới 5%",
         "; ".join(f"{r[0]}: {r[-1]}" for r in g1), gate("G1")],
        ["G2", "κ ≥ 0,6 trên mọi tiêu chí; mỗi lớp khoảng 30 mẫu đạt rubric; nhóm 1 đạt rubric ≥ 80%", "", gate("G2")],
        ["G3", "M1, M2 hoặc M3 hơn ô (b) tương ứng quá 1 độ lệch chuẩn", "", gate("G3")],
        ["G4", "Báo cáo đủ các mục của giai đoạn 8", "", gate("G4")],
        ["G5", "E3 hơn cùng cấu hình không pretrain quá 1 độ lệch chuẩn trên val", "", gate("G5")],
    ]
    bar_rows = [[f"`{label}`", "có" if prefix == "val+prior" else "không", fmt(value), f"`{name}`"]
                for (prefix, label), (value, name) in sorted(bars.items())]
    meta = image_meta or {}
    data_rows = [[k, meta.get(k)] for k in ("num_images", "ocr_model", "vlm_model", "missing")] if meta else []

    teacher_sections = []
    for prefix, title in TEACHER_SETS:
        subset = {k: v for k, v in runs.items() if k.startswith(prefix)}
        teacher_sections.append(section(f"### {title}", teacher_table(subset) if subset else ""))

    doc = [
        "# Báo cáo thí nghiệm: CoT distillation và các hướng liên quan",
        "File này được sinh tự động bởi `vimmsd.analysis.experiment_report`, không sửa tay. Nhận xét và quyết định nằm ở "
        "`reports/experiment_report_notes.yaml`. Mục chưa có kết quả được để trống. Số liệu trên val dùng để phát triển; "
        "private test chỉ xuất hiện sau bước đánh giá cuối.",
        section("## 1. Trạng thái các cổng quyết định",
                table(["Cổng", "Điều kiện", "Kết quả đo được", "Quyết định"], gates)),
        section("## 2. Thiết kế thí nghiệm", "Chép nguyên từ `docs/experiment_design.md`.", embed_doc(design)),
        section("## 3. Kế hoạch chạy", "Chép nguyên từ `docs/experiment_plan.md`.", embed_doc(plan)),
        section("## 4. Dữ liệu", table(["Mục", "Giá trị"], data_rows) if data_rows else "", note("data")),
        section("## 5. Giai đoạn 0: baseline và chẩn đoán",
                section("### 5.1. Baseline của nhóm trên Kaggle",
                        table(["Cấu hình", "Commit", "Val macro-F1", "Test macro-F1", "Thời gian (phút)", "Ghi chú"], kaggle)
                        if kaggle else "", note("kaggle")),
                section("### 5.2. Cột (a)/(b) trên val, chưa hiệu chỉnh prior", group_table(groups, STAGE0_GROUPS, "val")),
                section("### 5.3. Cột (a)/(b) trên val, có hiệu chỉnh prior", group_table(groups, STAGE0_GROUPS, "val+prior")),
                section("### 5.4. Mốc cho G1", table(["Lớp", "Hiệu chỉnh prior", "F1 tốt nhất", "Cấu hình"], bar_rows) if bar_rows else "",
                        note("stage0")),
                section("### 5.5. D1: kiểm tra shortcut trên baseline", d1_tables(diag_base) if diag_base else "", note("d1")),
                section("### 5.6. Kiểm tra thẻ nguồn trong caption", source_tag_section(output_dir), note("source_tags"),
                        group_table(groups, ("distill_flat_a", "distill_flat_a_notags"), "val")
                        if "distill_flat_a_notags" in groups else "", note("source_tags_retrain")),
                section("### 5.7. E1: kNN retrieval", knn_base, note("e1"))),
        section("## 6. Giai đoạn 1: thử teacher", *teacher_sections,
                section("### Đối chiếu với mốc G1 trên val",
                        table(["Lần chạy", "F1 `image`", "Mốc", "F1 `multi`", "Mốc", "Parse lỗi", "G1"], g1) if g1 else ""),
                note("stage1"), section("### E2: few-shot", note("e2"))),
        section("## 7. Giai đoạn 2: sinh reasoning", note("stage2")),
        section("## 8. Giai đoạn 3: phân nhóm và rationalization", note("stage3")),
        section("## 9. Giai đoạn 4: chấm theo rubric", note("stage4")),
        section("## 10. Giai đoạn 5: đóng gói dữ liệu", note("stage5")),
        section("## 11. Giai đoạn 6: ma trận thí nghiệm",
                section("### 11.1. Macro-F1 trên val", matrix_table(groups, "val")),
                section("### 11.2. Macro-F1 trên val, có hiệu chỉnh prior", matrix_table(groups, "val+prior")),
                section("### 11.3. Macro-F1 trên private test", matrix_table(groups, "test")),
                section("### 11.4. Macro-F1 trên private test, có hiệu chỉnh prior", matrix_table(groups, "test+prior")),
                section("### 11.5. Ablation: bỏ reasoning nhóm 3 (val)", group_table(groups, ABLATION_GROUPS, "val")),
                section("### 11.6. D1 sau distillation", d1_tables(diag_other) if diag_other else ""),
                section("### 11.7. E1 trên từng hàng", knn_other, note("e1_matrix")), note("stage6")),
        section("## 12. Giai đoạn 7: các nhánh tuỳ chọn",
                section("### B3: reasoning làm đầu vào", note("b3")),
                section("### B2: 3 model riêng + stacking", stacking_section(output_dir), note("b2")),
                section("### E3: pretrain trên dữ liệu ngoài", note("e3"))),
        section("## 13. Trả lời các câu hỏi thí nghiệm",
                table(["So sánh", "Câu hỏi", "Kết quả"], question_rows(design, notes.get("questions") or {}))),
        section("## 14. Kết quả phụ của teacher", note("teacher_extra")),
        section("## 15. Ví dụ định tính", note("examples")),
        section("## 16. Hạn chế", note("limitations")),
    ]
    return "\n\n".join(doc) + "\n"


def write_report(root=ROOT, output_dir=None):
    root = Path(root)
    output_dir = output_dir or load_config(root / "configs" / "base.yaml").paths.output_dir
    cache_dir = Path(output_dir).parent / "cache"
    notes_path = root / "reports" / "experiment_report_notes.yaml"
    notes = yaml.safe_load(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
    image = read_json(cache_dir / "image_preprocessed.json") or {}
    text = build_report(output_dir, (root / "docs" / "experiment_design.md").read_text(encoding="utf-8"),
                        (root / "docs" / "experiment_plan.md").read_text(encoding="utf-8"), notes,
                        kaggle_rows(root), image.get("meta"))
    path = root / "reports" / "experiment_report.md"
    path.write_text(text, encoding="utf-8")
    return path


def refresh_report():
    """Gọi cuối các script thí nghiệm. Chỉ chạy trên server: nơi khác không có đủ output và sẽ xoá kết quả khỏi báo cáo.
    Lỗi ở đây không được làm hỏng lần chạy vừa xong."""
    if detect_env() != "server" or os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        path = write_report()
        print(f"đã cập nhật {path}")
        return path
    except Exception as e:
        logger.warning("không cập nhật được báo cáo thí nghiệm: %s", e)
        return None
