"""Test CoT distillation + head phân cấp (docs/cot_distillation.md). Không cần GPU, mạng, model thật."""
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vimmsd.data import rationale as R
from vimmsd.models.heads import CascadeHead, HierarchicalHead, LabelIndex, compute_prior_shift
from vimmsd.training.losses import SarcasmLoss, build_training_loss
from vimmsd.utils.config import load_config
from vimmsd.utils.experiments import expand_runs, load_plan

LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]
NOT, TEXT, IMAGE, MULTI = range(4)


# ---------------------------------------------------------------- head


def test_hierarchical_head_is_a_distribution_in_label_order():
    torch.manual_seed(0)
    shuffled = ["multi-sarcasm", "not-sarcasm", "image-sarcasm", "text-sarcasm"]  # thứ tự nhãn bất kỳ
    idx = LabelIndex(shuffled)
    head = HierarchicalHead(8, 16, 0.0, idx, use_axis=True)
    z = torch.randn(5, 8, requires_grad=True)
    out = head(z, axis_probs=torch.rand(5, 2))
    p = out["logits"].exp()
    assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-5)
    p_s = torch.sigmoid(out["s_logit"])
    assert torch.allclose(p[:, idx.not_], 1 - p_s, atol=1e-5)
    p_c = torch.softmax(out["c_logits"], -1)
    assert torch.allclose(p[:, idx.multi], p_s * p_c[:, 2], atol=1e-5)
    # NLL đi qua cả tầng S lẫn tầng C
    F.nll_loss(out["logits"], torch.tensor([0, 1, 2, 3, 0])).backward()
    assert head.s[-1].weight.grad.abs().sum() > 0 and head.c[-1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        LabelIndex(["a", "b", "c", "d"])


def test_cascade_routes_hard():
    idx = LabelIndex(LABELS)
    head = CascadeHead(4, 8, 0.0, idx)
    z = torch.zeros(3, 4)
    with torch.no_grad():
        for layer in (head.m[-1], head.r[-1]):
            layer.weight.zero_()
        head.m[-1].bias.copy_(torch.tensor([0.0]))  # p_m = 0,5 -> nhánh multi
        head.r[-1].bias.copy_(torch.tensor([0.0, 5.0, 0.0]))
    assert (head(z)["logits"].argmax(-1) == MULTI).all()
    with torch.no_grad():
        head.m[-1].bias.copy_(torch.tensor([-0.5]))  # p_m ~ 0,38 < 0,5 -> nhánh R, dù R chia đều thì multi vẫn thua
        head.r[-1].bias.copy_(torch.tensor([0.1, 0.0, 0.0]))
    out = head(z)["logits"]
    assert (out.argmax(-1) == NOT).all() and torch.allclose(out.exp()[:, MULTI], torch.full((3,), 1e-8))


def test_prior_shift_matches_target_prior():
    train, val = [6062, 77, 442, 4224], [442, 1, 28, 942]
    flat = compute_prior_shift("flat", train, val, LABELS)["logits"]
    assert torch.allclose(flat, torch.log(torch.tensor(val) / sum(val)) - torch.log(torch.tensor(train) / sum(train)))
    hier = compute_prior_shift("hierarchical", train, val, LABELS)
    p_tr, p_val = 1 - 6062 / sum(train), 1 - 442 / sum(val)
    expected = math.log(p_val / (1 - p_val)) - math.log(p_tr / (1 - p_tr))
    assert hier["s"].item() == pytest.approx(expected) and hier["s"].item() > 0  # test có nhiều châm biếm hơn
    # model hiệu chỉnh hoàn hảo trên train: dịch xong thì xác suất tầng S bằng đúng prior của val
    head = HierarchicalHead(2, 4, 0.0, LabelIndex(LABELS))
    with torch.no_grad():
        for layer in (head.s[-1], head.c[-1]):
            layer.weight.zero_()
        head.s[-1].bias.fill_(math.log(p_tr / (1 - p_tr)))
    p = head(torch.zeros(1, 2), shift=hier)["logits"].exp()[0]
    assert 1 - p[NOT].item() == pytest.approx(p_val, abs=1e-4)
    casc = compute_prior_shift("cascade", train, val, LABELS)
    assert set(casc) == {"m", "r"} and casc["r"].shape == (3,)


def _tiny_classifier(monkeypatch, head_cfg):
    from transformers import RobertaConfig, RobertaModel

    from vimmsd.models import text_encoder
    from vimmsd.models.classifier import SarcasmClassifier

    tiny = RobertaModel(RobertaConfig(vocab_size=50, max_position_embeddings=40, hidden_size=16,
                                      num_hidden_layers=1, num_attention_heads=2, intermediate_size=32))
    monkeypatch.setattr(text_encoder.AutoModel, "from_pretrained", lambda name: tiny)
    monkeypatch.setattr(text_encoder.AutoTokenizer, "from_pretrained", lambda name: None)
    enc = text_encoder.TextEncoder("tiny", freeze=True)
    return SarcasmClassifier("text", enc, None, {}, num_classes=4, head_hidden=16, dropout=0.0,
                             head_cfg=head_cfg, labels=LABELS)


def test_flat_head_keeps_checkpoint_names(monkeypatch):
    model = _tiny_classifier(monkeypatch, None)
    keys = {k for k in model.state_dict() if not k.startswith("text_encoder.")}
    assert keys == {"head.1.weight", "head.1.bias", "head.4.weight", "head.4.bias"}  # như head cũ
    batch = {"input_ids": torch.randint(5, 50, (3, 6)), "attention_mask": torch.ones(3, 6, dtype=torch.long)}
    model.eval()
    raw = model(batch)
    model.set_prior_shift({"logits": torch.tensor([0.0, 0.0, 0.0, 10.0])})
    assert (model(batch).argmax(-1) == MULTI).all() and torch.allclose(model(batch), raw + torch.tensor([0, 0, 0, 10.0]))
    model.set_prior_shift(None)
    assert torch.allclose(model(batch), raw)


@pytest.mark.parametrize("head_type", ["flat", "hierarchical", "cascade"])
def test_end_to_end_loss_and_gradients(monkeypatch, head_type):
    torch.manual_seed(0)
    model = _tiny_classifier(monkeypatch, {"type": head_type, "axis_heads": True, "axis_targets": "teacher",
                                           "rationale_dim": 8})
    y = torch.tensor([NOT, TEXT, IMAGE, MULTI, MULTI, NOT])
    batch = {"input_ids": torch.randint(5, 50, (6, 7)), "attention_mask": torch.ones(6, 7, dtype=torch.long),
             "labels": y, "axis_teacher": torch.tensor([[0, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]]).float(),
             "axis_teacher_mask": torch.tensor([1.0, 0, 0, 1, 0, 0]),
             "rationale_emb": torch.randn(6, 8), "rationale_mask": torch.tensor([1.0, 0, 0, 1, 0, 0])}
    loss_fn = SarcasmLoss(torch.nn.CrossEntropyLoss(), head_type, LABELS, lambda_s=0.5, lambda_axis=0.3,
                          axis_targets="teacher", lambda_rat=0.3)
    model.train()
    out = model.forward_outputs(batch)
    assert out["logits"].shape == (6, 4) and out["axis_logits"].shape == (6, 2) and out["rationale"].shape == (6, 8)
    loss, parts = loss_fn(out, batch)
    expected = {"flat": {"cls"}, "hierarchical": {"cls", "s"}, "cascade": {"multi", "route"}}[head_type]
    assert set(parts) == expected | {"axis", "rat"}
    loss.backward()
    assert all(p.grad is not None for p in model.head.parameters())
    assert model.axis_head[-1].weight.grad is not None and model.rationale_proj.weight.grad is not None
    assert all(p.grad is None for p in model.text_encoder.parameters())  # encoder đóng băng


def test_hierarchical_ce_is_nll_and_axis_targets():
    log_p = torch.log_softmax(torch.randn(4, 4), -1)
    y = torch.tensor([0, 1, 2, 3])
    assert F.cross_entropy(log_p, y).item() == pytest.approx(F.nll_loss(log_p, y).item(), abs=1e-6)

    loss = SarcasmLoss(torch.nn.CrossEntropyLoss(), "flat", LABELS, axis_targets="teacher")
    batch = {"axis_teacher": torch.tensor([[0, 0], [0, 0], [0, 0], [1, 0]]).float(),
             "axis_teacher_mask": torch.tensor([0.0, 0, 0, 1])}
    target, mask = loss._axis(batch, y)
    # not / text / image: nhãn suy từ nhãn gốc; multi: nhãn của teacher nếu có, không có thì bỏ
    assert target.tolist() == [[0, 0], [1, 0], [0, 1], [1, 0]] and mask.tolist() == [True] * 4
    _, mask = loss._axis({"axis_teacher": batch["axis_teacher"], "axis_teacher_mask": torch.zeros(4)}, y)
    assert mask.tolist() == [True, True, True, False]
    gold = SarcasmLoss(torch.nn.CrossEntropyLoss(), "flat", LABELS)
    assert gold._axis({}, y)[0].tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]


def test_selection_score_excludes_rare_class():
    from vimmsd.training.trainer import selection_score

    m = {"macro_f1": 0.5, "weighted_f1": 0.7,
         "per_class": {"not-sarcasm": {"f1": 0.6}, "text-sarcasm": {"f1": 0.0}, "image-sarcasm": {"f1": 0.3},
                       "multi-sarcasm": {"f1": 0.9}}}
    assert selection_score(m) == 0.5 and selection_score(m, "weighted_f1") == 0.7
    assert selection_score(m, "macro_f1", ["text-sarcasm"]) == pytest.approx(0.6)


# ---------------------------------------------------------------- teacher output, phân nhóm, chấm


def _teacher_json(label, t=False, i=True):
    return json.dumps({"text_alone": {"sarcastic": t, "reason": "caption bình thường"},
                       "image_alone": {"sarcastic": i, "reason": "ảnh mâu thuẫn"},
                       "combined": {"reason": "ghép lại thấy mỉa mai"}, "label": label}, ensure_ascii=False)


def test_parse_teacher_output_variants():
    good = R.parse_teacher_output("Phân tích:\n```json\n" + _teacher_json("multi-sarcasm") + "\n```")
    assert good["label"] == "multi-sarcasm" and good["text_sarcastic"] is False and good["image_sarcastic"] is True
    loose = R.parse_teacher_output('{"text_alone": {"sarcastic": "có", "reason": "x"}, "image_alone": '
                                   '{"sarcastic": "không", "reason": "y"}, "combined": {"reason": "z"}, '
                                   '"label": "Text Sarcasm",}')
    assert loose["label"] == "text-sarcasm" and loose["text_sarcastic"] and not loose["image_sarcastic"]
    assert R.parse_teacher_output("không phải JSON") is None
    assert R.parse_teacher_output(_teacher_json("sarcasm?")) is None
    assert "Ảnh:" in R.rationale_text(good) and "multi-sarcasm" not in R.rationale_text(good)


def test_build_prompt_label_hint_and_truncation():
    p = R.build_prompt("cap " * 1000, "")
    assert "(không có)" in p and "Nhãn đúng" not in p and len(p) < 4000
    assert "Nhãn đúng của bài đăng này là: image-sarcasm" in R.build_prompt("cap", "chữ", label="image-sarcasm")


def _row(sid, label, k_label, idx=0):
    return {"id": sid, "sample_idx": idx, "label": label, "parsed": R.parse_teacher_output(_teacher_json(k_label))}


def test_group_samples_and_review_selection():
    raw = {"a": [_row("a", "multi-sarcasm", "multi-sarcasm", j) for j in range(3)],
           "b": [_row("b", "multi-sarcasm", l, j) for j, l in enumerate(["not-sarcasm", "multi-sarcasm", "x"])],
           "c": [_row("c", "text-sarcasm", "not-sarcasm", j) for j in range(3)]}
    grouped = R.group_samples(raw, {"a": "multi-sarcasm", "b": "multi-sarcasm", "c": "text-sarcasm"})
    assert [grouped[s]["group"] for s in "abc"] == [1, 2, 3]
    assert grouped["b"]["chosen"]["sample_idx"] == 1 and grouped["c"]["chosen"] is None
    assert R.group_of(1, 1) == 1 and R.group_of(0, 1) == 3
    picked = R.select_for_review(grouped, group1_fraction=0.0)
    assert picked == {"b", "c"}


def test_merge_dataset_verdicts_and_loader(tmp_path):
    raw = {sid: [_row(sid, "multi-sarcasm", "multi-sarcasm")] for sid in ("auto", "ok", "bad")}
    raw["g3"] = [_row("g3", "image-sarcasm", "not-sarcasm")]
    grouped = R.group_samples(raw, {"auto": "multi-sarcasm", "ok": "multi-sarcasm", "bad": "multi-sarcasm",
                                    "g3": "image-sarcasm"})
    rationalized = {"g3": [_row("g3", "image-sarcasm", "image-sarcasm")]}
    full = {c: "1" for c in R.RUBRIC}
    reviews = [{"id": "ok", **full, "edited_rationale": "sửa lại", "text_sarcastic": "1"},
               {"id": "bad", **full, "c3_combined": "0"},
               {"id": "g3", **full}]
    data = R.merge_dataset(grouped, rationalized, reviews, {"prompt_version": "v1"})
    s = data["samples"]
    assert s["auto"]["review"] == "auto" and s["ok"]["review"] == "pass" and s["bad"]["review"] == "fail"
    assert s["ok"]["rationale"] == "sửa lại" and s["ok"]["edited"] and s["ok"]["text_sarcastic"] is True
    assert s["g3"]["rationalized"] and s["g3"]["review"] == "pass" and s["g3"]["image_sarcastic"] is True
    path = tmp_path / "rationale_dataset.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    keep, meta = R.load_rationale_dataset(path)
    assert set(keep) == {"auto", "ok", "g3"} and meta["prompt_version"] == "v1"
    assert set(R.load_rationale_dataset(path, exclude_groups=[3])[0]) == {"auto", "ok"}


def test_cohen_kappa_and_rubric(tmp_path):
    assert R.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert R.cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0]) == pytest.approx(0.0)
    rows_a = [{"id": str(i), **{c: "1" for c in R.RUBRIC}} for i in range(4)]
    rows_b = [{"id": str(i), **{c: "1" for c in R.RUBRIC}, "c1_text": "0" if i == 0 else "1"} for i in range(4)]
    k = R.rubric_kappa(rows_a, rows_b)
    assert k["c2_image"]["kappa"] == 1.0 and k["c1_text"]["n"] == 4
    R.write_csv(tmp_path / "r.csv", [{"id": "1", "caption": "chữ có dấu, phẩy"}])
    assert R.read_csv(tmp_path / "r.csv")[0]["caption"] == "chữ có dấu, phẩy"


def test_make_id_lists_disjoint_and_complete():
    labels = {f"n{i}": "not-sarcasm" for i in range(120)}
    labels.update({f"t{i}": "text-sarcasm" for i in range(50)})
    labels.update({f"i{i}": "image-sarcasm" for i in range(80)})
    labels.update({f"m{i}": "multi-sarcasm" for i in range(90)})
    lists = R.make_id_lists(labels, pilot_dev=40, pilot_eval=30, not_batch=100)
    dev, ev = set(lists["pilot_dev"]), set(lists["pilot_eval"])
    assert not dev & ev and len(dev) == 160 and len(ev) == 30 + 10 + 30 + 30  # text chỉ còn 10 sau dev
    batches = [set(lists[f"batch{k}"]) for k in range(1, 5)]
    assert set().union(*batches) == set(labels) and sum(map(len, batches)) == len(labels)
    assert lists == R.make_id_lists(labels, pilot_dev=40, pilot_eval=30, not_batch=100)  # cố định theo seed


# ---------------------------------------------------------------- dataset, config, plan


def test_attach_rationales_and_collator(tmp_path):
    from vimmsd.data.dataset import ViMMSDCollator, ViMMSDDataset, attach_rationales

    root = tmp_path / "rationales"
    root.mkdir()
    samples = {"1": {"label": "multi-sarcasm", "text_sarcastic": False, "image_sarcastic": True, "group": 1,
                     "review": "auto", "rationale": "r1"},
               "2": {"label": "text-sarcasm", "text_sarcastic": True, "image_sarcastic": False, "group": 2,
                     "review": "pending", "rationale": "r2"}}
    (root / "rationale_dataset.json").write_text(json.dumps({"meta": {}, "samples": samples}), encoding="utf-8")
    np.save(root / "rationale_embeddings.npy", np.ones((1, 8), dtype=np.float32))
    (root / "rationale_embeddings_ids.json").write_text(json.dumps(["1"]))
    cfg = load_config("configs/distill_flat_d.yaml", [f"data.rationale.dir={root}", "model.head.rationale_dim=8"],
                      env="local")
    records = [{"id": str(i), "caption": f"cap {i}", "label": i % 4, "image_path": "x.jpg"} for i in range(3)]
    out, dim, n = attach_rationales(records, cfg)
    assert dim == 8 and n == 1 and out[1]["teacher_axis"] == (0.0, 1.0) and "teacher_axis" not in out[2]

    ds = ViMMSDDataset(out, load_image=False, rationale_dim=dim)
    batch = ViMMSDCollator()([ds[i] for i in range(3)])
    assert batch["axis_teacher"].tolist()[1] == [0.0, 1.0] and batch["axis_teacher_mask"].tolist() == [0, 1, 0]
    assert batch["rationale_emb"].shape == (3, 8) and batch["rationale_mask"].tolist() == [0, 1, 0]
    assert batch["rationale_emb"][0].abs().sum() == 0

    with pytest.raises(ValueError, match="rationale_dim"):
        attach_rationales(records, load_config("configs/distill_flat_d.yaml", [f"data.rationale.dir={root}"],
                                               env="local"))


DISTILL = ["distill_flat_a", "distill_flat_b", "distill_flat_c", "distill_flat_d", "distill_flat_d_nogroup3",
           "distill_hier_a", "distill_hier_b", "distill_hier_c", "distill_hier_d", "distill_hier_d_nogroup3",
           "hier_cascade"]


@pytest.mark.parametrize("name", DISTILL)
def test_distill_configs(name):
    cfg = load_config(f"configs/{name}.yaml", env="local")
    head = cfg.model.head
    assert head.type == ("cascade" if name == "hier_cascade" else "hierarchical" if "hier" in name else "flat")
    col = name.split("_")[2] if name.startswith("distill") else "a"
    assert bool(head.axis_heads) == (col in "bcd")
    assert (head.axis_targets == "teacher") == (col in "cd") and bool(cfg.data.rationale.use) == (col in "cd")
    assert (cfg.loss.lambda_rat > 0) == (col == "d") and (head.rationale_dim == 768) == (col == "d")
    assert cfg.data.rationale.exclude_groups == ([3] if "nogroup3" in name else [])
    assert not cfg.train.eval_test and cfg.eval.prior_correction
    assert cfg.train.select_exclude_labels == ["text-sarcasm"]
    loss = build_training_loss(cfg, [6062, 77, 442, 4224])
    assert loss.head_type == head.type


def test_base_defaults_keep_old_behaviour():
    cfg = load_config("configs/fusion_concat_frozen.yaml", env="local")
    assert cfg.model.head.type == "flat" and not cfg.model.head.axis_heads and cfg.train.eval_test
    loss = build_training_loss(cfg, [10, 1, 2, 5])
    logits, y = torch.randn(4, 4), torch.tensor([0, 1, 2, 3])
    total, parts = loss({"logits": logits}, {"labels": y})
    assert set(parts) == {"cls"} and total.item() == pytest.approx(F.cross_entropy(logits, y).item())


@pytest.mark.parametrize("plan", ["cot_distill_stage0", "cot_distill_grid", "cot_distill"])
def test_cot_plans(plan):
    specs = expand_runs(load_plan(f"configs/experiments/{plan}.yaml"))
    assert specs and len({s.name for s in specs}) == len(specs)
    if plan == "cot_distill_grid":
        assert len(specs) == 18
        cfg = load_config(specs[0].config, specs[0].overrides, env="local")
        assert (cfg.loss.lambda_axis, cfg.loss.lambda_rat) == (0.1, 0.1)


@pytest.mark.parametrize("head_type", ["hierarchical", "cascade"])
def test_trainer_runs_with_new_heads(monkeypatch, tmp_path, head_type):
    """Vòng lặp Trainer thật (CPU, 2 epoch) với head mới + loss phụ: chạy được, lưu best.pt, load lại được."""
    from vimmsd.training.trainer import Trainer
    from vimmsd.utils.checkpoint import load_checkpoint

    torch.manual_seed(0)
    cfg = load_config("configs/distill_hier_b.yaml", [
        f"model.head.type={head_type}", "train.epochs=2", "train.amp=false", "train.early_stop_patience=0",
        f"paths.local.output_dir={tmp_path}"], env="local")
    model = _tiny_classifier(monkeypatch, dict(cfg.model.head))

    def batch(n):
        y = torch.arange(n) % 4
        return {"ids": [str(i) for i in range(n)], "input_ids": torch.randint(5, 50, (n, 6)),
                "attention_mask": torch.ones(n, 6, dtype=torch.long), "labels": y}

    loaders = {"train": [batch(8), batch(8)], "val": [batch(8)]}
    trainer = Trainer(model, build_training_loss(cfg, [10, 3, 4, 8]), loaders, cfg, tmp_path)
    summary = trainer.fit()
    assert len(summary["history"]) == 2 and (tmp_path / "best.pt").exists()
    assert 0 <= summary["best_val_macro_f1"] <= 1 and "val_select" in summary["history"][0]
    metrics, _ = trainer.evaluate("val", prior_shift=compute_prior_shift(head_type, [10, 3, 4, 8], [3, 1, 1, 6],
                                                                          LABELS))
    assert model.prior_shift is None and set(metrics) >= {"macro_f1", "per_class"}
    fresh = _tiny_classifier(monkeypatch, dict(cfg.model.head))
    load_checkpoint(tmp_path / "best.pt", fresh)


@pytest.mark.parametrize("load_in_4bit", [False, True])
def test_teacher_4bit_only_changes_model_kwargs(monkeypatch, load_in_4bit):
    import transformers

    seen = {}

    class FakeModel:
        def eval(self):
            return self

    def fake_model(name, **kwargs):
        seen.update(kwargs)
        return FakeModel()

    processor = type("P", (), {"tokenizer": type("T", (), {})()})()
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", fake_model)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda name, **kw: processor)
    R.TeacherVLM("fake", load_in_4bit=load_in_4bit)
    assert seen["device_map"] == "cuda"
    quant = seen.get("quantization_config")
    if load_in_4bit:
        assert quant.load_in_4bit and quant.bnb_4bit_quant_type == "nf4" and quant.bnb_4bit_use_double_quant
    else:
        assert quant is None
