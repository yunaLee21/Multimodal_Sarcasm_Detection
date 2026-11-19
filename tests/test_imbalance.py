"""Experiment C (Phase 9): oversampling B2 và augmentation caption B4/B5, không cần mạng/GPU."""
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import WeightedRandomSampler

from vimmsd.analysis.calibration import apply_class_multipliers, tune_class_multipliers
from vimmsd.data.augmentation import (ACCEPT, REJECT, BackTranslator, accepted_variants, augment_records,
                                      auto_review, read_manifest, restore_emojis, write_manifest)
from vimmsd.data.dataset import (ViMMSDDataset, build_train_sampler, draws_per_epoch, load_all_records,
                                 sampling_summary, sampling_weights)
from vimmsd.models.fusion import manifold_mixup
from vimmsd.training.losses import LogitAdjustmentLoss, ManifoldMixupLoss, build_training_loss
from vimmsd.utils.config import load_config

LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]
ROOT = Path(__file__).resolve().parents[1]


def records(counts, augmented=0):
    out = [{"id": f"{c}_{i}", "caption": f"câu {c} {i}", "image_path": "x.jpg", "label": c}
           for c, n in enumerate(counts) for i in range(n)]
    out += [{"id": f"1_{i}#aug0", "caption": f"bản sinh {i}", "image_path": "x.jpg", "label": 1, "parent_id": f"1_{i}"}
            for i in range(augmented)]
    return out


def dataset(counts, augmented=0):
    return ViMMSDDataset(records(counts, augmented), load_image=False)


def test_sqrt_sampling_gives_class_share_proportional_to_sqrt_count():
    labels = [0] * 400 + [1] * 4 + [2] * 100
    weights = sampling_weights(labels, 4, power=0.5)
    share = Counter()
    for label, w in zip(labels, weights):
        share[label] += w
    total = sum(share.values())
    assert share[0] / total == pytest.approx(20 / 32)
    assert share[1] / total == pytest.approx(2 / 32)
    uniform = sampling_weights(labels, 4, power=1.0)
    assert sum(w for l, w in zip(labels, uniform) if l == 1) == pytest.approx(1.0)


def test_sampler_keeps_draws_per_epoch_of_original_train():
    plain = dataset([30, 3, 6, 20])
    assert build_train_sampler(plain, 4, "random", seed=0) is None
    weighted = build_train_sampler(plain, 4, "weighted", power=0.5, seed=0)
    assert isinstance(weighted, WeightedRandomSampler) and len(weighted) == len(plain)

    augmented = dataset([30, 3, 6, 20], augmented=3)
    assert draws_per_epoch(augmented) == 59 and len(augmented) == 62
    random = build_train_sampler(augmented, 4, "random", seed=0)
    drawn = list(random)
    assert len(drawn) == 59 and len(set(drawn)) == 59
    assert list(build_train_sampler(augmented, 4, "random", seed=0)) == drawn
    with pytest.raises(ValueError):
        build_train_sampler(plain, 4, "balanced")


def test_sampling_summary_counts_draws_and_augmented_samples():
    ds = dataset([30, 3, 6, 20], augmented=3)
    summary = sampling_summary(ds, LABELS, build_train_sampler(ds, 4, "weighted", power=0.5, seed=0))
    classes = summary["classes"]
    assert summary["draws_per_epoch"] == 59 and summary["dataset_size"] == 62
    assert sum(c["draws_per_epoch"] for c in classes.values()) == pytest.approx(59)
    assert classes["text-sarcasm"]["samples"] == 6 and classes["text-sarcasm"]["augmented"] == 3
    assert classes["text-sarcasm"]["unique_seen_per_epoch"] <= 6
    uniform = sampling_summary(ds, LABELS, None)
    assert uniform["classes"]["not-sarcasm"]["draws_per_epoch"] == pytest.approx(59 * 30 / 62)


def test_auto_review_rejects_rule_violations():
    parent = "Hôm nay trời đẹp quá, đi chơi với 2 đứa bạn thân thôi"
    known = {"một caption khác trong train"}
    verdict, reasons, sim = auto_review(parent, "Trời hôm nay đẹp ghê, đi chơi cùng 2 người bạn thân", known)
    assert verdict == ACCEPT and not reasons and sim < 0.9
    assert auto_review(parent, parent + ".", known)[0] == REJECT
    assert "đổi con số" in auto_review(parent, "Trời hôm nay đẹp ghê, đi chơi cùng 3 người bạn thân", known)[1]
    assert "thêm từ chỉ nhãn" in auto_review(parent, "Câu mỉa mai: trời đẹp, đi chơi với 2 bạn thân", known)[1]
    assert any("độ dài" in r for r in auto_review(parent, "trời đẹp 2", known)[1])
    assert "lặp từ" in auto_review(parent, "đi đi đi đi chơi với 2 bạn thân trời đẹp", known)[1]
    assert "trùng caption khác trong train" in auto_review("Một caption khác, trong train!", "một caption khác trong train", known)[1]
    assert auto_review(parent, "  ", known)[1] == ["rỗng"]
    assert auto_review("-kem", "Dày", known)[1] == ["caption gốc quá ngắn (1 từ)"]
    assert "caption gốc quá dài (151 từ)" in auto_review(" ".join(["chữ"] * 151), "một bản dịch", known)[1]


def test_similarity_of_long_captions_is_not_collapsed_by_autojunk():
    parent = "Cách kiếm 1 triệu nhanh nhất là lên Shopee thu thập voucher. " * 6
    candidate = "Cách kiếm được 1 triệu nhanh nhất là vào Shopee để thu thập voucher. " * 6
    assert auto_review(parent, candidate, set())[2] > 0.8


def test_restore_emojis_appends_lost_emojis_once():
    text, lost = restore_emojis("vui quá 😂😂 đi chơi 🎉", "rất vui khi đi chơi")
    assert text == "rất vui khi đi chơi 😂🎉" and lost == ["😂", "🎉"]
    assert restore_emojis("vui 😂", "vui ghê 😂") == ("vui ghê 😂", [])


def row(pid, k, caption, verdict=ACCEPT, similarity=0.5, review=None, parent_caption=None):
    return {"parent_id": pid, "candidate": k, "label": "text-sarcasm", "parent_caption": parent_caption or f"câu 1 {pid[-1]}",
            "caption": caption, "auto_verdict": verdict, "auto_reasons": [], "similarity": similarity, "review": review}


def test_accepted_variants_keep_one_per_parent_and_respect_review():
    rows = [row("1_0", 0, "a", similarity=0.4), row("1_0", 1, "b", similarity=0.7),
            row("1_1", 0, "c", verdict=REJECT), row("1_1", 1, "d", verdict=REJECT, review="accept"),
            row("1_2", 0, "e", review="reject")]
    chosen = accepted_variants(rows)
    assert {pid: [r["caption"] for r in v] for pid, v in chosen.items()} == {"1_0": ["b"], "1_1": ["d"]}
    assert set(accepted_variants(rows, require_review=True)) == {"1_1"}


def test_augment_records_only_for_parents_in_train():
    train = records([2, 2, 0, 0])
    rows = [row("1_0", 1, "biến thể 0"), row("1_9", 0, "không có trong train")]
    extra = augment_records(train, rows)
    assert len(extra) == 1
    aug = extra[0]
    assert aug["id"] == "1_0#aug1" and aug["parent_id"] == "1_0" and aug["label"] == 1
    assert aug["caption"] == "biến thể 0" and aug["image_path"] == "x.jpg"
    with pytest.raises(ValueError, match="lệch dữ liệu"):
        augment_records(train, [row("1_0", 0, "x", parent_caption="caption cũ")])


def test_back_translator_groups_candidates_per_caption(monkeypatch):
    bt = BackTranslator.__new__(BackTranslator)
    bt.tok_vi2en = bt.model_vi2en = bt.tok_en2vi = bt.model_en2vi = None

    def fake(texts, tok, model, tgt_lang, num_return_sequences=1, **_):
        return [f"{t}|{tgt_lang}|{k}" for t in texts for k in range(num_return_sequences)]

    monkeypatch.setattr(bt, "_translate", fake)
    out = bt(["a", "b", "c"], batch_size=2, num_return_sequences=2)
    assert out == [[f"{t}|en_XX|0|vi_VN|{k}" for k in range(2)] for t in "abc"]


@pytest.fixture
def official_data(tmp_path):
    img = tmp_path / "images"
    img.mkdir()
    Image.new("RGB", (8, 8)).save(img / "0.jpg")
    train = {str(i): {"image": "0.jpg", "caption": f"caption {i}", "label": LABELS[i % 4]} for i in range(20)}
    (tmp_path / "train.json").write_text(json.dumps(train, ensure_ascii=False), encoding="utf-8")
    for name, start in (("public", 100), ("private", 200)):
        ann = {str(i): {"image": "0.jpg", "caption": f"{name} {i}", "label": LABELS[i % 4]} for i in range(start, start + 4)}
        (tmp_path / f"{name}.json").write_text(json.dumps(ann), encoding="utf-8")
    (tmp_path / "cache").mkdir()
    return tmp_path


def load(data_dir, *overrides):
    cfg = load_config(ROOT / "configs/fusion_concat_frozen_reg_aug.yaml", overrides=[
        f"paths.local.data_dir={data_dir}", f"paths.local.cache_dir={data_dir / 'cache'}",
        "data.train_json=train.json", "data.train_image_dir=images", "data.public_test_json=public.json",
        "data.public_test_labels=null", "data.public_test_image_dir=images", "data.private_test_json=private.json",
        "data.private_test_labels=null", "data.private_test_image_dir=images", *overrides], env="local")
    return load_all_records(cfg)


def test_load_all_records_adds_augmentation_to_train_only(official_data):
    with pytest.raises(FileNotFoundError, match="augment_captions"):
        load(official_data)
    rows = [row("1", 0, "caption một, viết lại", parent_caption="caption 1"),
            row("5", 0, "caption năm viết lại", parent_caption="caption 5", review="reject")]
    write_manifest(official_data / "cache" / "augmentation_manifest.jsonl", rows)
    assert read_manifest(official_data / "cache" / "augmentation_manifest.jsonl") == rows

    splits = load(official_data)
    assert len(splits["train"]) == 21 and len(splits["val"]) == 4 and len(splits["test"]) == 4
    assert [r["id"] for r in splits["train"] if r.get("parent_id")] == ["1#aug0"]
    held_out = load(official_data, "data.fold.index=0", "data.fold.k=2", "data.fold.seed=0")
    assert all(r["parent_id"] in {x["id"] for x in held_out["train"]} for r in held_out["train"] if r.get("parent_id"))
    assert len(load(official_data, "data.augment.file=null")["train"]) == 20


def test_review_csv_roundtrip(tmp_path):
    spec = importlib.util.spec_from_file_location("augment_captions", ROOT / "scripts" / "augment_captions.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    rows = [row("1_0", 0, "a"), row("1_0", 1, "b")]
    csv_path = script.review_csv_path(tmp_path / "augmentation_manifest.jsonl")
    assert csv_path.name == "augmentation_review.csv"
    script.write_review_csv(csv_path, rows)

    def fill(*verdicts):
        with open(csv_path, encoding="utf-8-sig") as f:
            table = list(csv.DictReader(f))
        for item, verdict in zip(table, verdicts):
            item["review"] = verdict
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=script.REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(table)

    fill("", "Reject")
    assert [r["review"] for r in script.import_review([dict(r) for r in rows], csv_path)] == [None, "reject"]
    fill("maybe", "")
    with pytest.raises(ValueError, match="accept"):
        script.import_review([dict(r) for r in rows], csv_path)


def test_logit_adjustment_matches_manual_shifted_cross_entropy():
    counts = [8, 1, 1, 2]
    logits = torch.randn(5, 4)
    targets = torch.tensor([0, 1, 2, 3, 1])
    loss_fn = LogitAdjustmentLoss(counts, tau=0.5)
    priors = torch.tensor(counts, dtype=torch.float)
    expected = F.cross_entropy(logits + 0.5 * (priors / priors.sum()).log(), targets)
    assert torch.isclose(loss_fn(logits, targets), expected)


def test_manifold_mixup_keeps_minority_anchor_weight():
    torch.manual_seed(7)
    z = torch.arange(24, dtype=torch.float).reshape(4, 6)
    labels = torch.tensor([0, 1, 0, 2])
    mixed, lam, perm = manifold_mixup(z, labels, alpha=0.2, minority_labels=(1, 2))
    assert mixed.shape == z.shape and perm.shape == labels.shape
    assert (lam[[1, 3]] >= 0.5).all()
    assert torch.allclose(mixed, lam * z + (1 - lam) * z[perm])


def test_manifold_mixup_loss_uses_soft_targets():
    logits = torch.randn(2, 4)
    y = torch.tensor([1, 3])
    lam = torch.tensor([[0.75], [0.25]])
    perm = torch.tensor([1, 0])
    loss_fn = ManifoldMixupLoss(alpha=0.2)
    loss, parts = loss_fn({"logits": logits, "mixup_lambda": lam, "mixup_permutation": perm}, {"labels": y})
    one_hot = F.one_hot(y, 4).float()
    target = lam * one_hot + (1 - lam) * one_hot[perm]
    expected = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    assert torch.isclose(loss, expected) and set(parts) == {"cls"}


def test_advanced_loss_configs():
    cfg = load_config(ROOT / "configs/fusion_concat_frozen_reg_mixup.yaml", env="local")
    loss = build_training_loss(cfg, [30, 3, 6, 20])
    assert loss.uses_mixup and loss.mixup_minority_labels == (1, 2)
    invalid = load_config(ROOT / "configs/fusion_concat_frozen_reg_mixup.yaml",
                          overrides=["loss.class_weight=sqrt_inverse"], env="local")
    with pytest.raises(ValueError, match="không class weight"):
        build_training_loss(invalid, [30, 3, 6, 20])


def test_class_multipliers_change_decision_and_renormalize_probabilities():
    probabilities = np.array([[0.60, 0.25, 0.10, 0.05], [0.80, 0.10, 0.05, 0.05]])
    adjusted = apply_class_multipliers(probabilities, [1.0, 3.0, 1.0, 1.0])
    assert adjusted[0].argmax() == 1 and adjusted[1].argmax() == 0
    assert np.allclose(adjusted.sum(axis=1), 1.0)


def test_calibration_optimization_does_not_reduce_calibration_macro_f1():
    probabilities = np.array([
        [0.70, 0.20, 0.05, 0.05], [0.60, 0.25, 0.10, 0.05],
        [0.50, 0.40, 0.05, 0.05], [0.45, 0.40, 0.10, 0.05],
        [0.20, 0.10, 0.60, 0.10], [0.15, 0.10, 0.65, 0.10],
        [0.10, 0.10, 0.10, 0.70], [0.10, 0.10, 0.20, 0.60],
    ])
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    multipliers, info = tune_class_multipliers(probabilities, labels, maxiter=8, popsize=6, seed=0)
    assert multipliers.shape == (4,) and multipliers[0] == 1.0
    assert np.all((multipliers >= 0.1) & (multipliers <= 10.0))
    assert info["calibration_macro_f1"] >= info["baseline_macro_f1"]
