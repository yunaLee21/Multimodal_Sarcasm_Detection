"""Test D1 (scripts/diagnose_shortcuts.py). Không cần GPU, mạng, model thật."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vimmsd.analysis.shortcut_check import (compare_predictions, integrated_gradients, rank_words, strip_surface_signals,
                                            stratified_indices, swap_image_views, without_surface_signals)

LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]


def fake_dataset():
    records = [{"id": str(i), "caption": f"c{i}", "image_path": f"{i}.jpg", "label": 0} for i in range(3)]
    return SimpleNamespace(records=records, texts=["t0", "t1", "t2"], ocrs=["o0", "o1", "o2"],
                           descriptions=["d0", "d1", "d2"], image_texts=None)


def test_swap_moves_image_ocr_description_and_keeps_caption():
    ds = fake_dataset()
    out = swap_image_views(ds, [1, 2, 0])
    assert [r["image_path"] for r in out.records] == ["1.jpg", "2.jpg", "0.jpg"]
    assert out.ocrs == ["o1", "o2", "o0"] and out.descriptions == ["d1", "d2", "d0"]
    assert out.texts == ds.texts and [r["id"] for r in out.records] == ["0", "1", "2"]
    assert ds.ocrs == ["o0", "o1", "o2"] and ds.records[0]["image_path"] == "0.jpg"  # không sửa dataset gốc


@pytest.mark.parametrize("raw,expected", [
    ("Hay quá 😂😂 :))", "Hay quá"),
    ("Thật á!!! Ừ...", "Thật á Ừ"),
    ("Một câu, bình thường.", "Một câu, bình thường."),
    ("😂", ""),
    (None, ""),
])
def test_strip_surface_signals(raw, expected):
    assert strip_surface_signals(raw) == expected


def test_surface_removal_only_preprocesses_changed_captions():
    ds = SimpleNamespace(records=[{"caption": "bình thường"}, {"caption": "hay quá 😂"}, {"caption": "hay quá!!!"}],
                         texts=["đã xử lý", "x", "y"])
    calls = []
    out, changed = without_surface_signals(ds, lambda t: calls.append(t) or f"p:{t}")
    assert out.texts == ["đã xử lý", "p:hay quá", "p:hay quá"]
    assert changed.tolist() == [False, True, True] and calls == ["hay quá"]
    assert ds.texts == ["đã xử lý", "x", "y"]


def test_compare_predictions():
    probs = torch.tensor([[1.0, 0, 0, 0], [0, 0, 0, 1.0], [0, 0, 0, 1.0]])
    other = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 0, 0.5, 0.5]])
    a = {"ids": ["a", "b", "c"], "labels": [0, 3, 3], "preds": [0, 3, 3], "probs": probs}
    b = {"ids": ["a", "b", "c"], "labels": [0, 3, 3], "preds": [0, 0, 3], "probs": other}
    out = compare_predictions(a, b, LABELS)
    assert out["flip_rate"] == pytest.approx(1 / 3)
    assert out["mean_prob_shift"] == pytest.approx((0 + 1 + 0.5) / 3)
    assert out["per_class"]["multi-sarcasm"] == {"n": 2, "flip_rate": 0.5, "mean_prob_shift": 0.75}
    assert "image-sarcasm" not in out["per_class"]
    only = compare_predictions(a, b, LABELS, np.array([True, False, False]))
    assert only["n"] == 1 and only["flip_rate"] == 0
    with pytest.raises(ValueError):
        compare_predictions(a, {**b, "ids": ["c", "b", "a"]}, LABELS)


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask):
        return SimpleNamespace(last_hidden_state=torch.tanh(inputs_embeds) * attention_mask.unsqueeze(-1))


class FakeModel(nn.Module):
    """Pooled = trung bình token; phi tuyến nên IG phải cần nhiều bước mới khớp completeness."""

    def __init__(self):
        super().__init__()
        self.text_encoder = SimpleNamespace(model=FakeEncoder())
        self.encoder = self.text_encoder.model
        self.head = nn.Linear(4, 4)
        self.tokenizer = SimpleNamespace(all_special_ids=[0, 1, 2], pad_token_id=1)

    def encode(self, batch):
        return {"aux": 1}, None

    def classify(self, text, image):
        assert text["aux"] == 1  # các luồng khác giữ nguyên
        weights = text["mask"].unsqueeze(-1).float()
        return self.head((text["tokens"] * weights).sum(1) / weights.sum(1))


def test_integrated_gradients_is_complete_and_ignores_special_and_padding():
    torch.manual_seed(0)
    model = FakeModel()
    batch = {"input_ids": torch.tensor([[0, 3, 4, 2], [0, 5, 2, 1]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])}
    scores, target, error = integrated_gradients(model, batch, steps=64)
    assert scores.shape == (2, 4) and target.shape == (2,)
    assert error.max() < 1e-3
    assert scores[0, 0] == 0 and scores[0, 3] == 0 and scores[1, 2] == 0 and scores[1, 3] == 0
    assert scores[0, 1] != 0 and scores[1, 1] != 0
    assert not any(p.grad is not None for p in model.parameters())


def test_stratified_indices_caps_each_class():
    out = stratified_indices([0, 0, 0, 3, 3, 2], per_class=2)
    classes = [[0, 0, 0, 3, 3, 2][i] for i in out]
    assert classes.count(0) == 2 and classes.count(3) == 2 and classes.count(2) == 1
    assert out == sorted(out) == sorted(set(out))


def test_rank_words_merges_subwords_and_reports_punctuation():
    vocab = {0: "<s>", 1: "<pad>", 2: "</s>", 3: "cư@@", 4: "ời", 5: "!"}
    tokenizer = SimpleNamespace(convert_ids_to_tokens=lambda ids: [vocab[i] for i in ids],
                                all_special_tokens=["<s>", "<pad>", "</s>"])
    rows = [([0, 3, 4, 5, 2, 1], [9.0, 1.0, 2.0, 1.0, 9.0, 9.0], 3)] * 2
    out = rank_words(tokenizer, rows, LABELS, min_count=2)["multi-sarcasm"]
    assert out["top_words"][0] == {"word": "cười", "mean_attribution": 3.0, "count": 2}
    assert out["punct_share"] == pytest.approx(0.25)
