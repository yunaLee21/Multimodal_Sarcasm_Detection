"""WS-A image dependence (scripts/diagnose_image_dependence.py). No GPU, network or real model."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from vimmsd.analysis.image_dependence import (build_conditions, compare, diff_label_map, encode_test_set,
                                              image_feature_mean, run_condition, same_label_derangement)

LABELS = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3])


class TinyModel(nn.Module):
    """encode/classify like VimmsdClassifier: text dict (tokens, mask, pooled), image dict (tokens, pooled)."""

    def __init__(self, use_image):
        super().__init__()
        torch.manual_seed(0)
        self.use_image = use_image
        self.text = nn.Linear(3, 4)
        self.img = nn.Linear(2, 4) if use_image else None
        self.head = nn.Linear(4, 4)

    def encode(self, batch):
        x = batch["x"]
        text = {"tokens": x.unsqueeze(1), "mask": torch.ones(len(x), 1, dtype=torch.bool), "pooled": x}
        image = None
        if self.use_image:
            px = batch["pixel_values"]
            image = {"tokens": px.unsqueeze(1).repeat(1, 3, 1), "pooled": px}
        return text, image

    def classify(self, text, image):
        z = self.text(text["pooled"])
        if image is not None:
            z = z + 3 * self.img(image["tokens"].mean(1) + image["pooled"])
        return self.head(torch.tanh(z))


def loader(n=10, bs=4):
    g = torch.Generator().manual_seed(1)
    x, px = torch.randn(n, 3, generator=g), torch.randn(n, 2, generator=g)
    return [{"ids": [str(i) for i in range(s, min(s + bs, n))], "labels": torch.as_tensor(LABELS[s:s + bs]),
             "x": x[s:s + bs], "pixel_values": px[s:s + bs]} for s in range(0, n, bs)]


def constants():
    return {"mean": {"tokens": torch.zeros(3, 2), "pooled": torch.zeros(2)},
            "grey": {"tokens": torch.ones(3, 2), "pooled": torch.ones(2)}}


def all_probs(model):
    enc = encode_test_set(model, loader(), torch.device("cpu"), False)
    conds = build_conditions(enc["labels"], [0, 1])
    return enc, {k: run_condition(model, enc, v, torch.device("cpu"), False, constants()) for k, v in conds.items()}


def test_caption_only_predictions_never_change():
    enc, probs = all_probs(TinyModel(use_image=False))
    assert enc["image"] is None
    for name, p in probs.items():
        assert np.array_equal(p, probs["original"]), name


def test_image_model_changes_under_swap_and_mask():
    model = TinyModel(use_image=True)
    enc, probs = all_probs(model)
    direct = torch.cat([model.classify(*model.encode(b)).softmax(-1) for b in loader()]).detach().numpy()
    assert np.allclose(probs["original"], direct, atol=1e-6)  # cached features reproduce the direct forward
    for name in ("swap_random_s0", "swap_same_label_s0", "swap_diff_label_s0", "mask_mean", "mask_grey"):
        assert not np.allclose(probs[name], probs["original"]), name


def test_swap_maps_respect_constraints():
    rng = np.random.default_rng(0)
    src, fixed = same_label_derangement(LABELS, rng)
    assert np.array_equal(LABELS[src], LABELS)
    assert fixed.tolist() == (LABELS == 1).tolist()  # the singleton class keeps its own image
    d = diff_label_map(LABELS, rng)
    assert np.all(LABELS[d] != LABELS)
    conds = build_conditions(LABELS, [0, 1, 2])
    for s in (0, 1, 2):
        r = conds[f"swap_random_s{s}"]
        assert sorted(r) == list(range(len(LABELS))) and not np.any(r == np.arange(len(LABELS)))


def test_image_feature_mean():
    enc = nn.Module()
    enc.forward = lambda px: {"tokens": px.unsqueeze(1).repeat(1, 2, 1), "pooled": px}
    batches = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0]])]
    out = image_feature_mean(enc, batches, torch.device("cpu"), False, torch.float32)
    assert out["n"] == 3
    assert torch.allclose(out["pooled"], torch.tensor([3.0, 4.0]))
    assert out["tokens"].shape == (2, 2)


def test_compare():
    p0 = np.array([[0.9, 0, 0, 0.1], [0.1, 0, 0, 0.9]])
    p1 = np.array([[0.2, 0, 0, 0.8], [0.1, 0, 0, 0.9]])
    out = compare([0, 3], p0, p1, 4)
    assert out["flip"].tolist() == [True, False]
    assert out["d_p_gold"] == pytest.approx([-0.7, 0.0])
    assert out["flip_matrix"][0, 3] == 1 and out["flip_matrix"][3, 3] == 1
