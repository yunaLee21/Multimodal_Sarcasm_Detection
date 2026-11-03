"""Test E1 (scripts/knn_eval.py). Không cần GPU, mạng, model thật."""
import numpy as np
import pytest

from vimmsd.analysis.knn import _best, grid_search, grid_search_cv, knn_probs, nearest_neighbors, similarity_stats

NAMES = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]


def test_nearest_neighbors_sorted_cosine():
    memory = np.array([[1.0, 0], [0, 1.0], [0.6, 0.8]], dtype=np.float32)
    sims, idx = nearest_neighbors(np.array([[1.0, 0]], dtype=np.float32), memory, 2)
    assert idx.tolist() == [[0, 2]] and sims[0] == pytest.approx([1.0, 0.6])
    with pytest.raises(ValueError):
        nearest_neighbors(memory, memory, 4)


def test_knn_probs_weights_and_prior_correction():
    sims = np.array([[0.9, 0.7, 0.1]])
    neigh = np.array([[0, 3, 3]])
    same = knn_probs(sims, neigh, 3, 0.2, [10, 10, 10, 10], [10, 10, 10, 10])[0]
    w = np.exp((np.array([0.9, 0.7, 0.1]) - 0.9) / 0.2)
    assert same[0] == pytest.approx(w[0] / w.sum()) and same[3] == pytest.approx((w[1] + w[2]) / w.sum())
    shifted = knn_probs(sims, neigh, 3, 0.2, [30, 10, 10, 10], [10, 10, 10, 30])[0]
    assert shifted[3] > same[3] and shifted.sum() == pytest.approx(1)
    only_k1 = knn_probs(sims, neigh, 1, 0.2, [10, 10, 10, 10], [10, 10, 10, 10])[0]
    assert only_k1[0] == pytest.approx(1)


def test_knn_probs_small_tau_does_not_overflow():
    out = knn_probs(np.array([[1.0, -1.0]]), np.array([[0, 3]]), 2, 1e-3, [1, 1, 1, 1], [0, 0, 0, 0])
    assert np.isfinite(out).all() and out[0, 0] == pytest.approx(1)


def fixture():
    labels = np.array([0, 3] * 20)
    model = np.tile([0.5, 0, 0, 0.5], (40, 1)) + np.tile([0.01, 0, 0, 0], (40, 1))  # model luôn đoán lớp 0
    model /= model.sum(1, keepdims=True)
    neigh = np.stack([labels] * 3, axis=1)  # láng giềng luôn đúng
    sims = np.tile([0.9, 0.8, 0.7], (40, 1))
    grid = {"k": [1, 3], "tau": [0.1], "lambda": [0.5, 1.0]}
    return labels, model, sims, neigh, [20, 0, 0, 20], [20, 0, 0, 20], NAMES, grid, "macro_f1", ["text-sarcasm"]


def test_grid_contains_baseline_and_retrieval_wins_when_neighbors_are_right():
    rows = grid_search(*fixture())
    assert rows[0]["lambda"] == 1 and rows[0]["k"] is None and len(rows) == 3
    best = _best(rows)
    assert best["lambda"] == 0.5 and best["accuracy"] == 1 and rows[0]["accuracy"] == 0.5


def test_tie_prefers_no_retrieval():
    labels, model, sims, neigh, *rest = fixture()
    assert _best(grid_search(labels, np.eye(4)[labels], sims, neigh, *rest))["lambda"] == 1


def test_cv_estimates_out_of_sample_gain():
    out = grid_search_cv(*fixture(), folds=2)
    assert out["accuracy"] == 1 and len(out["chosen"]) == 2 and all(c["lambda"] == 0.5 for c in out["chosen"])


def test_cv_does_not_gain_from_random_neighbors():
    rng = np.random.default_rng(0)
    labels = np.array([0, 3] * 200)
    model = np.zeros((400, 4))
    model[np.arange(400), labels] = 0.6
    model[np.arange(400), 3 - labels] = 0.4
    neigh = rng.choice([0, 3], size=(400, 3))
    args = (labels, model, np.tile([0.9, 0.8, 0.7], (400, 1)), neigh, [20, 0, 0, 20], [20, 0, 0, 20], NAMES,
            {"k": [3], "tau": [0.1], "lambda": [0.5, 1.0]}, "macro_f1", ["text-sarcasm"])
    assert grid_search_cv(*args)["accuracy"] == pytest.approx(1)  # model đã đúng, retrieval không được chọn


def test_similarity_stats():
    out = similarity_stats(np.array([[0.95, 0.1], [0.5, 0.1]]), np.array([[3, 0], [0, 0]]), np.array([3, 3]))
    assert out["top1_mean"] == pytest.approx(0.725) and out["top1_at_least_0.9"] == 0.5 and out["top1_label_match"] == 0.5
