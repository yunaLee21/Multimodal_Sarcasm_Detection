"""Test B2 (scripts/stack_predictions.py). Không cần GPU, mạng, model thật."""
import numpy as np
import pytest

from vimmsd.analysis.stacking import assemble_oof, cross_validated, features, fit_candidates, prior_ratio
from vimmsd.data.dataset import fold_of
from vimmsd.utils.experiments import expand_runs, load_plan

NAMES = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]


def records(n=50):
    return [{"id": str(i), "label": 2 if i % 5 == 0 else 0} for i in range(n)]


def test_folds_are_stratified_disjoint_and_stable():
    recs = records()
    folds = fold_of(recs, k=5, seed=1)
    assert set(folds) == {r["id"] for r in recs} and set(folds.values()) == set(range(5))
    for f in range(5):
        members = [r for r in recs if folds[r["id"]] == f]
        assert sum(r["label"] == 2 for r in members) == 2 and len(members) == 10
    assert fold_of(list(reversed(recs)), k=5, seed=1) == folds
    assert fold_of(recs, k=5, seed=2) != folds


def test_oof_rejects_predictions_from_a_model_that_saw_the_sample():
    folds = {"a": 0, "b": 1}
    good = {0: {"ids": ["a"], "probs": [[0.9, 0.1]]}, 1: {"ids": ["b"], "probs": [[0.2, 0.8]]}}
    assert assemble_oof(good, folds, ["b", "a"]).tolist() == [[0.2, 0.8], [0.9, 0.1]]
    with pytest.raises(ValueError, match="đã học"):
        assemble_oof({0: {"ids": ["a", "b"], "probs": [[1, 0], [1, 0]]}}, folds, ["a", "b"])
    with pytest.raises(ValueError, match="hai lần"):
        assemble_oof({0: {"ids": ["a", "a"], "probs": [[1, 0], [1, 0]]}, 1: good[1]}, folds, ["a", "b"])
    with pytest.raises(ValueError, match="không có"):
        assemble_oof({0: good[0]}, folds, ["a", "b"])


def test_stacking_learns_which_model_to_trust():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 2, 3], size=300)
    good = np.full((300, 4), 0.05)
    good[np.arange(300), y] = 0.85
    noise = rng.dirichlet(np.ones(4), size=300)
    x = features([noise, good])
    assert x.shape == (300, 8) and np.isfinite(x).all()
    shift = prior_ratio([100, 0, 100, 100], [100, 0, 100, 100])
    assert np.allclose(shift, 0)
    cands = fit_candidates(x, y, x, 4, shift, {"C": [1.0], "class_weight": [None]})
    assert set(cands) == {(1.0, "none", False), (1.0, "none", True)}
    probs = cands[(1.0, "none", False)]
    assert np.allclose(probs.sum(1), 1) and (probs.argmax(1) == y).mean() > 0.95
    cv = cross_validated(y, cands, NAMES, "macro_f1", ["text-sarcasm"])
    assert cv["accuracy"] > 0.95 and len(cv["chosen"]) == 5


def test_b2_plan_has_every_fold_and_full_run():
    specs = expand_runs(load_plan("configs/experiments/b2_stacking.yaml"))
    names = {s.name for s in specs}
    assert len(specs) == 18
    for m in ("text", "image", "multi"):
        assert {f"b2_{m}_f{i}_s42" for i in range(5)} | {f"b2_{m}_full_s42"} <= names
    fold3 = next(s for s in specs if s.name == "b2_image_f3_s42")
    assert "data.fold.index=3" in fold3.overrides and "train.num_workers=0" in fold3.overrides
