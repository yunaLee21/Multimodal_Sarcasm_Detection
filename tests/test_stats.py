"""Test thống kê (src/vimmsd/analysis/stats.py). Không cần dữ liệu thật."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from vimmsd.analysis import stats as S


def test_macro_f1_matches_sklearn_and_exclude():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    assert S.macro_f1(y, p, 4) == pytest.approx(f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0))
    assert S.macro_f1(y, p, 4, exclude=(1,)) == pytest.approx(
        f1_score(y, p, average="macro", labels=[0, 2, 3], zero_division=0))
    # lớp không có mẫu và không được dự đoán: F1 = 0, như sklearn zero_division=0
    assert S.macro_f1([0, 0], [0, 0], 2) == pytest.approx(0.5)


def test_paired_contrast_identical_models_is_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 200)
    p = [rng.integers(0, 3, 200) for _ in range(3)]
    r = S.paired_contrast(y, p, p, 3, n_boot=200, seed=0)
    assert r["mean"] == 0 and r["ci95"] == [0.0, 0.0] and r["per_seed"] == [0, 0, 0]


def test_paired_contrast_detects_clear_gain_and_is_reproducible():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 400)
    good = [np.where(rng.random(400) < 0.8, y, rng.integers(0, 3, 400)) for _ in range(3)]
    bad = [np.where(rng.random(400) < 0.4, y, rng.integers(0, 3, 400)) for _ in range(3)]
    r = S.paired_contrast(y, good, bad, 3, n_boot=300, seed=0)
    assert r["mean"] > 0.2 and r["ci95"][0] > 0 and r["p_boot"] < 0.01
    assert r == S.paired_contrast(y, good, bad, 3, n_boot=300, seed=0)
    with pytest.raises(ValueError):
        S.paired_contrast(y, good, bad[:2], 3, n_boot=10)


def test_group_bootstrap_resamples_whole_groups():
    groups = np.array(["a", "a", "a", "b", "c"])
    for idx in S.bootstrap_indices(5, 50, 0, groups):
        picked = list(idx)
        # nhóm a (chỉ số 0,1,2) luôn xuất hiện trọn vẹn: số lần 0, 1, 2 xuất hiện bằng nhau
        assert picked.count(0) == picked.count(1) == picked.count(2)
    assert S.bootstrap_indices(5, 3, 0).shape == (3, 5)


def test_group_bootstrap_widens_interval_when_groups_are_correlated():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(40), 10)               # 40 nhóm × 10 mẫu
    y = np.repeat(rng.integers(0, 2, 40), 10)
    flip = np.repeat(rng.random(40) < 0.3, 10)           # lỗi theo nhóm
    a = [np.where(flip, 1 - y, y)]
    b = [y.copy()]
    plain = S.paired_contrast(y, a, b, 2, n_boot=300, seed=0)
    grouped = S.paired_contrast(y, a, b, 2, n_boot=300, seed=0, groups=groups)
    assert (grouped["ci95"][1] - grouped["ci95"][0]) > (plain["ci95"][1] - plain["ci95"][0])


def test_holm():
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert S.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    adj = S.holm([0.2, 0.01, 0.6])
    assert adj[1] == pytest.approx(0.03) and adj[0] == pytest.approx(0.4) and adj[2] == pytest.approx(0.6)


def test_per_class_and_subset_table():
    y = np.array([0, 0, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 0])
    pc = S.per_class(y, p, 3)
    assert pc[1] == {"precision": pytest.approx(2 / 3), "recall": 1.0, "f1": pytest.approx(0.8), "support": 2}
    rows = S.subset_table(y, {"m": [p]}, {"all": np.ones(5, bool), "none": np.zeros(5, bool)}, 3)
    assert rows[0]["support"] == 5 and rows[0]["labels"] == [2, 2, 1] and rows[1]["m"] is None
    # subset chỉ có lớp 0 và 1: lớp 2 vắng mặt không bị tính F1 = 0
    only01 = S.subset_table(y, {"m": [p]}, {"01": y < 2}, 3)[0]
    assert only01["classes_scored"] == [0, 1]
    assert only01["m"] == pytest.approx(S.macro_f1(y[:4], p[:4], 3, exclude=(2,)))
