"""Test lọc nhiễu (src/vimmsd/data/denoise.py). Không cần GPU, mạng, model thật."""
import numpy as np
import pytest

from vimmsd.data import denoise as D

MINOR = ("image-sarcasm", "text-sarcasm")


def test_normalize_caption():
    assert D.normalize_caption("  Song Joong Ki &amp; Song\n\tHye  Kyo ") == "song joong ki & song hye kyo"
    decomposed = "Việt"          # ệ dạng tổ hợp (NFD)
    assert D.normalize_caption(decomposed) == D.normalize_caption("Việt") == "việt"
    assert D.normalize_caption(None) == ""


def test_id_sort_and_group_by_key():
    assert sorted(["10", "2", "a"], key=D.id_sort_key) == ["2", "10", "a"]
    groups = D.group_by_key({"10": "x", "2": "x", "3": "y", "4": "z", "11": "z", "5": "x"})
    assert groups == [["2", "5", "10"], ["4", "11"]]


def test_union_find_and_components():
    comps = D.components_from_pairs(["1", "2", "3", "4", "5"], [("3", "2"), ("4", "3")])
    assert comps == [["2", "3", "4"]]
    uf = D.UnionFind(["7", "3", "9"])
    uf.union("9", "7")
    assert uf.groups() == {"7": "g7", "9": "g7", "3": "g3"}


def test_resolve_group_same_label_keeps_smallest_id():
    out = D.resolve_group(["12", "3", "40"], {"12": "image-sarcasm", "3": "image-sarcasm", "40": "image-sarcasm"}, MINOR)
    assert out["3"][0] == "keep" and out["12"][0] == out["40"][0] == "drop_duplicate"


def test_resolve_group_majority_relabel():
    labels = {"1": "not-sarcasm", "2": "multi-sarcasm", "3": "multi-sarcasm"}
    out = D.resolve_group(["1", "2", "3"], labels, MINOR)
    assert out["2"][:2] == ("relabel_majority", "multi-sarcasm")
    assert out["1"][0] == out["3"][0] == "drop_duplicate"


def test_resolve_group_no_majority_drops_all():
    out = D.resolve_group(["1", "2"], {"1": "not-sarcasm", "2": "multi-sarcasm"}, MINOR)
    assert {a for a, _, _ in out.values()} == {"drop_conflict"}


def test_resolve_group_protects_minority():
    # không có đa số: mẫu lớp hiếm được giữ và gắn cờ, bản sao cùng nhãn hiếm thì bỏ
    labels = {"1": "not-sarcasm", "2": "image-sarcasm", "3": "image-sarcasm", "4": "multi-sarcasm"}
    out = D.resolve_group(list(labels), labels, MINOR)
    assert out["1"][0] == out["4"][0] == "drop_conflict"
    assert out["2"][:2] == ("flag_conflict", "image-sarcasm") and out["3"][0] == "drop_duplicate"
    # đa số là lớp thường: mẫu lớp hiếm vẫn được giữ
    labels = {"1": "not-sarcasm", "2": "not-sarcasm", "3": "not-sarcasm", "4": "text-sarcasm"}
    out = D.resolve_group(list(labels), labels, MINOR)
    assert out["1"][0] == "relabel_majority" and out["4"][0] == "flag_conflict"
    assert [a for a, _, _ in out.values()].count("drop_duplicate") == 2
    # đa số là lớp hiếm: giữ một mẫu lớp hiếm, bỏ mẫu lớp thường
    labels = {"1": "not-sarcasm", "2": "image-sarcasm", "3": "image-sarcasm"}
    out = D.resolve_group(list(labels), labels, MINOR)
    assert out["2"][:2] == ("relabel_majority", "image-sarcasm") and out["1"][0] == out["3"][0] == "drop_duplicate"


def test_decide_mislabel_rules_and_cap():
    ids = ["1", "2", "3", "4", "5"]
    labels = ["not-sarcasm", "multi-sarcasm", "image-sarcasm", "not-sarcasm", "multi-sarcasm"]
    issue = [True, True, True, True, False]
    knn = [0.1, 0.0, 0.0, 0.5, 0.0]
    conf = [0.3, 0.1, 0.05, 0.2, 0.01]
    actions, capped = D.decide_mislabel(ids, labels, issue, knn, conf, {"not-sarcasm", "multi-sarcasm"}, 0.2, 10)
    assert actions == {"1": "drop_mislabel", "2": "drop_mislabel", "3": "flag_mislabel", "4": "flag_mislabel"}
    assert capped == 0
    actions, capped = D.decide_mislabel(ids, labels, issue, knn, conf, {"not-sarcasm", "multi-sarcasm"}, 0.2, 1)
    assert actions["2"] == "drop_mislabel" and actions["1"] == "flag_mislabel" and capped == 1


def test_stratified_sample_reproducible_and_proportional():
    labels = {str(i): ("not-sarcasm" if i < 80 else "image-sarcasm" if i < 98 else "text-sarcasm") for i in range(100)}
    a = D.stratified_sample(list(labels), labels, 10, seed=42)
    assert a == D.stratified_sample(list(labels), labels, 10, seed=42) and len(a) == 10
    counts = {k: sum(labels[i] == k for i in a) for k in set(labels.values())}
    assert counts == {"not-sarcasm": 8, "image-sarcasm": 1, "text-sarcasm": 1}
    assert D.stratified_sample(["1", "2"], labels, 10, seed=0) == ["1", "2"]


def test_hamming_and_cosine_pairs():
    hashes = np.array([0b0, 0b111, 0xFF00, 0b1], dtype=np.uint64)
    assert sorted(map(tuple, D.hamming_pairs(hashes, 3).tolist())) == [(0, 1), (0, 3), (1, 3)]
    emb = np.array([[1, 0], [0.99, 0.141], [0, 1]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert D.cosine_pairs(emb, 0.95).tolist() == [[0, 1]]


def test_knn_agreement_excludes_same_group():
    feats = np.array([[1, 0], [1, 0.01], [0.9, 0.1], [0, 1]], dtype=np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    labels = np.array([0, 0, 1, 1])
    # 0 và 1 cùng nhóm: láng giềng gần nhất hợp lệ của 0 là 2 (nhãn khác)
    agree = D.knn_agreement(feats, labels, np.array([0, 0, 1, 2]), k=1)
    assert agree.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
    agree = D.knn_agreement(feats, labels, np.arange(4), k=1)
    assert agree[0] == pytest.approx(1.0)
