"""Test fold nội bộ có nhóm (src/vimmsd/data/splits.py). Không cần dữ liệu thật."""
from vimmsd.data import splits as S


def test_build_groups_caption_or_image_transitive():
    captions = {"1": "Haha", "2": " haha ", "3": "khác", "4": "riêng", "5": "lẻ"}
    md5s = {"1": "a", "2": "b", "3": "b", "4": "c", "5": "d"}
    groups = S.build_groups(captions, md5s)
    # 1-2 cùng caption, 2-3 cùng ảnh -> 1, 2, 3 một nhóm
    assert groups["1"] == groups["2"] == groups["3"] == "g1"
    assert groups["4"] == "g4" and groups["5"] == "g5"


def _toy(n_per_label=12):
    labels, groups = {}, {}
    for j, lab in enumerate(["a", "b", "c"]):
        for i in range(n_per_label):
            sid = str(j * 100 + i)
            labels[sid] = lab
            groups[sid] = f"g{j * 100 + i // 2}"  # nhóm 2 mẫu
    return labels, groups


def test_group_folds_valid_and_reproducible():
    labels, groups = _toy()
    folds = S.make_group_folds(labels, groups, k=3, seed=0)
    assert folds == S.make_group_folds(labels, groups, k=3, seed=0)
    assert S.validate_folds(labels, groups, folds, 3) == []
    assert all(sum(1 for i in folds if folds[i] == f) > 0 for f in range(3))


def test_validate_catches_split_group_missing_id_and_support():
    labels, groups = _toy()
    folds = S.make_group_folds(labels, groups, k=3, seed=0)
    broken = dict(folds)
    a, b = "0", "1"  # cùng nhóm g0
    broken[b] = (broken[a] + 1) % 3
    assert any("nhiều fold" in p for p in S.validate_folds(labels, groups, broken, 3))
    del broken[b]
    assert any("không có fold" in p for p in S.validate_folds(labels, groups, broken, 3))
    assert any("cần >= 99" in p for p in S.validate_folds(labels, groups, folds, 3, {"a": 99}))


def test_overlap_stats():
    splits = {"train": {"captions": {"1": "X", "2": "y"}, "md5s": {"1": "m1", "2": "m2"}},
              "test": {"captions": {"1": "x ", "2": "z", "3": "q"}, "md5s": {"1": "m9", "2": "m2", "3": "m3"}}}
    o = S.overlap_stats(splits)["train↔test"]
    assert o == {"n": 3, "caption": 1, "image_md5": 1, "caption_and_image": 0, "caption_or_image": 2}


def test_digest_stable():
    assert S.digest({"2": "b", "10": "a"}) == S.digest({"10": "a", "2": "b"})
