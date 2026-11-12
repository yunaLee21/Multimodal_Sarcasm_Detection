"""Test augmentation có duyệt tay (src/vimmsd/data/augment_review.py, dataset.apply_augmentation). Không cần model."""
import pytest

from vimmsd.data import augment_review as A
from vimmsd.data.dataset import apply_augmentation


def test_select_parents_and_auto_screen():
    train = {"10": {"caption": "a", "label": "text-sarcasm"}, "2": {"caption": "b", "label": "not-sarcasm"},
             "3": {"caption": "c", "label": "image-sarcasm"}}
    assert A.select_parents(train, ["text-sarcasm", "image-sarcasm"]) == ["3", "10"]
    norm = {"đã có", "a"}
    assert A.auto_screen("Câu gốc", "  ", norm, set()) == "empty"
    assert A.auto_screen("Câu gốc", "câu  GỐC", norm, set()) == "unchanged"
    assert A.auto_screen("Câu gốc", "Đã có", norm, set()) == "duplicate_of_train_caption"
    assert A.auto_screen("Câu gốc", "mới", norm, {"mới"}) == "duplicate_candidate"
    assert A.auto_screen("Câu gốc", "mới", norm, set()) is None


def _row(verdict, **crit):
    return {"verdict": verdict, **{c: crit.get(c, "y") for c in A.CRITERIA}}


def test_merge_reviews_requires_all_reviewers_and_all_criteria():
    reviews = {"A": {"c1": _row("accept"), "c2": _row("accept"), "c3": _row("accept", no_invented_content="n"),
                     "c4": _row("reject")},
               "B": {"c1": _row("y"), "c2": _row("reject"), "c3": _row("accept")}}
    m = A.merge_reviews(["c1", "c2", "c3", "c4"], reviews)
    assert m["c1"]["accepted"] is True
    assert m["c2"]["accepted"] is False and "reviewers disagree" in m["c2"]["reasons"]
    assert m["c3"]["accepted"] is False and m["c3"]["verdicts"]["A"] == A.REJECT   # tiêu chí không đủ y
    assert m["c4"]["accepted"] is False and any("no verdict" in r for r in m["c4"]["reasons"])


def test_choose_variants_keeps_at_most_one_per_parent():
    cands = [{"parent_id": "5", "candidate_id": "5__bt1", "rank": 1}, {"parent_id": "5", "candidate_id": "5__bt0", "rank": 0},
             {"parent_id": "7", "candidate_id": "7__bt0", "rank": 0}]
    merged = {"5__bt0": {"accepted": True}, "5__bt1": {"accepted": True}, "7__bt0": {"accepted": False}}
    chosen = A.choose_variants(cands, merged)
    assert list(chosen) == ["5"] and chosen["5"]["candidate_id"] == "5__bt0"


def test_cohen_kappa():
    assert A.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert A.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert A.cohen_kappa([None], ["a"]) is None


def test_apply_augmentation_only_adds_variants_of_training_portion():
    label2id = {"not-sarcasm": 0, "text-sarcasm": 1}
    train = [{"id": "1", "caption": "gốc 1", "label": 1, "image_path": "x.jpg", "ocr": "o"},
             {"id": "2", "caption": "gốc 2", "label": 0, "image_path": "y.jpg"}]
    held_out = [{"id": "3", "caption": "gốc 3", "label": 1, "image_path": "z.jpg"}]
    splits = {"train": train, "val": held_out}
    variants = [{"variant_id": "1__bt0", "parent_id": "1", "label": "text-sarcasm", "caption": "biến thể 1"},
                {"variant_id": "3__bt0", "parent_id": "3", "label": "text-sarcasm", "caption": "biến thể 3"}]
    out = apply_augmentation(splits, variants, label2id)
    added = out["train"][2:]
    assert [r["id"] for r in added] == ["1__bt0"]           # mẫu gốc 3 đang ở val: không thêm
    assert added[0]["parent_id"] == "1" and added[0]["image_path"] == "x.jpg" and added[0]["ocr"] == "o"
    assert added[0]["caption"] == "biến thể 1" and added[0]["label"] == 1
    assert out["val"] is held_out and len(train) == 2      # không sửa đầu vào
    with pytest.raises(ValueError, match="khác nhãn"):
        apply_augmentation(splits, [{**variants[0], "label": "not-sarcasm"}], label2id)
