"""Test sheet giải thích do người viết. Không cần GPU, mạng, model thật."""
import itertools
import json

import pytest

from vimmsd.data.human_rationale import collect_drafts, interleave_batches, label_from_flags, multi_case, sample_ids


@pytest.mark.parametrize("flags,label,case", [
    ((False, False, False), "not-sarcasm", None),
    ((True, False, False), "text-sarcasm", None),
    ((False, True, False), "image-sarcasm", None),
    ((True, True, False), "multi-sarcasm", 1),
    ((True, True, True), "multi-sarcasm", 1),
    ((True, False, True), "multi-sarcasm", 2),
    ((False, True, True), "multi-sarcasm", 3),
    ((False, False, True), "multi-sarcasm", 4),
])
def test_decision_table(flags, label, case):
    assert label_from_flags(*flags) == label and multi_case(*flags) == case


def test_sample_respects_quota_and_exclusions():
    labels = {str(i): "image-sarcasm" if i < 10 else "not-sarcasm" for i in range(40)}
    labels["573"] = "image-sarcasm"  # mẫu ví dụ trong prompt
    chosen = sample_ids(labels, {"image-sarcasm": 20, "not-sarcasm": 5}, exclude={"0"})
    assert len(chosen["image-sarcasm"]) == 9 and "0" not in chosen["image-sarcasm"] and "573" not in chosen["image-sarcasm"]
    assert len(chosen["not-sarcasm"]) == 5
    assert chosen == sample_ids(labels, {"image-sarcasm": 20, "not-sarcasm": 5}, exclude={"0"})


def test_batches_keep_class_mix():
    chosen = {"a": [f"a{i}" for i in range(8)], "b": [f"b{i}" for i in range(4)]}
    out = interleave_batches(chosen, 4)
    assert sorted(s for _, s in out) == sorted(chosen["a"] + chosen["b"])
    for b, group in itertools.groupby(out, key=lambda x: x[0]):
        ids = [s for _, s in group]
        assert sum(s.startswith("a") for s in ids) == 2 and sum(s.startswith("b") for s in ids) == 1


def test_drafts_only_from_correct_unlabelled_outputs(tmp_path):
    def write(name, rows):
        d = tmp_path / name
        d.mkdir()
        (d / "train.shard0of1.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return d
    ok = {"label": "multi-sarcasm", "text_reason": "a"}
    first = write("v8", [{"id": "1", "split": "train", "label": "multi-sarcasm", "with_label": False, "parsed": ok},
                         {"id": "2", "split": "train", "label": "not-sarcasm", "with_label": False, "parsed": ok},
                         {"id": "3", "split": "train", "label": "multi-sarcasm", "with_label": True, "parsed": ok}])
    second = write("v7", [{"id": "1", "split": "train", "label": "multi-sarcasm", "with_label": False, "parsed": {**ok, "text_reason": "b"}},
                          {"id": "4", "split": "train", "label": "multi-sarcasm", "with_label": False, "parsed": None}])
    drafts = collect_drafts([first, second])
    assert set(drafts) == {"1"} and drafts["1"] == ("v8", ok)
