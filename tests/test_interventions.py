"""Test can thiệp input của Experiment B (src/vimmsd/analysis/interventions.py). Không cần model."""
import numpy as np
import pytest

from vimmsd.analysis import interventions as I


class _DS:
    def __init__(self, n, streams=True, load_image=True):
        self.records = [{"id": str(i), "image_path": f"img{i}.jpg", "caption": f"cap{i}", "label": i % 2} for i in range(n)]
        self.texts = [f"cap{i}" for i in range(n)]
        self.ocrs = [f"ocr{i}" for i in range(n)] if streams else None
        self.descriptions = [f"desc{i}" for i in range(n)] if streams else None
        self.image_texts = None
        self.load_image = load_image

    def __len__(self):
        return len(self.records)


def test_donor_manifest_no_self_donors_reproducible_and_same_label():
    ids, labels = [str(i) for i in range(20)], [i % 2 for i in range(20)]
    m = I.make_donor_manifest(ids, labels, n_perm=5, seed=3)
    assert len(m["permutations"]) == 5 and m == I.make_donor_manifest(ids, labels, n_perm=5, seed=3)
    for p in m["permutations"]:
        assert sorted(p) == list(range(20)) and all(p[i] != i for i in range(20))
    s = I.make_donor_manifest(ids, labels, n_perm=2, seed=0, policy="same_label")
    for p in s["permutations"]:
        assert all(labels[p[i]] == labels[i] and p[i] != i for i in range(20))
    with pytest.raises(ValueError, match="chỉ có 1"):
        I.make_donor_manifest(["a", "b", "c"], [0, 0, 1], policy="same_label")


def test_interventions_change_only_their_input():
    ds = _DS(4)
    perm = [1, 2, 3, 0]
    px = I.intervene(ds, "pixels", perm)
    assert [r["image_path"] for r in px.records] == ["img1.jpg", "img2.jpg", "img3.jpg", "img0.jpg"]
    assert px.ocrs == ds.ocrs and px.descriptions == ds.descriptions and px.texts == ds.texts
    assert [r["id"] for r in px.records] == ["0", "1", "2", "3"]          # id, caption, nhãn giữ nguyên
    oc = I.intervene(ds, "ocr", perm)
    assert oc.ocrs == ["ocr1", "ocr2", "ocr3", "ocr0"] and oc.descriptions == ds.descriptions and oc.records is ds.records
    de = I.intervene(ds, "description", perm)
    assert de.descriptions[0] == "desc1" and de.ocrs == ds.ocrs
    jo = I.intervene(ds, "joint", perm)
    assert jo.records[0]["image_path"] == "img1.jpg" and jo.ocrs[0] == "ocr1" and jo.descriptions[0] == "desc1"
    assert jo.texts == ds.texts
    assert I.intervene(ds, "remove_ocr").ocrs == [""] * 4 and I.intervene(ds, "remove_ocr").descriptions == ds.descriptions
    assert ds.ocrs == ["ocr0", "ocr1", "ocr2", "ocr3"]                      # không sửa bản gốc


def test_interventions_reject_inapplicable_models():
    concat = _DS(3, streams=False)
    assert I.applicable(concat, "pixels") and not I.applicable(concat, "ocr")
    with pytest.raises(ValueError, match="multistream"):
        I.intervene(concat, "ocr", [1, 2, 0])
    text_streams = _DS(3, streams=True, load_image=False)
    with pytest.raises(ValueError, match="không dùng ảnh"):
        I.intervene(text_streams, "pixels", [1, 2, 0])


def test_disruption_metrics():
    base = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    new = np.array([[0.9, 0.1], [0.7, 0.3], [0.6, 0.4]])
    y = np.array([0, 1, 1])
    d = I.disruption(base, new, y, 2)
    o = d["overall"]
    assert o["flip_rate"] == pytest.approx(1 / 3) and o["tv_shift"] == pytest.approx(0.5 / 3)
    assert o["transitions"] == [[2, 0], [1, 0]] and o["accuracy_before"] == pytest.approx(2 / 3)
    assert o["accuracy_after"] == pytest.approx(1 / 3)
    assert d["by_true_class"][1]["flip_rate"] == pytest.approx(0.5) and d["by_true_class"][0]["flip_rate"] == 0
    assert o["prob_change_of_original_pred"] == pytest.approx(-0.5 / 3)
    avg = I.average_over_permutations([d, d])
    assert avg["overall"]["flip_rate"] == pytest.approx(1 / 3) and avg["overall"]["transitions"] == [[4, 0], [2, 0]]
