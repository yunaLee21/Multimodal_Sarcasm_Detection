"""Experiment B: mức phụ thuộc của model vào từng nguồn input (docs/empirical_analysis_plan.md mục 7), không train lại.

Can thiệp trên CÙNG các mẫu val, caption và id giữ nguyên:
- pixels: chỉ thay ảnh (OCR, mô tả giữ của mẫu gốc);
- ocr / description: chỉ thay OCR / mô tả (model M);
- joint: thay đồng thời ảnh, OCR, mô tả từ một mẫu cho;
- remove_ocr / remove_description: bỏ hẳn stream đó (chuỗi rỗng -> cờ *_present = 0 của collator).
Chính sách OCR: OCR của mẫu cho đã được làm sạch theo caption của chính mẫu cho (clean_ocr lúc dựng dataset) và giữ
nguyên như vậy khi ghép với caption khác. Mẫu cho lấy từ hoán vị cố định không có điểm bất động (donor manifest),
dùng chung cho mọi model và seed. Nhãn gốc không còn chắc đúng sau khi thay: thay đổi điểm đo mức xáo trộn, không phải
độ chính xác trên một bài toán mới."""
import copy

import numpy as np

INTERVENTIONS = ("pixels", "ocr", "description", "joint", "remove_ocr", "remove_description")
NEEDS_STREAMS = {"ocr", "description", "remove_ocr", "remove_description"}


def derangement(n, rng):
    if n < 2:
        raise ValueError("cần ít nhất 2 mẫu")
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def make_donor_manifest(ids, labels, n_perm=5, seed=0, policy="random"):
    """{"ids", "policy", "seed", "permutations": [[chỉ số mẫu cho theo vị trí]]}. policy=same_label: mẫu cho cùng nhãn
    (đối chứng riêng, không trộn với random). Lớp chỉ có 1 mẫu thì không có mẫu cho cùng nhãn: báo lỗi."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    perms = []
    for _ in range(n_perm):
        if policy == "random":
            perms.append(derangement(len(ids), rng).tolist())
        elif policy == "same_label":
            perm = np.arange(len(ids))
            for c in np.unique(labels):
                idx = np.flatnonzero(labels == c)
                if len(idx) < 2:
                    raise ValueError(f"lớp {c} chỉ có {len(idx)} mẫu, không dựng được mẫu cho cùng nhãn")
                perm[idx] = idx[derangement(len(idx), rng)]
            perms.append(perm.tolist())
        else:
            raise ValueError(f"policy không hợp lệ: {policy}")
    return {"ids": list(ids), "policy": policy, "seed": seed, "permutations": perms}


def intervene(dataset, kind, perm=None):
    """Bản sao nông của ViMMSDDataset sau can thiệp `kind`. perm: chỉ số mẫu cho theo vị trí (trừ remove_*)."""
    if kind not in INTERVENTIONS:
        raise ValueError(f"can thiệp không hợp lệ: {kind}")
    if kind in NEEDS_STREAMS and dataset.ocrs is None:
        raise ValueError(f"{kind} cần model có stream OCR/mô tả riêng (multistream)")
    out = copy.copy(dataset)
    if kind in ("pixels", "joint"):
        if not dataset.load_image:
            raise ValueError(f"{kind}: model không dùng ảnh")
        out.records = [{**rec, "image_path": dataset.records[j]["image_path"]} for rec, j in zip(dataset.records, perm)]
    if kind in ("ocr", "joint") and dataset.ocrs is not None:
        out.ocrs = [dataset.ocrs[j] for j in perm]
    if kind in ("description", "joint") and dataset.descriptions is not None:
        out.descriptions = [dataset.descriptions[j] for j in perm]
    if kind == "joint" and dataset.image_texts is not None:
        out.image_texts = [dataset.image_texts[j] for j in perm]
    if kind == "remove_ocr":
        out.ocrs = [""] * len(dataset)
    if kind == "remove_description":
        out.descriptions = [""] * len(dataset)
    return out


def applicable(dataset, kind):
    if kind in ("pixels", "joint"):
        return bool(dataset.load_image) or (kind == "joint" and (dataset.ocrs is not None or dataset.image_texts is not None))
    return dataset.ocrs is not None


def disruption(base_probs, new_probs, y, n_classes):
    """So sánh dự đoán gốc và sau can thiệp. Trả về tổng thể và theo lớp gốc: flip_rate, tv_shift (nửa chuẩn L1),
    confidence_change (xác suất lớn nhất), prob_change_of_original_pred, ma trận chuyển dự đoán, accuracy/macro-F1
    so với nhãn gốc trước và sau."""
    from vimmsd.analysis.stats import macro_f1

    base_probs, new_probs, y = np.asarray(base_probs), np.asarray(new_probs), np.asarray(y)
    bp, np_ = base_probs.argmax(1), new_probs.argmax(1)
    rows = np.arange(len(y))

    def summarize(mask):
        if not mask.any():
            return None
        trans = np.zeros((n_classes, n_classes), int)
        np.add.at(trans, (bp[mask], np_[mask]), 1)
        return {"n": int(mask.sum()),
                "flip_rate": float(np.mean(bp[mask] != np_[mask])),
                "tv_shift": float(np.mean(0.5 * np.abs(base_probs[mask] - new_probs[mask]).sum(1))),
                "confidence_change": float(np.mean(new_probs[mask].max(1) - base_probs[mask].max(1))),
                "prob_change_of_original_pred": float(np.mean(new_probs[rows[mask], bp[mask]] - base_probs[rows[mask], bp[mask]])),
                "accuracy_before": float(np.mean(bp[mask] == y[mask])),
                "accuracy_after": float(np.mean(np_[mask] == y[mask])),
                "transitions": trans.tolist()}

    overall = summarize(np.ones(len(y), bool))
    overall["macro_f1_before"] = macro_f1(y, bp, n_classes)
    overall["macro_f1_after"] = macro_f1(y, np_, n_classes)
    return {"overall": overall, "by_true_class": {int(c): summarize(y == c) for c in range(n_classes)}}


def average_over_permutations(results):
    """Trung bình các chỉ số số học qua các hoán vị; ma trận chuyển cộng dồn."""
    def avg(items):
        items = [x for x in items if x is not None]
        if not items:
            return None
        out = {}
        for k in items[0]:
            vals = [x[k] for x in items]
            out[k] = np.sum(vals, axis=0).tolist() if k == "transitions" else float(np.mean(vals))
        out["n_permutations"] = len(items)
        return out

    classes = results[0]["by_true_class"].keys()
    return {"overall": avg([r["overall"] for r in results]),
            "by_true_class": {c: avg([r["by_true_class"][c] for r in results]) for c in classes}}
