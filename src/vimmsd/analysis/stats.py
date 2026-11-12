"""Thống kê cho nghiên cứu thực nghiệm (docs/empirical_analysis_plan.md mục 11), scripts/empirical_stats.py.

- Bootstrap ghép cặp: lấy lại mẫu đánh giá có hoàn lại, dùng CÙNG chỉ số cho cả hai model (và mọi seed), tính hiệu
  macro-F1. Khoảng tin cậy chỉ phản ánh biến động do mẫu đánh giá, không phản ánh ngẫu nhiên khi train.
- Bootstrap theo nhóm: lấy lại cả nhóm (caption hoặc ảnh trùng) thay cho từng mẫu, làm phân tích độ nhạy.
- Holm trong một họ so sánh khai báo trước."""
import numpy as np
import pandas as pd


def macro_f1(y, pred, n_classes, exclude=()):
    """Macro-F1 trên các lớp 0..n_classes-1 trừ `exclude` (chỉ số). Lớp không có mẫu thật và không được dự đoán
    tính F1 = 0 như sklearn (zero_division=0), để khớp compute_metrics của repo."""
    y, pred = np.asarray(y), np.asarray(pred)
    f1s = []
    for c in range(n_classes):
        if c in exclude:
            continue
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def bootstrap_indices(n, n_boot, seed, groups=None):
    """Mảng (n_boot, ≈n) chỉ số. groups: mảng nhãn nhóm độ dài n -> lấy lại nhóm có hoàn lại rồi nối các mẫu."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return rng.integers(0, n, size=(n_boot, n))
    codes, uniq = pd.factorize(pd.Series(groups))
    members = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        out.append(np.concatenate([members[g] for g in pick]))
    return out


def paired_contrast(y, preds_a, preds_b, n_classes, exclude=(), n_boot=2000, seed=0, groups=None):
    """preds_a, preds_b: list dự đoán theo seed (cùng thứ tự mẫu, cùng số seed, ghép theo vị trí).
    Trả về hiệu từng seed, hiệu trung bình, và khoảng 95% + p hai phía của hiệu trung bình qua bootstrap."""
    y = np.asarray(y)
    a, b = [np.asarray(p) for p in preds_a], [np.asarray(p) for p in preds_b]
    if len(a) != len(b):
        raise ValueError("số seed của hai model phải bằng nhau để ghép cặp")
    per_seed = [macro_f1(y, pa, n_classes, exclude) - macro_f1(y, pb, n_classes, exclude) for pa, pb in zip(a, b)]
    boots = []
    for idx in bootstrap_indices(len(y), n_boot, seed, groups):
        yy = y[idx]
        boots.append(np.mean([macro_f1(yy, pa[idx], n_classes, exclude) - macro_f1(yy, pb[idx], n_classes, exclude)
                              for pa, pb in zip(a, b)]))
    boots = np.asarray(boots)
    p = min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return {"per_seed": per_seed, "mean": float(np.mean(per_seed)), "ci95": [float(np.quantile(boots, 0.025)),
            float(np.quantile(boots, 0.975))], "p_boot": float(p), "n_boot": n_boot,
            "resampling": "groups" if groups is not None else "examples"}


def holm(pvalues):
    """p đã hiệu chỉnh Holm–Bonferroni, cùng thứ tự đầu vào."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def per_class(y, pred, n_classes):
    """precision, recall, F1, support từng lớp."""
    y, pred = np.asarray(y), np.asarray(pred)
    out = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        npred, ntrue = np.sum(pred == c), np.sum(y == c)
        prec = tp / npred if npred else 0.0
        rec = tp / ntrue if ntrue else 0.0
        out.append({"precision": float(prec), "recall": float(rec),
                    "f1": float(0.0 if tp == 0 else 2 * tp / (npred + ntrue)), "support": int(ntrue)})
    return out


def subset_table(y, preds_by_model, masks, n_classes, exclude=()):
    """masks: {tên subset: mảng bool}. Mỗi dòng: subset, support, phân bố nhãn, macro-F1 trung bình theo seed của
    từng model, tính trên các lớp CÓ MẶT trong subset (lớp vắng mặt không bị tính F1 = 0). Subset nhỏ chỉ mang tính
    mô tả (mục 6)."""
    y = np.asarray(y)
    rows = []
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        counts = np.bincount(y[mask], minlength=n_classes) if mask.any() else np.zeros(n_classes, int)
        skip = tuple(set(exclude) | {c for c in range(n_classes) if counts[c] == 0})
        row = {"subset": name, "support": int(mask.sum()), "labels": counts.tolist(),
               "classes_scored": [c for c in range(n_classes) if c not in skip]}
        for model, preds in preds_by_model.items():
            row[model] = (float(np.mean([macro_f1(y[mask], np.asarray(p)[mask], n_classes, skip) for p in preds]))
                          if mask.any() else None)
        rows.append(row)
    return rows
