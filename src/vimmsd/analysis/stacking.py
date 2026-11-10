"""B2: gộp dự đoán của 3 model riêng bằng một classifier nhỏ (docs/experiment_design.md mục 5.6).

Classifier gộp chỉ học trên dự đoán out-of-fold của train: dự đoán trên chính dữ liệu model đã học quá tự tin."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from vimmsd.training.metrics import compute_metrics
from vimmsd.training.trainer import selection_score


def assemble_oof(fold_predictions, folds, ids):
    """fold_predictions: {fold: {"ids": [...], "probs": (n, C)}} của model train khi giữ lại fold đó.
    Trả về ma trận (len(ids), C) theo thứ tự `ids`. Báo lỗi nếu một mẫu được dự đoán bởi model đã học nó, bị dự
    đoán hai lần, hoặc thiếu dự đoán."""
    index = {sid: i for i, sid in enumerate(ids)}
    out, seen = None, np.zeros(len(ids), dtype=bool)
    for fold, pred in fold_predictions.items():
        probs = np.asarray(pred["probs"], dtype=float)
        out = np.full((len(ids), probs.shape[1]), np.nan) if out is None else out
        for sid, p in zip(pred["ids"], probs):
            if folds[sid] != fold:
                raise ValueError(f"mẫu {sid} thuộc fold {folds[sid]} nhưng được dự đoán bởi model giữ lại fold {fold}, "
                                 "tức là model đã học mẫu này")
            if seen[index[sid]]:
                raise ValueError(f"mẫu {sid} được dự đoán hai lần")
            out[index[sid]], seen[index[sid]] = p, True
    if not seen.all():
        raise ValueError(f"{(~seen).sum()} mẫu không có dự đoán out-of-fold")
    return out


def features(prob_list):
    """Log xác suất của từng model, nối lại."""
    return np.concatenate([np.log(np.clip(np.asarray(p, dtype=float), 1e-6, 1)) for p in prob_list], axis=1)


def prior_ratio(train_counts, target_counts):
    train = np.maximum(np.asarray(train_counts, dtype=float), 1)
    target = np.maximum(np.asarray(target_counts, dtype=float), 1)
    return np.log(target / target.sum()) - np.log(train / train.sum())


def score(labels, probs, names, metric="macro_f1", exclude=()):
    m = compute_metrics(labels, np.asarray(probs).argmax(-1), names)
    return {"score": selection_score(m, metric, exclude), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "f1": {k: v["f1"] for k, v in m["per_class"].items()}}


def fit_candidates(x_train, y_train, x_eval, n_classes, shift, grid):
    """Mỗi tổ hợp C × class_weight × có/không dịch prior: classifier học trên x_train, trả về xác suất trên x_eval."""
    out = {}
    for c in grid["C"]:
        for weight in grid["class_weight"]:
            clf = LogisticRegression(C=c, class_weight=weight, max_iter=2000)
            clf.fit(x_train, y_train)
            logits = np.full((len(x_eval), n_classes), -1e9)
            logits[:, clf.classes_] = clf.decision_function(x_eval)
            for shifted in (False, True):
                z = logits + shift if shifted else logits
                z = z - z.max(1, keepdims=True)
                p = np.exp(z)
                out[(c, weight or "none", shifted)] = p / p.sum(1, keepdims=True)
    return out


def cross_validated(labels, candidates, names, metric, exclude, folds=5, seed=0):
    """Chọn tổ hợp trên các fold còn lại của val, chấm trên fold giữ lại, ghép dự đoán rồi chấm chung (như E1)."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(labels), dtype=int)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % folds
    final, chosen = None, []
    for f in range(folds):
        keep = fold != f
        best = max(candidates, key=lambda k: score(labels[keep], candidates[k][keep], names, metric, exclude)["score"])
        chosen.append(list(map(str, best)))
        probs = candidates[best]
        final = np.zeros_like(probs) if final is None else final
        final[fold == f] = probs[fold == f]
    return {**score(labels, final, names, metric, exclude), "chosen": chosen}
