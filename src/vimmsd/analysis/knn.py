"""E1: kNN retrieval lúc suy luận (docs/experiment_design.md mục 5.5). Không train lại, bộ nhớ chỉ chứa train."""
import numpy as np
import torch
from tqdm import tqdm

from vimmsd.training.metrics import compute_metrics
from vimmsd.training.trainer import move_to_device, selection_score


@torch.no_grad()
def collect_outputs(model, loader, device, use_amp=False):
    """z đã chuẩn hoá L2 và xác suất của model (theo prior_shift đang đặt trên model), cùng thứ tự với loader."""
    model.eval()
    zs, probs, labels, ids = [], [], [], []
    for batch in tqdm(loader, desc="z", leave=False):
        batch = move_to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model.forward_outputs(batch)
        zs.append(torch.nn.functional.normalize(out["z"].float(), dim=-1).cpu())
        probs.append(out["logits"].float().softmax(-1).cpu())
        labels.extend(batch["labels"].tolist())
        ids.extend(batch["ids"])
    return {"z": torch.cat(zs).numpy(), "probs": torch.cat(probs).numpy(), "labels": np.array(labels), "ids": ids}


def nearest_neighbors(query, memory, k, batch_size=1024):
    """Top-k cosine (z đã chuẩn hoá) của từng query. Trả về (sim, chỉ số), giảm dần theo sim."""
    if k > len(memory):
        raise ValueError(f"k={k} lớn hơn số mẫu trong bộ nhớ {len(memory)}")
    q, m = torch.from_numpy(query), torch.from_numpy(memory)
    if torch.cuda.is_available():
        q, m = q.cuda(), m.cuda()
    sims, idx = [], []
    for start in range(0, len(q), batch_size):
        values, indices = (q[start:start + batch_size] @ m.T).topk(k, dim=-1)
        sims.append(values.cpu())
        idx.append(indices.cpu())
    return torch.cat(sims).numpy(), torch.cat(idx).numpy()


def knn_probs(sims, neighbor_labels, k, tau, train_counts, target_counts):
    """p_kNN(c) ∝ w_c · prior_đích(c) / prior_train(c), w_c = Σ exp(sim_j / τ) trên k láng giềng thuộc lớp c.
    Đếm lớp kẹp tối thiểu 1 như compute_prior_shift. Trừ sim lớn nhất trước khi mũ hoá để không tràn số với τ nhỏ."""
    train = np.maximum(np.asarray(train_counts, dtype=float), 1)
    target = np.maximum(np.asarray(target_counts, dtype=float), 1)
    ratio = (target / target.sum()) / (train / train.sum())
    weights = np.exp((sims[:, :k] - sims[:, :1]) / tau)
    out = np.zeros((len(sims), len(train)))
    rows = np.repeat(np.arange(len(sims)), k)
    np.add.at(out, (rows, neighbor_labels[:, :k].ravel()), weights.ravel())
    out *= ratio
    return out / out.sum(1, keepdims=True)


def _evaluate(labels, probs, names, metric, exclude, mask=None):
    y, p = (labels, probs) if mask is None else (labels[mask], probs[mask])
    m = compute_metrics(y, p.argmax(-1), names)
    return {"score": selection_score(m, metric, exclude), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "weighted_f1": m["weighted_f1"], "f1": {k: v["f1"] for k, v in m["per_class"].items()}}


def grid_search(labels, model_probs, sims, neighbor_labels, train_counts, target_counts, names, grid, metric, exclude, mask=None):
    """Dò k, τ, λ theo tiêu chí chọn checkpoint. λ = 1 (không retrieval) luôn nằm trong lưới nên điểm không thể thấp hơn
    mốc: cần grid_search_cv để biết phần tăng còn lại ngoài mẫu đã dò. mask: chỉ chấm trên các mẫu này."""
    rows = [{"k": None, "tau": None, "lambda": 1.0, **_evaluate(labels, model_probs, names, metric, exclude, mask)}]
    for k in grid["k"]:
        for tau in grid["tau"]:
            nn_probs = knn_probs(sims, neighbor_labels, k, tau, train_counts, target_counts)
            for lam in grid["lambda"]:
                if lam == 1:
                    continue
                final = lam * model_probs + (1 - lam) * nn_probs
                rows.append({"k": k, "tau": tau, "lambda": lam, **_evaluate(labels, final, names, metric, exclude, mask)})
    return rows


def _best(rows):
    return max(rows, key=lambda r: (r["score"], r["lambda"]))  # bằng điểm thì ưu tiên ít retrieval hơn


def grid_search_cv(labels, model_probs, sims, neighbor_labels, train_counts, target_counts, names, grid, metric, exclude,
                   folds=5, seed=0):
    """Ước lượng ngoài mẫu: chọn tham số trên các fold còn lại của val, ghép dự đoán của từng fold rồi chấm chung.
    Chia theo lớp để mỗi fold đều có `image-sarcasm`."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(labels), dtype=int)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % folds
    final, chosen = np.zeros_like(model_probs), []
    for f in range(folds):
        best = _best(grid_search(labels, model_probs, sims, neighbor_labels, train_counts, target_counts, names, grid, metric,
                                 exclude, mask=fold != f))
        chosen.append({"k": best["k"], "tau": best["tau"], "lambda": best["lambda"]})
        probs = model_probs
        if best["lambda"] != 1:
            nn_probs = knn_probs(sims, neighbor_labels, best["k"], best["tau"], train_counts, target_counts)
            probs = best["lambda"] * model_probs + (1 - best["lambda"]) * nn_probs
        final[fold == f] = probs[fold == f]
    return {**_evaluate(labels, final, names, metric, exclude), "chosen": chosen}


def similarity_stats(sims, neighbor_labels, labels):
    top1 = sims[:, 0]
    return {"top1_mean": float(top1.mean()), "top1_quantiles": {str(q): float(np.quantile(top1, q)) for q in (0.1, 0.5, 0.9)},
            "top1_at_least_0.9": float((top1 >= 0.9).mean()), "top1_label_match": float((neighbor_labels[:, 0] == labels).mean())}
