import copy
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from vimmsd.analysis.interpretability import merge_subwords
from vimmsd.data.dataset import ViMMSDDataset
from vimmsd.training.metrics import compute_metrics
from vimmsd.training.trainer import predict

REPEATED_PUNCT_RE = re.compile(r"[^\w\s]{2,}")  # "!!!", "...", ":))"


def derangement(n, rng):
    """Hoán vị không có điểm bất động: đảm bảo mẫu nào cũng nhận ảnh của mẫu khác."""
    if n < 2:
        raise ValueError("cần ít nhất 2 mẫu để tráo ảnh")
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def swap_images(records, seed=0, image_pool=None):
    """Giữ nguyên text, thay ảnh của từng mẫu bằng ảnh của mẫu khác.
    image_pool=None: tráo ảnh trong chính tập records; nếu truyền list records khác
    (ví dụ toàn bộ mẫu not-sarcasm) thì lấy ảnh ngẫu nhiên từ pool đó."""
    rng = np.random.default_rng(seed)
    swapped = copy.deepcopy(records)
    if image_pool is None:
        perm = derangement(len(records), rng)
        for rec, j in zip(swapped, perm):
            rec["image_path"] = records[j]["image_path"]
    else:
        for rec in swapped:
            choices = [r for r in image_pool if r["image_path"] != rec["image_path"]]
            rec["image_path"] = choices[rng.integers(len(choices))]["image_path"]
    return swapped


def run_shortcut_check(model, records, collator, labels, device, target_labels=("multi-sarcasm", "image-sarcasm"),
                       text_preprocessor=None, image_pool=None, batch_size=32, seed=0, use_amp=False, cache_dir=None):
    """Nếu model thực sự dùng ảnh thì dự đoán cho các mẫu sarcasm liên quan tới ảnh phải đổi khi tráo ảnh.
    flip_rate thấp -> model chủ yếu dựa vào text (shortcut learning)."""
    target_ids = {labels.index(l) for l in target_labels}
    subset = [r for r in records if r["label"] in target_ids]
    swapped = swap_images(subset, seed=seed, image_pool=image_pool)

    def run(recs):
        ds = ViMMSDDataset(recs, text_preprocessor=text_preprocessor, cache_dir=cache_dir)
        return predict(model, DataLoader(ds, batch_size=batch_size, collate_fn=collator), device, use_amp)

    orig, swap = run(subset), run(swapped)
    y = np.array(orig["labels"])
    p_orig, p_swap = np.array(orig["preds"]), np.array(swap["preds"])
    changed = p_orig != p_swap
    # total variation distance giữa 2 phân phối xác suất, 0 = không đổi, 1 = đổi hoàn toàn
    tv = 0.5 * (orig["probs"] - swap["probs"]).abs().sum(-1).numpy()

    per_class = {}
    for cid in sorted(target_ids):
        m = y == cid
        per_class[labels[cid]] = {
            "n": int(m.sum()),
            "flip_rate": float(changed[m].mean()),
            "acc_original": float((p_orig[m] == cid).mean()),
            "acc_swapped": float((p_swap[m] == cid).mean()),
            "mean_prob_shift": float(tv[m].mean()),
        }
    return {
        "n": len(subset),
        "flip_rate": float(changed.mean()),
        "acc_original": float((p_orig == y).mean()),
        "acc_swapped": float((p_swap == y).mean()),
        "mean_prob_shift": float(tv.mean()),
        "per_class": per_class,
        "ids": orig["ids"],
        "preds_original": p_orig.tolist(),
        "preds_swapped": p_swap.tolist(),
    }


def swap_image_views(dataset, perm):
    """Bản sao dataset, mỗi mẫu nhận CÙNG LÚC ảnh, OCR và mô tả của mẫu perm[i], caption giữ nguyên
    (docs/experiment_design.md mục 6.6). Chỉ đổi ảnh thì model vẫn thấy nội dung ảnh gốc qua OCR và mô tả.
    OCR của mẫu cho đã được làm sạch theo caption của chính nó, nên có thể khác chút so với khi ghép với caption mới."""
    swapped = copy.copy(dataset)
    swapped.records = [{**rec, "image_path": dataset.records[j]["image_path"]} for rec, j in zip(dataset.records, perm)]
    for attr in ("ocrs", "descriptions", "image_texts"):
        values = getattr(dataset, attr)
        if values is not None:
            setattr(swapped, attr, [values[j] for j in perm])
    return swapped


def strip_surface_signals(caption):
    """Xoá emoji và chuỗi dấu câu lặp trong caption thô (trước tiền xử lý, vì emoji bị đổi thành chữ ở đó)."""
    import emoji

    text = emoji.replace_emoji(caption or "", replace=" ")
    return " ".join(REPEATED_PUNCT_RE.sub(" ", text).split())


def without_surface_signals(dataset, preprocess):
    """Bản sao dataset đã xoá tín hiệu bề mặt khỏi caption, kèm mask các mẫu bị đổi. Mẫu không đổi giữ text đã xử lý;
    `preprocess` chỉ gọi cho mẫu bị đổi, và không ghi vào file cache caption dùng chung."""
    stripped = copy.copy(dataset)
    texts, changed, memo = [], [], {}
    for rec, text in zip(dataset.records, dataset.texts):
        clean = strip_surface_signals(rec["caption"])
        is_changed = clean != " ".join((rec["caption"] or "").split())
        if is_changed and clean not in memo:
            memo[clean] = preprocess(clean)
        texts.append(memo[clean] if is_changed else text)
        changed.append(is_changed)
    stripped.texts = texts
    return stripped, np.array(changed)


def compare_predictions(base, other, labels, mask=None):
    """So dự đoán trước và sau khi thay đổi đầu vào. mean_prob_shift: total variation distance giữa 2 phân phối xác
    suất; cần xem cùng flip_rate vì model nghiêng hẳn về một lớp có thể ít đổi nhãn dù xác suất đã dịch nhiều."""
    if base["ids"] != other["ids"]:
        raise ValueError("hai lần dự đoán phải cùng thứ tự mẫu")
    y, p_base, p_other = np.array(base["labels"]), np.array(base["preds"]), np.array(other["preds"])
    tv = 0.5 * (base["probs"] - other["probs"]).abs().sum(-1).numpy()
    mask = np.ones(len(y), dtype=bool) if mask is None else mask
    per_class = {}
    for i, name in enumerate(labels):
        m = mask & (y == i)
        if m.any():
            per_class[name] = {"n": int(m.sum()), "flip_rate": float((p_base[m] != p_other[m]).mean()),
                               "mean_prob_shift": float(tv[m].mean())}
    return {
        "n": int(mask.sum()),
        "flip_rate": float((p_base[mask] != p_other[mask]).mean()) if mask.any() else float("nan"),
        "mean_prob_shift": float(tv[mask].mean()) if mask.any() else float("nan"),
        "per_class": per_class,
        "metrics": compute_metrics(y, p_other, labels),
    }


def integrated_gradients(model, batch, steps=20):
    """Integrated Gradients của log xác suất lớp dự đoán theo embedding token CAPTION. Baseline: embedding <pad> ở
    mọi token thường, token đặc biệt giữ nguyên. OCR, mô tả và ảnh cố định. completeness_error: |tổng điểm - (f(x) -
    f(baseline))|, nhỏ thì đủ số bước. Trả về điểm (B, L), lớp đích (B,), completeness_error (B,)."""
    model.eval()
    encoder = model.text_encoder.model
    embed = encoder.get_input_embeddings()
    ids, mask = batch["input_ids"], batch["attention_mask"]
    special = torch.isin(ids, torch.tensor(model.tokenizer.all_special_ids, device=ids.device))
    with torch.no_grad():
        text, image = model.encode(batch)
        inputs = embed(ids)
        baseline = torch.where(special.unsqueeze(-1), inputs, embed(torch.full_like(ids, model.tokenizer.pad_token_id)))

    def log_probs(embeds):
        tokens = encoder(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        return model.classify({**text, "tokens": tokens, "mask": mask.bool(), "pooled": tokens[:, 0]}, image).log_softmax(-1)

    with torch.no_grad():
        start, end = log_probs(baseline), log_probs(inputs)
        target = end.argmax(-1, keepdim=True)
    total = torch.zeros_like(inputs)
    for alpha in (torch.arange(steps, device=ids.device).float() + 0.5) / steps:  # quy tắc trung điểm
        point = (baseline + alpha * (inputs - baseline)).requires_grad_(True)
        total += torch.autograd.grad(log_probs(point).gather(1, target).sum(), point)[0]
    scores = ((inputs - baseline) * total / steps).sum(-1) * mask
    delta = (end - start).gather(1, target).squeeze(1)
    return scores.detach(), target.squeeze(1), (scores.sum(1) - delta).abs().detach()


def stratified_indices(classes, per_class, seed=0):
    """Tối đa per_class chỉ số cho mỗi giá trị trong `classes`."""
    rng = np.random.default_rng(seed)
    classes = np.asarray(classes)
    out = []
    for c in np.unique(classes):
        idx = np.flatnonzero(classes == c)
        out.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    return sorted(out)


def rank_words(tokenizer, attributions, labels, min_count=5, top_k=20):
    """attributions: list (input_ids, điểm, lớp dự đoán). Xếp hạng từ theo điểm trung bình trong từng lớp dự đoán,
    kèm punct_share: tỉ lệ độ lớn điểm nằm ở token không có chữ hay số."""
    per_class = defaultdict(lambda: defaultdict(list))
    magnitude, punct = defaultdict(float), defaultdict(float)
    for input_ids, scores, pred in attributions:
        name = labels[pred]
        for word, score in zip(*merge_subwords(tokenizer, input_ids, scores)):
            per_class[name][word.lower()].append(score)
            magnitude[name] += abs(score)
            if not any(ch.isalnum() for ch in word):
                punct[name] += abs(score)
    out = {}
    for name, words in per_class.items():
        ranked = sorted(((w, float(np.mean(s)), len(s)) for w, s in words.items() if len(s) >= min_count),
                        key=lambda row: row[1], reverse=True)
        out[name] = {"punct_share": punct[name] / magnitude[name] if magnitude[name] else 0.0,
                     "top_words": [{"word": w, "mean_attribution": a, "count": n} for w, a, n in ranked[:top_k]]}
    return out
