"""WS-A: how much do trained models depend on the image? Inference only (reports/plan/eacl_srw_run_plan.md §2).

The caption stays fixed and only the image-encoder output changes. The image features of every test sample are
computed once with the run's own frozen encoder. Each condition then maps sample i to an image source:
- swap: the features of another sample (a derangement, or a label-constrained map);
- mask: a constant feature (the mean over train images, or the encoding of a grey image).
The fusion and head run on (text_i, image_source(i)). A caption-only model (no image encoder) gets image=None
under every condition, so its predictions cannot change. That is the control.
"""
import numpy as np
import torch

from vimmsd.analysis.shortcut_check import derangement


# ---------------------------------------------------------------- image source maps

def same_label_derangement(labels, rng):
    """Image of another sample with the same gold label. A class with a single sample keeps its own image
    (reported via the returned `fixed` mask)."""
    labels = np.asarray(labels)
    src = np.arange(len(labels))
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) >= 2:
            src[idx] = idx[derangement(len(idx), rng)]
    return src, src == np.arange(len(labels))


def diff_label_map(labels, rng):
    """Image of a random sample whose gold label differs (sampling with replacement)."""
    labels = np.asarray(labels)
    src = np.empty(len(labels), dtype=int)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        pool = np.flatnonzero(labels != c)
        if len(pool) == 0:
            raise ValueError("diff-label swap needs at least two classes")
        src[idx] = rng.choice(pool, size=len(idx), replace=True)
    return src


def build_conditions(labels, swap_seeds):
    """{condition name: index map or the string 'mean' / 'grey'}. 'original' is the identity."""
    n = len(labels)
    conds = {"original": np.arange(n)}
    for s in swap_seeds:
        conds[f"swap_random_s{s}"] = derangement(n, np.random.default_rng(s))
        conds[f"swap_same_label_s{s}"] = same_label_derangement(labels, np.random.default_rng(1000 + s))[0]
        conds[f"swap_diff_label_s{s}"] = diff_label_map(labels, np.random.default_rng(2000 + s))
    conds["mask_mean"] = "mean"
    conds["mask_grey"] = "grey"
    return conds


# ---------------------------------------------------------------- feature cache + head passes

def _to(d, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


@torch.no_grad()
def encode_test_set(model, loader, device, use_amp):
    """One pass with the original inputs: per-batch text features (CPU) and per-sample image features (CPU).
    Returns dict(ids, labels, text_batches, batch_index, image) where image = None for a caption-only model."""
    from vimmsd.training.trainer import move_to_device

    model.eval()
    ids, labels, text_batches, batch_index, tokens, pooled = [], [], [], [], [], []
    start = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            text, image = model.encode(batch)
        n = len(batch["ids"])
        ids.extend(batch["ids"])
        labels.extend(batch["labels"].tolist())
        text_batches.append(None if text is None else _to(text, "cpu"))
        batch_index.append(np.arange(start, start + n))
        start += n
        if image is not None:
            tokens.append(image["tokens"].cpu())
            pooled.append(image["pooled"].cpu())
    image = {"tokens": torch.cat(tokens), "pooled": torch.cat(pooled)} if tokens else None
    return {"ids": ids, "labels": np.asarray(labels), "text_batches": text_batches,
            "batch_index": batch_index, "image": image}


@torch.no_grad()
def run_condition(model, enc, source, device, use_amp, constants=None):
    """Probabilities (N, C) under one condition. `source` is an index map, or 'mean' / 'grey' (looked up
    in `constants`, each a dict tokens (T, H) / pooled (H,))."""
    probs = []
    for text, idx in zip(enc["text_batches"], enc["batch_index"]):
        image = None
        if enc["image"] is not None:
            if isinstance(source, str):
                c = constants[source]
                image = {"tokens": c["tokens"].unsqueeze(0).expand(len(idx), -1, -1).contiguous(),
                         "pooled": c["pooled"].unsqueeze(0).expand(len(idx), -1).contiguous()}
            else:
                src = torch.as_tensor(source[idx])
                image = {"tokens": enc["image"]["tokens"][src], "pooled": enc["image"]["pooled"][src]}
            image = _to(image, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model.classify(None if text is None else _to(text, device), image)
        probs.append(logits.float().softmax(-1).cpu())
    return torch.cat(probs).numpy()


@torch.no_grad()
def image_feature_mean(image_encoder, pixel_batches, device, use_amp, dtype):
    """Mean image-encoder output (tokens per position, pooled) over an iterable of pixel_values batches.
    Accumulated in float32, returned in `dtype` on CPU."""
    tok_sum = pool_sum = None
    n = 0
    for pixel_values in pixel_batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = image_encoder(pixel_values.to(device))
        t, p = out["tokens"].float().sum(0), out["pooled"].float().sum(0)
        tok_sum = t if tok_sum is None else tok_sum + t
        pool_sum = p if pool_sum is None else pool_sum + p
        n += out["tokens"].shape[0]
    return {"tokens": (tok_sum / n).to(dtype).cpu(), "pooled": (pool_sum / n).to(dtype).cpu(), "n": n}


# ---------------------------------------------------------------- comparison

def compare(labels, p_orig, p_cond, n_classes):
    """Per-sample changes between the original and one condition."""
    labels = np.asarray(labels)
    rows = np.arange(len(labels))
    pred_o, pred_c = p_orig.argmax(1), p_cond.argmax(1)
    return {
        "pred_orig": pred_o, "pred_cond": pred_c,
        "flip": pred_o != pred_c,
        "d_p_gold": p_cond[rows, labels] - p_orig[rows, labels],
        "flip_matrix": np.bincount(pred_o * n_classes + pred_c, minlength=n_classes ** 2).reshape(n_classes, n_classes),
    }
