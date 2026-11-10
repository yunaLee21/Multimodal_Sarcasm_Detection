"""Lọc nhiễu tập train ViMMSD: trùng lặp chính xác, gần trùng, nghi gán nhãn sai (scripts/denoise_data.py).

Chỉ đụng tới train; public/private test giữ nguyên. Nhãn phụ thuộc quan hệ ảnh–caption, nên chỉ coi là trùng khi
**cả ảnh và caption** đều giống; cùng caption khác ảnh (hoặc ngược lại) chỉ dùng để gom nhóm khi chia fold.
Lớp hiếm (image-sarcasm, text-sarcasm) không bao giờ bị bỏ tự động, trừ bản sao chính xác cùng nhãn.

Các hàm thuần (chuẩn hoá caption, gom nhóm, luật quyết định) không cần GPU; phần nhúng ảnh/caption nằm cuối file."""
import hashlib
import html
import re
import unicodedata
from collections import Counter

import numpy as np

DEFAULTS = {
    "seed": 42,
    "minority_labels": ["image-sarcasm", "text-sarcasm"],
    "majority_fraction": 2 / 3,         # nhóm khác nhãn: nhãn chiếm >= tỉ lệ này thì giữ 1 mẫu mang nhãn đó
    "phash_max_distance": 6,            # Hamming trên pHash 64 bit
    "image_cosine": 0.95,               # CLIP image embedding
    "caption_cosine": 0.95,             # multilingual-e5 trên caption gốc
    "near_image_rule": "phash_and_clip",  # gần trùng ảnh (để xử lý) cần cả hai; gom nhóm chia fold chỉ cần một
    "image_model": "openai/clip-vit-base-patch32",
    "text_model": "intfloat/multilingual-e5-base",
    "text_prefix": "query: ",
    "text_max_length": 512,
    "cv_folds": 5,
    "lr_C": 1.0,
    "lr_max_iter": 3000,
    "min_oof_macro_f1": 0.40,           # thấp hơn thì xác suất quá yếu: không áp luật drop_mislabel
    "knn_k": 10,
    "knn_agree_max": 0.2,
    "droppable_labels": ["not-sarcasm", "multi-sarcasm"],
    "max_drop_fraction": 0.05,          # trần drop_mislabel, tính trên số mẫu train gốc
    "review_size": 100,
}

ACTIONS = ["keep", "drop_duplicate", "relabel_majority", "drop_conflict", "flag_conflict", "drop_leak_test",
           "drop_mislabel", "flag_mislabel"]
DROP_ACTIONS = {"drop_duplicate", "drop_conflict", "drop_leak_test", "drop_mislabel"}


def normalize_caption(text):
    """html.unescape, Unicode NFC, gộp khoảng trắng, strip, lowercase."""
    text = unicodedata.normalize("NFC", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip().lower()


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def id_sort_key(sid):
    """id số so theo giá trị (2 < 10), id khác so theo chuỗi, đặt sau."""
    s = str(sid)
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # nén đường đi
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # gốc là id nhỏ nhất để group_id ổn định giữa các lần chạy
            if id_sort_key(rb) < id_sort_key(ra):
                ra, rb = rb, ra
            self.parent[rb] = ra

    def groups(self):
        """{id: group_id}, group_id = 'g<id nhỏ nhất trong nhóm>'."""
        return {x: f"g{self.find(x)}" for x in self.parent}


def group_by_key(keys):
    """keys: {id: khoá}. Trả về list nhóm (>= 2 id, sắp theo id) có cùng khoá, nhóm sắp theo id đầu."""
    buckets = {}
    for sid, key in keys.items():
        buckets.setdefault(key, []).append(sid)
    groups = [sorted(ids, key=id_sort_key) for ids in buckets.values() if len(ids) > 1]
    return sorted(groups, key=lambda g: id_sort_key(g[0]))


def components_from_pairs(ids, pairs):
    """Thành phần liên thông (>= 2 id) từ danh sách cạnh (a, b)."""
    uf = UnionFind(ids)
    for a, b in pairs:
        uf.union(a, b)
    comps = {}
    for sid in ids:
        comps.setdefault(uf.find(sid), []).append(sid)
    groups = [sorted(c, key=id_sort_key) for c in comps.values() if len(c) > 1]
    return sorted(groups, key=lambda g: id_sort_key(g[0]))


def resolve_group(ids, labels, minority=(), majority_fraction=2 / 3, kind="exact"):
    """Luật xử lý một nhóm trùng (cùng ảnh và cùng caption). labels: {id: nhãn}.
    Trả về {id: (action, new_label, reason)}; new_label là nhãn sau khi xử lý (None nếu bị bỏ).

    - Cùng nhãn: giữ id nhỏ nhất, còn lại drop_duplicate (áp dụng cả lớp hiếm).
    - Khác nhãn, có nhãn chiếm >= majority_fraction: giữ id nhỏ nhất mang nhãn đó (relabel_majority, nhãn nhóm),
      các mẫu còn lại drop_duplicate.
    - Khác nhãn, không có đa số: drop_conflict cả nhóm.
    Lớp hiếm không bao giờ bị bỏ vì xung đột: mẫu lớp hiếm khác nhãn chốt được giữ lại với flag_conflict
    (nhiều mẫu cùng nhãn hiếm thì giữ một, còn lại là bản sao chính xác cùng nhãn nên drop_duplicate)."""
    ids = sorted(ids, key=id_sort_key)
    minority = set(minority)
    counts = Counter(labels[i] for i in ids)
    out = {}
    if len(counts) == 1:
        keep = ids[0]
        for i in ids:
            out[i] = ("keep", labels[i], "") if i == keep else (
                "drop_duplicate", None, f"{kind} duplicate of {keep}, same label")
        return out

    detail = ", ".join(f"{lab} {n}" for lab, n in counts.most_common())
    top, n_top = counts.most_common(1)[0]
    majority = top if n_top / len(ids) >= majority_fraction else None
    keep = next(i for i in ids if labels[i] == majority) if majority else None
    kept_minority = {}  # nhãn hiếm -> id được giữ
    for i in ids:
        lab = labels[i]
        if i == keep:
            out[i] = ("relabel_majority", majority, f"{kind} group with labels [{detail}], kept as majority {majority}")
        elif lab in minority and lab != majority:
            if lab in kept_minority:
                out[i] = ("drop_duplicate", None, f"{kind} duplicate of {kept_minority[lab]}, same minority label")
            else:
                kept_minority[lab] = i
                out[i] = ("flag_conflict", lab, f"{kind} group with labels [{detail}], minority label kept for review")
        elif majority:
            out[i] = ("drop_duplicate", None, f"{kind} duplicate of {keep}, group relabelled to majority {majority}")
        else:
            out[i] = ("drop_conflict", None, f"{kind} group with labels [{detail}], no {majority_fraction:.2f} majority")
    return out


def decide_mislabel(ids, labels, cl_issue, knn_agree, self_confidence, droppable, knn_agree_max, max_drop):
    """Luật bước 3. Trả về ({id: action}, số mẫu bị cắt do trần).
    drop_mislabel: cleanlab gắn cờ, knn_agree <= knn_agree_max và nhãn thuộc droppable. Gắn cờ còn lại: flag_mislabel.
    Quá max_drop mẫu thì chỉ giữ drop cho max_drop mẫu self_confidence thấp nhất, phần còn lại thành flag_mislabel."""
    actions, candidates = {}, []
    for i, sid in enumerate(ids):
        if not cl_issue[i]:
            continue
        if knn_agree[i] <= knn_agree_max and labels[i] in droppable:
            candidates.append(i)
        actions[sid] = "flag_mislabel"
    candidates.sort(key=lambda i: (self_confidence[i], id_sort_key(ids[i])))
    for i in candidates[:max_drop]:
        actions[ids[i]] = "drop_mislabel"
    return actions, max(0, len(candidates) - max_drop)


def stratified_sample(ids, labels, n, seed):
    """Lấy n id ngẫu nhiên, chia theo nhãn tỉ lệ với cỡ lớp (largest remainder), mỗi lớp có mặt ít nhất 1 mẫu."""
    rng = np.random.default_rng(seed)
    ids = sorted(ids, key=id_sort_key)
    if len(ids) <= n:
        return ids
    by_label = {}
    for sid in ids:
        by_label.setdefault(labels[sid], []).append(sid)
    names = sorted(by_label)
    quota = {k: max(1, int(n * len(by_label[k]) / len(ids))) for k in names}
    rest = sorted(names, key=lambda k: -(n * len(by_label[k]) / len(ids) % 1))
    while sum(quota.values()) < n:
        k = next(k for k in rest if quota[k] < len(by_label[k]))
        quota[k] += 1
        rest.remove(k)
        rest.append(k)
    while sum(quota.values()) > n:
        k = max(names, key=lambda k: quota[k])
        quota[k] -= 1
    picked = []
    for k in names:
        members = by_label[k]
        take = min(quota[k], len(members))
        picked += [members[j] for j in rng.choice(len(members), size=take, replace=False)]
    return sorted(picked, key=id_sort_key)


def hamming_pairs(hashes, max_distance, chunk=2048):
    """Cặp (i, j), i < j, có khoảng cách Hamming <= max_distance. hashes: mảng uint64."""
    h = np.asarray(hashes, dtype=np.uint64)
    out = []
    for start in range(0, len(h), chunk):
        block = h[start:start + chunk, None] ^ h[None, :]
        dist = np.bitwise_count(block)
        rows, cols = np.nonzero(dist <= max_distance)
        rows = rows + start
        keep = cols > rows
        out.append(np.stack([rows[keep], cols[keep]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)


def cosine_pairs(emb, threshold, chunk=2048):
    """Cặp (i, j), i < j, có cosine >= threshold. emb đã chuẩn hoá L2."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
    if torch.cuda.is_available():
        x = x.cuda()
    out = []
    for start in range(0, len(x), chunk):
        sims = x[start:start + chunk] @ x.T
        rows, cols = torch.nonzero(sims >= threshold, as_tuple=True)
        rows = rows + start
        keep = cols > rows
        out.append(torch.stack([rows[keep], cols[keep]], 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)


def knn_agreement(features, labels, groups, k, chunk=1024):
    """Tỉ lệ trong k láng giềng cosine gần nhất có cùng nhãn, bỏ chính mẫu đó và các mẫu cùng group.
    features đã chuẩn hoá L2; labels, groups: mảng số nguyên cùng độ dài."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    y = torch.as_tensor(np.asarray(labels))
    g = torch.as_tensor(np.asarray(groups))
    if torch.cuda.is_available():
        x, y, g = x.cuda(), y.cuda(), g.cuda()
    out = []
    for start in range(0, len(x), chunk):
        sims = x[start:start + chunk] @ x.T
        same_group = g[start:start + chunk, None] == g[None, :]  # gồm cả chính mẫu đó
        sims = sims.masked_fill(same_group, float("-inf"))
        kk = min(k, len(x))
        values, idx = sims.topk(kk, dim=-1)
        valid = torch.isfinite(values)
        agree = ((y[idx] == y[start:start + chunk, None]) & valid).sum(1).float()
        out.append((agree / valid.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- nhúng ảnh / caption (cần model)

def phash_int(path):
    import imagehash

    from vimmsd.data.image_io import open_image_rgb

    return int(str(imagehash.phash(open_image_rgb(path))), 16)


def clip_image_embeddings(paths, model_name, device, batch_size=64):
    """Embedding ảnh CLIP (chuẩn hoá L2). Ảnh được pad vuông bằng màu mean như lúc train (data.image_resize=pad)."""
    import torch
    from tqdm import tqdm
    from transformers import CLIPImageProcessor, CLIPModel

    from vimmsd.data.image_io import PadToSquare, encoder_mean_color, open_image_rgb

    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    pad = PadToSquare(encoder_mean_color(model_name))
    out = []
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch_size), desc="clip", leave=False):
            images = [pad(open_image_rgb(p)) for p in paths[start:start + batch_size]]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            feats = model.get_image_features(pixel_values=pixels)
            if not torch.is_tensor(feats):  # transformers mới trả về output object
                feats = feats.pooler_output if getattr(feats, "pooler_output", None) is not None else feats[0]
            out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def text_embeddings(texts, model_name, device, prefix="query: ", max_length=512, batch_size=64):
    """Embedding caption bằng sentence encoder (mean pooling theo attention mask, chuẩn hoá L2), như cách dùng e5."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="text", leave=False):
            batch = [prefix + (t or "") for t in texts[start:start + batch_size]]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)
