"""Fold nội bộ có nhóm cho nghiên cứu thực nghiệm (docs/empirical_analysis_plan.md mục 4.2), scripts/build_empirical_splits.py.

Nhóm = các mẫu train nối với nhau bởi caption chuẩn hoá trùng khớp HOẶC ảnh trùng md5, bắc cầu (union-find).
Fold = StratifiedGroupKFold theo nhãn trên nhóm, seed cố định. Validator kiểm tra: mỗi id đúng một fold, không nhóm nào
nằm ở hai fold, mọi lớp có mặt ở mọi fold (đủ ngưỡng tối thiểu)."""
import hashlib
from collections import Counter

from vimmsd.data.denoise import UnionFind, group_by_key, id_sort_key, normalize_caption


def build_groups(captions, md5s):
    """captions, md5s: {id: caption gốc}, {id: md5 ảnh}. Trả về {id: group_id} ('g<id nhỏ nhất>')."""
    ids = sorted(captions, key=id_sort_key)
    uf = UnionFind(ids)
    for keys in ({i: normalize_caption(captions[i]) for i in ids}, {i: md5s[i] for i in ids}):
        for g in group_by_key(keys):
            for a, b in zip(g, g[1:]):
                uf.union(a, b)
    return uf.groups()


def make_group_folds(labels, groups, k=3, seed=42):
    """labels, groups: {id: nhãn}, {id: group_id}. Trả về {id: fold} bằng StratifiedGroupKFold(shuffle, seed)."""
    from sklearn.model_selection import StratifiedGroupKFold

    ids = sorted(labels, key=id_sort_key)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = {}
    for f, (_, test_idx) in enumerate(cv.split(ids, [labels[i] for i in ids], [groups[i] for i in ids])):
        for j in test_idx:
            folds[ids[j]] = f
    return folds


def validate_folds(labels, groups, folds, k, min_support=None):
    """Danh sách lỗi (rỗng = hợp lệ). min_support: {nhãn: số mẫu tối thiểu mỗi fold}, mặc định 1 cho mọi lớp có mặt."""
    problems = []
    missing = sorted(set(labels) - set(folds), key=id_sort_key)
    extra = sorted(set(folds) - set(labels), key=id_sort_key)
    if missing:
        problems.append(f"{len(missing)} id không có fold, vd {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} id lạ trong fold, vd {extra[:3]}")
    bad = sorted({f for f in folds.values() if f not in range(k)})
    if bad:
        problems.append(f"fold ngoài khoảng 0..{k - 1}: {bad}")
    span = {}
    for i, f in folds.items():
        span.setdefault(groups[i], set()).add(f)
    split_groups = [g for g, fs in span.items() if len(fs) > 1]
    if split_groups:
        problems.append(f"{len(split_groups)} nhóm nằm ở nhiều fold, vd {split_groups[:3]}")
    min_support = min_support or {lab: 1 for lab in set(labels.values())}
    support = fold_support(labels, folds, k)
    for f in range(k):
        for lab, need in min_support.items():
            if support[f].get(lab, 0) < need:
                problems.append(f"fold {f}: {lab} có {support[f].get(lab, 0)} mẫu, cần >= {need}")
    return problems


def fold_support(labels, folds, k):
    out = [Counter() for _ in range(k)]
    for i, f in folds.items():
        out[f][labels[i]] += 1
    return out


def overlap_stats(splits):
    """splits: {tên: {"captions": {id: caption}, "md5s": {id: md5}}}. Với mỗi cặp split: số mẫu của split sau có caption
    chuẩn hoá / ảnh md5 / cả hai trùng với split trước."""
    names = list(splits)
    out = {}
    for a_i, a in enumerate(names):
        cap_a = {normalize_caption(c) for c in splits[a]["captions"].values()}
        md5_a = set(splits[a]["md5s"].values())
        both_a = {(normalize_caption(splits[a]["captions"][i]), splits[a]["md5s"][i]) for i in splits[a]["captions"]}
        for b in names[a_i + 1:]:
            cb, mb = splits[b]["captions"], splits[b]["md5s"]
            out[f"{a}↔{b}"] = {
                "n": len(cb),
                "caption": sum(normalize_caption(c) in cap_a for c in cb.values()),
                "image_md5": sum(m in md5_a for m in mb.values()),
                "caption_and_image": sum((normalize_caption(cb[i]), mb[i]) in both_a for i in cb),
                "caption_or_image": sum(normalize_caption(cb[i]) in cap_a or mb[i] in md5_a for i in cb),
            }
    return out


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def digest(mapping):
    """sha256 ổn định của một dict {khoá: giá trị chuỗi} (vd md5 của mọi ảnh), để ghi vào manifest."""
    h = hashlib.sha256()
    for key in sorted(mapping, key=id_sort_key):
        h.update(f"{key}\t{mapping[key]}\n".encode())
    return h.hexdigest()
