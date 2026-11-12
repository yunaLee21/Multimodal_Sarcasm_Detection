"""Augmentation caption lớp hiếm có duyệt tay (docs/empirical_analysis_plan.md mục 8.2), scripts/augment_minority.py.

Luồng: generate (back-translation, tối đa 2 ứng viên mỗi mẫu gốc) -> lọc tự động (rỗng, không đổi, trùng caption train)
-> sheet cho từng người duyệt (thấy nhãn gốc và ảnh, không thấy kết quả model) -> import: chỉ nhận ứng viên cả hai người
duyệt chấp nhận, mỗi mẫu gốc giữ tối đa 1 -> augmentation_manifest.jsonl. Biến thể kế thừa nhóm và fold của mẫu gốc
và chỉ được thêm vào phần train của fold (dataset.apply_augmentation)."""
from collections import Counter

from vimmsd.data.denoise import id_sort_key, normalize_caption

ACCEPT, REJECT = "accept", "reject"
VERDICTS = {"accept": ACCEPT, "a": ACCEPT, "y": ACCEPT, "yes": ACCEPT, "1": ACCEPT,
            "reject": REJECT, "r": REJECT, "n": REJECT, "no": REJECT, "0": REJECT}
# tiêu chí trong sheet (mục 8.2); mọi tiêu chí phải "y" thì verdict mới được là accept
CRITERIA = ["meaning_preserved", "sarcastic_intent_preserved", "modality_label_still_valid", "no_invented_content"]


def select_parents(train, labels):
    """train: {id: {"caption", "label", ...}}. Mẫu gốc = mẫu train thuộc `labels`, sắp theo id."""
    return sorted((i for i, r in train.items() if r["label"] in set(labels)), key=id_sort_key)


def auto_screen(parent_caption, candidate, train_norm_captions, seen):
    """Lý do loại tự động, hoặc None nếu chuyển cho người duyệt. seen: caption chuẩn hoá đã giữ của cùng mẫu gốc."""
    norm = normalize_caption(candidate)
    if not norm:
        return "empty"
    if norm == normalize_caption(parent_caption):
        return "unchanged"
    if norm in train_norm_captions:
        return "duplicate_of_train_caption"
    if norm in seen:
        return "duplicate_candidate"
    return None


def parse_verdict(value):
    return VERDICTS.get(str(value or "").strip().lower())


def merge_reviews(candidate_ids, reviews):
    """reviews: {reviewer: {candidate_id: row}} với row có "verdict" và các tiêu chí CRITERIA.
    Ứng viên được nhận khi MỌI người duyệt đều accept và đánh "y" mọi tiêu chí; thiếu đánh giá hoặc bất đồng = loại.
    Trả về {candidate_id: {"accepted": bool, "verdicts": {...}, "reasons": [...]}}."""
    out = {}
    for cid in candidate_ids:
        verdicts, reasons = {}, []
        for who, rows in reviews.items():
            row = rows.get(cid)
            v = parse_verdict(row.get("verdict")) if row else None
            verdicts[who] = v
            if v is None:
                reasons.append(f"{who}: no verdict")
                continue
            if v == ACCEPT:
                failed = [c for c in CRITERIA if str(row.get(c, "")).strip().lower() not in ("y", "yes", "1")]
                if failed:
                    verdicts[who] = REJECT
                    reasons.append(f"{who}: accept but criteria not all y ({', '.join(failed)})")
            elif row.get("reason"):
                reasons.append(f"{who}: {row['reason']}")
        accepted = bool(verdicts) and all(v == ACCEPT for v in verdicts.values())
        if not accepted and len({v for v in verdicts.values() if v}) > 1:
            reasons.append("reviewers disagree")
        out[cid] = {"accepted": accepted, "verdicts": verdicts, "reasons": reasons}
    return out


def choose_variants(candidates, merged):
    """candidates: list dict (parent_id, candidate_id, rank). Mỗi mẫu gốc giữ tối đa 1 ứng viên được nhận
    (rank nhỏ nhất). Trả về {parent_id: candidate}."""
    chosen = {}
    for c in sorted(candidates, key=lambda c: (id_sort_key(c["parent_id"]), c["rank"])):
        if merged.get(c["candidate_id"], {}).get("accepted") and c["parent_id"] not in chosen:
            chosen[c["parent_id"]] = c
    return chosen


def cohen_kappa(a, b):
    """Cohen's kappa giữa hai danh sách nhãn cùng độ dài (bỏ cặp có None). None nếu không đủ dữ liệu."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def variant_id(parent_id, rank):
    return f"{parent_id}__bt{rank}"
