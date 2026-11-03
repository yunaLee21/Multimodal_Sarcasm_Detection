"""Giải thích do người viết, thay cho teacher LLM (reports/experiment_report.md mục 7).

make_explanation_sheet.py chọn mẫu train và xuất sheet Excel; người viết điền 3 cờ (caption riêng, ảnh riêng, khi
ghép) và 3 lý do tiếng Việt. Các cờ suy ra nhãn theo bảng quyết định của docs/labeling_guidelines.md mục 3."""
import random
from collections import defaultdict

from vimmsd.data.rationale import DATASET_EXAMPLE_IDS, LABELS, load_raw

# số mẫu mỗi lớp: ưu tiên image-sarcasm (lớp G1 cần), text-sarcasm chỉ có 77 mẫu train
DEFAULT_QUOTA = {"image-sarcasm": 150, "multi-sarcasm": 130, "not-sarcasm": 90, "text-sarcasm": 30}
REASON_KEYS = ("text_reason", "image_reason", "combined_reason")


def label_from_flags(caption, image, combined):
    """Bảng quyết định mục 3: cả hai có, hoặc kết hợp tạo mỉa mai -> multi; chỉ một bên -> bên đó; không có -> not."""
    if (caption and image) or combined:
        return "multi-sarcasm"
    if caption:
        return "text-sarcasm"
    if image:
        return "image-sarcasm"
    return "not-sarcasm"


def multi_case(caption, image, combined):
    """Trường hợp 1–4 của multi-sarcasm (docs/experiment_design.md mục 1): 1 cả hai, 2 chỉ caption, 3 chỉ ảnh,
    4 không bên nào, ý mỉa mai chỉ có khi ghép. None nếu không phải multi."""
    if label_from_flags(caption, image, combined) != "multi-sarcasm":
        return None
    return 1 if caption and image else 2 if caption else 3 if image else 4


def sample_ids(labels_by_id, quota=None, exclude=(), seed=42):
    """Chọn ngẫu nhiên theo lớp, bỏ `exclude` và các mẫu ví dụ trong prompt. Lớp thiếu mẫu thì lấy hết."""
    quota = quota or DEFAULT_QUOTA
    excluded = set(exclude) | set(DATASET_EXAMPLE_IDS)
    by_label = defaultdict(list)
    for sid, label in sorted(labels_by_id.items()):
        if sid not in excluded:
            by_label[label].append(sid)
    rng = random.Random(seed)
    chosen = {label: rng.sample(by_label[label], min(n, len(by_label[label]))) for label, n in quota.items()}
    return chosen


def interleave_batches(chosen, n_batches, seed=42):
    """Chia mẫu thành các đợt cùng tỉ lệ lớp, xáo trong từng đợt. Trả về list (đợt, id)."""
    rng = random.Random(seed)
    batches = [[] for _ in range(n_batches)]
    for label in sorted(chosen):
        ids = list(chosen[label])
        rng.shuffle(ids)
        for k, sid in enumerate(ids):
            batches[k % n_batches].append(sid)
    out = []
    for b, ids in enumerate(batches, start=1):
        rng.shuffle(ids)
        out += [(b, sid) for sid in ids]
    return out


def collect_drafts(raw_dirs, split="train"):
    """Lý do của teacher LLM đã sinh trước đây, chỉ khi teacher tự đoán đúng nhãn (không cho nhãn). raw_dirs theo thứ
    tự ưu tiên: thư mục đầu có nháp cho id thì dùng thư mục đó. Trả về {id: (tên thư mục, parsed)}."""
    drafts = {}
    for raw_dir in raw_dirs:
        for sid, rows in load_raw(raw_dir, split).items():
            if sid in drafts:
                continue
            for row in rows:
                parsed = row.get("parsed")
                if not row.get("with_label") and parsed and row.get("label") in LABELS and parsed.get("label") == row["label"]:
                    drafts[sid] = (raw_dir.name, parsed)
                    break
    return drafts
