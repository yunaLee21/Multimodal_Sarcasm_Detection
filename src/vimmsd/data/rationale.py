"""Reasoning của teacher cho CoT distillation (docs/cot_distillation.md mục 3-4).

Luồng dữ liệu, mỗi bước một script:
    make_id_lists.py         -> rationales/ids/*.txt
    generate_rationales.py   -> rationales/raw/<prompt_version>[_label]/<split>.shard*.jsonl  (mỗi dòng một lần sinh)
    eval_teacher.py          -> đánh giá teacher (confusion matrix, F1, tỉ lệ parse lỗi)
    build_rationale_dataset.py group / rationalize / export / kappa / merge -> rationale_dataset.json
    embed_rationales.py      -> rationale_embeddings.npy + rationale_embeddings_ids.json
Dataset đọc lại bằng `load_rationale_dataset` (chỉ mẫu đạt rubric) khi bật data.rationale.use."""
import csv
import json
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
LABELS = ("not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm")
RUBRIC = ("c1_text", "c2_image", "c3_combined", "c4_conclusion")
PASS_REVIEWS = ("pass", "auto")  # auto = nhóm 1 không nằm trong mẫu kiểm tra ngẫu nhiên

# CẦN ĐỐI CHIẾU với hướng dẫn gán nhãn của UIT DSC 2024 trước khi chạy cả tập (docs/cot_distillation_plan.md 1.1).
LABEL_DEFINITIONS = """- not-sarcasm: bài đăng không có ý mỉa mai.
- text-sarcasm: ý mỉa mai nằm ở caption; ảnh không góp phần tạo ra ý mỉa mai.
- image-sarcasm: ý mỉa mai nằm ở ảnh (kể cả chữ trong ảnh); caption không góp phần tạo ra ý mỉa mai.
- multi-sarcasm: ý mỉa mai chỉ hiểu được khi đọc caption cùng với ảnh (cả hai cùng góp phần)."""

PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích mỉa mai (sarcasm) trên mạng xã hội Việt Nam. Cho một bài đăng gồm ảnh (đính kèm) và caption.

Caption: \"\"\"{caption}\"\"\"
Chữ đọc được trong ảnh (OCR, có thể sai hoặc thiếu): \"\"\"{ocr}\"\"\"

Định nghĩa nhãn:
{definitions}
{label_hint}
Phân tích theo đúng 4 bước:
1. Chỉ đọc caption, không nhìn ảnh: caption có mỉa mai không? Dẫn chứng cụ thể từ caption.
2. Chỉ nhìn ảnh (kể cả chữ trong ảnh), không đọc caption: ảnh có mỉa mai không? Mô tả khách quan những gì thấy trong ảnh, không suy diễn.
3. Ghép caption với ảnh: có mâu thuẫn, phóng đại, hay ý mỉa mai mới xuất hiện không?
4. Kết luận đúng một nhãn.

Chỉ trả lời bằng MỘT object JSON, không thêm chữ nào khác:
{{"text_alone": {{"sarcastic": true hoặc false, "reason": "..."}}, "image_alone": {{"sarcastic": true hoặc false, "reason": "..."}}, "combined": {{"reason": "..."}}, "label": "not-sarcasm | text-sarcasm | image-sarcasm | multi-sarcasm"}}"""

LABEL_HINT = """
Nhãn đúng của bài đăng này là: {label}. Hãy giải thích vì sao bài đăng có nhãn này theo đúng 4 bước, và kết luận đúng nhãn đó.
"""

MAX_CAPTION_CHARS = 1500  # caption tin tức dài tới ~850 từ; cắt để prompt không quá dài


def build_prompt(caption, ocr="", label=None):
    """Prompt CoT 4 bước. `label` != None: rationalization (đưa nhãn đúng, yêu cầu giải thích)."""
    caption = (caption or "").strip()
    if len(caption) > MAX_CAPTION_CHARS:
        caption = caption[:MAX_CAPTION_CHARS] + " ..."
    return PROMPT_TEMPLATE.format(caption=caption, ocr=(ocr or "").strip() or "(không có)",
                                  definitions=LABEL_DEFINITIONS,
                                  label_hint=LABEL_HINT.format(label=label) if label else "")


_LABEL_ALIASES = {
    "not-sarcasm": "not-sarcasm", "non-sarcasm": "not-sarcasm", "not sarcasm": "not-sarcasm", "none": "not-sarcasm",
    "no-sarcasm": "not-sarcasm", "text-sarcasm": "text-sarcasm", "text sarcasm": "text-sarcasm",
    "image-sarcasm": "image-sarcasm", "image sarcasm": "image-sarcasm", "multi-sarcasm": "multi-sarcasm",
    "multi sarcasm": "multi-sarcasm", "multimodal-sarcasm": "multi-sarcasm",
}
_TRUE = {"true", "yes", "có", "co", "1"}
_FALSE = {"false", "no", "không", "khong", "0"}


def normalize_label(value):
    if not isinstance(value, str):
        return None
    v = value.strip().lower().replace("_", "-")
    return _LABEL_ALIASES.get(v) or _LABEL_ALIASES.get(v.replace("-", " "))


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    return None


def _json_candidates(text):
    text = re.sub(r"```(?:json)?", "", text or "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    raw = text[start:end + 1]
    return [raw, re.sub(r",\s*([}\]])", r"\1", raw)]  # bản thứ hai bỏ dấu phẩy thừa trước } / ]


def parse_teacher_output(text):
    """Đầu ra của teacher -> dict chuẩn hoá, hoặc None nếu không đọc được.
    {"text_sarcastic": bool, "image_sarcastic": bool, "text_reason", "image_reason", "combined_reason", "label"}"""
    for raw in _json_candidates(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        text_alone, image_alone = obj.get("text_alone") or {}, obj.get("image_alone") or {}
        combined = obj.get("combined") or {}
        label = normalize_label(obj.get("label"))
        t, i = _to_bool(text_alone.get("sarcastic")), _to_bool(image_alone.get("sarcastic"))
        if label is None or t is None or i is None:
            return None
        return {"text_sarcastic": t, "image_sarcastic": i,
                "text_reason": str(text_alone.get("reason", "")).strip(),
                "image_reason": str(image_alone.get("reason", "")).strip(),
                "combined_reason": str(combined.get("reason", "")).strip() if isinstance(combined, dict) else "",
                "label": label}
    return None


def rationale_text(parsed):
    """Văn bản reasoning dùng để mã hoá (embed_rationales.py) và cho người chấm đọc. Không kèm nhãn kết luận,
    để embedding là lập luận chứ không phải tên nhãn."""
    return (f"Caption: {parsed['text_reason']} Ảnh: {parsed['image_reason']} "
            f"Khi ghép: {parsed['combined_reason']}").strip()


# ---------------------------------------------------------------- đọc / ghi output thô của teacher

def read_jsonl(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_raw(raw_dir, split=None):
    """{id: [dòng output]} từ mọi file jsonl trong `raw_dir` (gộp các shard), sắp theo sample_idx."""
    paths = sorted(Path(raw_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"không có file .jsonl trong {raw_dir}")
    out = {}
    for row in read_jsonl(paths):
        if split and row.get("split") != split:
            continue
        out.setdefault(str(row["id"]), []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: r.get("sample_idx", 0))
    return out


def predicted_label(rows):
    """Nhãn đa số trong các lần sinh đọc được (hoà thì lấy lần sinh sớm nhất). None nếu không lần nào đọc được."""
    labels = [r["parsed"]["label"] for r in rows if r.get("parsed")]
    if not labels:
        return None
    counts = Counter(labels)
    best = max(counts.values())
    return next(l for l in labels if counts[l] == best)


# ---------------------------------------------------------------- phân nhóm (mục 4.1)

def group_of(n_correct, k):
    if n_correct >= math.ceil(k / 2):
        return 1
    return 2 if n_correct > 0 else 3


def group_samples(raw, labels_by_id):
    """{id: {label, k, n_correct, group, chosen}} với chosen = lần sinh đúng đầu tiên (None ở nhóm 3)."""
    grouped = {}
    for sid, rows in raw.items():
        gold = labels_by_id.get(sid)
        if gold is None:
            continue
        correct = [r for r in rows if r.get("parsed") and r["parsed"]["label"] == gold]
        k = len(rows)
        grouped[sid] = {"label": gold, "k": k, "n_correct": len(correct), "group": group_of(len(correct), k),
                        "chosen": correct[0] if correct else None}
    return grouped


def group_stats(grouped):
    stats = {}
    for g in grouped.values():
        s = stats.setdefault(g["label"], Counter())
        s[f"group{g['group']}"] += 1
        s["total"] += 1
    return {label: dict(c) for label, c in sorted(stats.items())}


# ---------------------------------------------------------------- xuất file chấm và gộp kết quả chấm (mục 4.2-4.4)

REVIEW_COLUMNS = ["id", "label", "group", "rationalized", "caption", "ocr", "image_path", "text_sarcastic",
                  "image_sarcastic", "rationale", *RUBRIC, "edited_rationale", "reviewer", "notes"]


def final_parsed(entry, rationalized_rows):
    """Reasoning dùng cho mẫu: nhóm 1/2 là lần sinh đúng đã chọn, nhóm 3 là lần rationalization đọc được và khớp nhãn."""
    if entry["chosen"] is not None:
        return entry["chosen"]["parsed"], False
    for row in rationalized_rows or []:
        if row.get("parsed") and row["parsed"]["label"] == entry["label"]:
            return row["parsed"], True
    return None, True


def select_for_review(grouped, group1_fraction=0.15, seed=42):
    """Toàn bộ nhóm 2, 3 và `group1_fraction` của nhóm 1 (lấy ngẫu nhiên, cố định theo seed)."""
    rng = random.Random(seed)
    group1 = sorted(sid for sid, g in grouped.items() if g["group"] == 1)
    sampled = set(rng.sample(group1, round(len(group1) * group1_fraction))) if group1 else set()
    return {sid for sid, g in grouped.items() if g["group"] != 1} | sampled


def review_rows(grouped, review_ids, rationalized, records):
    rows = []
    for sid in sorted(review_ids, key=lambda s: (LABELS.index(grouped[s]["label"]), grouped[s]["group"], s)):
        entry, rec = grouped[sid], records.get(sid, {})
        parsed, is_rat = final_parsed(entry, rationalized.get(sid))
        rows.append({
            "id": sid, "label": entry["label"], "group": entry["group"], "rationalized": int(is_rat),
            "caption": rec.get("caption", ""), "ocr": rec.get("ocr", ""), "image_path": rec.get("image_path", ""),
            "text_sarcastic": "" if parsed is None else int(parsed["text_sarcastic"]),
            "image_sarcastic": "" if parsed is None else int(parsed["image_sarcastic"]),
            "rationale": "" if parsed is None else rationale_text(parsed),
            **{c: "" for c in RUBRIC}, "edited_rationale": "", "reviewer": "", "notes": "",
        })
    return rows


def write_csv(path, rows, columns=REVIEW_COLUMNS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel mở đúng tiếng Việt
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _score(value):
    """Ô rubric: 1 / đạt / x / pass = 1, 0 / không / fail = 0, trống = None (chưa chấm)."""
    v = str(value or "").strip().lower()
    if v in {"1", "đạt", "dat", "x", "pass", "y", "yes", "true"}:
        return 1
    if v in {"0", "không", "khong", "fail", "n", "no", "false"}:
        return 0
    return None


def review_verdict(row):
    scores = [_score(row.get(c)) for c in RUBRIC]
    if any(s is None for s in scores):
        return "pending"
    return "pass" if all(scores) else "fail"


def cohen_kappa(a, b):
    """Cohen's kappa của hai dãy nhãn cùng độ dài. 1 = đồng ý hoàn toàn, 0 = như ngẫu nhiên."""
    if len(a) != len(b) or not a:
        raise ValueError("hai người chấm phải có cùng số mẫu (> 0)")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def rubric_kappa(rows_a, rows_b):
    """κ theo từng tiêu chí trên các id cả hai người đã chấm."""
    by_id = {r["id"]: r for r in rows_b}
    common = [(r, by_id[r["id"]]) for r in rows_a if r["id"] in by_id]
    out = {}
    for c in RUBRIC:
        pairs = [(_score(x.get(c)), _score(y.get(c))) for x, y in common]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        out[c] = {"n": len(pairs), "kappa": cohen_kappa(*zip(*pairs)) if pairs else None}
    return out


def merge_dataset(grouped, rationalized, review_csv_rows, meta):
    """Gộp phân nhóm + rationalization + kết quả chấm thành rationale_dataset.json (mục 4.4).
    Mẫu chưa chấm: nhóm 1 là "auto" (giữ), nhóm 2/3 là "pending" (loader bỏ qua)."""
    reviews = {}
    for row in review_csv_rows:
        reviews[row["id"]] = row  # file sau ghi đè file trước nếu một id được chấm nhiều lần
    samples = {}
    for sid, entry in grouped.items():
        parsed, is_rat = final_parsed(entry, rationalized.get(sid))
        review = reviews.get(sid)
        verdict = review_verdict(review) if review else ("auto" if entry["group"] == 1 else "pending")
        if parsed is None:
            verdict = "fail" if review else "pending"
        text = rationale_text(parsed) if parsed else ""
        t = parsed["text_sarcastic"] if parsed else None
        i = parsed["image_sarcastic"] if parsed else None
        edited = False
        if review:
            if (review.get("edited_rationale") or "").strip():
                text, edited = review["edited_rationale"].strip(), True
            for key in ("text_sarcastic", "image_sarcastic"):  # người chấm có thể sửa nhãn phụ
                v = _score(review.get(key))
                if v is not None:
                    t, i = (bool(v), i) if key == "text_sarcastic" else (t, bool(v))
        samples[sid] = {"label": entry["label"], "text_sarcastic": t, "image_sarcastic": i, "rationale": text,
                        "n_correct": entry["n_correct"], "k": entry["k"], "group": entry["group"],
                        "rationalized": bool(is_rat and entry["chosen"] is None), "review": verdict, "edited": edited}
    return {"meta": meta, "samples": samples}


def load_rationale_dataset(path, exclude_groups=()):
    """{id: entry} của các mẫu đạt rubric (review pass / auto), bỏ các nhóm trong `exclude_groups`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    keep = {}
    for sid, s in data["samples"].items():
        if s.get("review") not in PASS_REVIEWS or s.get("group") in set(exclude_groups):
            continue
        if s.get("text_sarcastic") is None or s.get("image_sarcastic") is None:
            continue
        keep[str(sid)] = s
    return keep, data.get("meta", {})


# ---------------------------------------------------------------- danh sách id (make_id_lists.py)

def make_id_lists(labels_by_id, seed=42, pilot_dev=40, pilot_eval=30, not_batch=2000):
    """Tập thử teacher (pilot_dev, pilot_eval: rời nhau, cân bằng theo lớp) và 4 lô sinh reasoning theo thứ tự ưu tiên:
    batch1 = toàn bộ text + image, batch2 = toàn bộ multi, batch3 = `not_batch` mẫu not ngẫu nhiên, batch4 = not còn lại."""
    rng = random.Random(seed)
    by_label = {}
    for sid, label in sorted(labels_by_id.items()):
        by_label.setdefault(label, []).append(sid)
    for ids in by_label.values():
        rng.shuffle(ids)
    lists = {"pilot_dev": [], "pilot_eval": []}
    for label in LABELS:
        ids = by_label.get(label, [])
        n_dev = min(pilot_dev, len(ids))
        lists["pilot_dev"] += ids[:n_dev]
        lists["pilot_eval"] += ids[n_dev:n_dev + min(pilot_eval, len(ids) - n_dev)]
    nots = by_label.get("not-sarcasm", [])
    lists["batch1"] = by_label.get("text-sarcasm", []) + by_label.get("image-sarcasm", [])
    lists["batch2"] = list(by_label.get("multi-sarcasm", []))
    lists["batch3"] = nots[:not_batch]
    lists["batch4"] = nots[not_batch:]
    return {name: sorted(ids) for name, ids in lists.items()}


def read_ids(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_ids(path, ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def split_records(cfg, split, with_ocr=True):
    """{id: record} của một split ("train", "public_test", "private_test"), nhãn dạng tên (None nếu không có),
    kèm OCR từ image_preprocessed.json nếu có (không có thì OCR rỗng, cảnh báo)."""
    from vimmsd.data.dataset import IGNORE_LABEL, load_all_records
    from vimmsd.utils.config import _to_config

    cfg = _to_config(json.loads(json.dumps(cfg)))  # bản sao, không đổi cfg của người gọi
    cfg.data.image_text.use_ocr, cfg.data.image_text.use_description = bool(with_ocr), False
    try:
        records = load_all_records(cfg)
    except (FileNotFoundError, ValueError) as e:
        if not with_ocr:
            raise
        logger.warning("không đọc được OCR (%s), teacher chỉ thấy caption và ảnh", e)
        return split_records(cfg, split, with_ocr=False)
    if split not in records:
        raise KeyError(f"không có split {split!r}, có: {sorted(records)}")
    labels = list(cfg.data.labels)
    return {r["id"]: {**r, "label_name": labels[r["label"]] if r["label"] != IGNORE_LABEL else None}
            for r in records[split]}


# ---------------------------------------------------------------- teacher (Qwen2.5-VL qua transformers)

class TeacherVLM:
    """Sinh reasoning bằng VLM của transformers (mặc định Qwen/Qwen2.5-VL-7B-Instruct).
    k lần sinh mỗi mẫu bằng num_return_sequences (cần temperature > 0). device_map="auto" để chia model lớn
    (vd bản 32B) ra nhiều GPU; khi đó chạy 1 tiến trình thay vì mỗi GPU một shard.
    load_in_4bit=True: lượng tử hoá nf4 (bitsandbytes) cho GPU ít VRAM, bản 7B còn khoảng 5.4 GB."""

    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=640 * 28 * 28, device_map="cuda",
                 load_in_4bit=False):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        from vimmsd.data.image_text import _dtype_kwargs, vlm_dtype

        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=128 * 28 * 28, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"  # sinh theo batch cần pad bên trái
        if load_in_4bit:
            dtype_kwargs = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=vlm_dtype(),
                bnb_4bit_use_double_quant=True)}
        else:
            dtype_kwargs = _dtype_kwargs(vlm_dtype())
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, device_map=device_map,
                                                                 **dtype_kwargs).eval()

    def generate(self, images, prompts, k=1, temperature=0.7, top_p=0.9, max_new_tokens=512):
        """images: list[PIL.Image], prompts: list[str] -> list[list[str]] (k chuỗi mỗi mẫu)."""
        import torch

        texts = [self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
            add_generation_prompt=True) for p in prompts]
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.model.device)
        sample = temperature > 0
        gen = dict(max_new_tokens=max_new_tokens, do_sample=sample, num_return_sequences=k if sample else 1)
        if sample:
            gen.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen)
        decoded = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        per = k if sample else 1
        return [[d.strip() for d in decoded[j * per:(j + 1) * per]] for j in range(len(prompts))]
