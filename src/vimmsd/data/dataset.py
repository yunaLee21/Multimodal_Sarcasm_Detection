import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision import transforms

from vimmsd.data.augmentation import build_image_augment
from vimmsd.data.image_io import PadToSquare, encoder_mean_color, open_image_rgb
from vimmsd.data.image_text import compose_image_text, image_key
from vimmsd.data.preprocessing import CAPTION_FILTERS, TextPreprocessor, clean_ocr

logger = logging.getLogger(__name__)

IGNORE_LABEL = -1  # mẫu không có nhãn


def load_label_file(path):
    """Đọc file nhãn tham chiếu của public/private test: {"results": {id: nhãn}, "phase": ...}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    results = raw.get("results") if isinstance(raw, dict) else None
    if not isinstance(results, dict):
        raise ValueError(f"{path} cần object results (id -> nhãn)")
    return {str(k): v for k, v in results.items()}


def load_records(json_path, image_dir, label2id, image_texts=None, labels_path=None):
    """Đọc file annotation ViMMSD: {id: {"image", "caption", "label"}} (hoặc list các dict).
    Trả về list dict thống nhất: id, image_path, caption, label (int, -1 nếu không có nhãn).
    image_texts: {"ocr": cache, "description": cache} tạo bởi scripts/extract_image_text.py.
    labels_path: file nhãn {"results": {id: nhãn}}, điền cho mẫu có label null."""
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.items() if isinstance(raw, dict) else ((str(i), r) for i, r in enumerate(raw))
    ref_labels = load_label_file(labels_path) if labels_path else {}

    image_texts = image_texts or {}
    records, missing = [], Counter()
    for sid, r in items:
        label = r.get("label")
        if label is None:
            label = ref_labels.get(str(sid))
        rec = {
            "id": str(sid),
            "image_path": str(Path(image_dir) / r["image"]),
            "caption": r.get("caption", ""),
            "label": label2id[label] if label is not None else IGNORE_LABEL,
        }
        key = image_key(image_dir, r["image"])
        for field, cache in image_texts.items():
            missing[field] += key not in cache
            rec[field] = cache.get(key, "")
        records.append(rec)

    # ảnh không có trong cache khác với ảnh có text rỗng: thiếu nghĩa là chưa chạy (hoặc chạy lỗi) bước trích xuất
    for field, n in missing.items():
        if n and n == len(records):
            raise ValueError(f"cache '{field}' không có ảnh nào của {json_path} (thư mục ảnh {Path(image_dir).name}): "
                             "chạy scripts/extract_image_text.py cho split này hoặc kiểm tra data.image_text.dir")
        if n:
            logger.warning("cache '%s' thiếu %d/%d ảnh của %s, các mẫu này không có text từ ảnh",
                           field, n, len(records), json_path)
    return records


def split_records(records, val_ratio, test_ratio, seed):
    """Chia stratified train/val/test từ tập train có nhãn (data.split=random)."""
    labels = [r["label"] for r in records]
    holdout = val_ratio + test_ratio
    train, rest, _, rest_labels = train_test_split(
        records, labels, test_size=holdout, stratify=labels, random_state=seed
    )
    if test_ratio == 0:
        return train, rest, []
    val, test = train_test_split(
        rest, test_size=test_ratio / holdout, stratify=rest_labels, random_state=seed
    )
    return train, val, test


class ViMMSDDataset(Dataset):
    def __init__(self, records, text_preprocessor=None, image_transform=None,
                 load_image=True, cache_dir=None, use_image_text=False, separate_image_text=False,
                 missing_image_color=(0, 0, 0), rationale_dim=None):
        self.records = records
        # CoT distillation: rationale_dim != None thì mỗi item có nhãn phụ của teacher và embedding reasoning
        # (vector 0 + cờ 0 khi mẫu không có), xem attach_rationales
        self.rationale_dim = rationale_dim
        self.image_transform = image_transform
        self.load_image = load_image
        self.missing_image_color = tuple(missing_image_color)
        self.separate_image_text = bool(separate_image_text and use_image_text)

        captions = [r["caption"] for r in records]
        if text_preprocessor is not None:
            captions = text_preprocessor.process_all(captions, cache_dir=cache_dir)
        self.texts = captions
        self.image_texts = None
        self.ocrs = None
        self.descriptions = None
        if use_image_text and self.separate_image_text:
            ocrs = [clean_ocr(r.get("ocr", ""), r["caption"]) for r in records]
            descriptions = [r.get("description", "") or "" for r in records]
            if text_preprocessor is not None:
                ocrs = text_preprocessor.process_all(ocrs, cache_dir=cache_dir)
                descriptions = text_preprocessor.process_all(descriptions, cache_dir=cache_dir)
            self.ocrs = ocrs
            self.descriptions = descriptions
        elif use_image_text:
            image_texts = [compose_image_text(clean_ocr(r.get("ocr", ""), r["caption"]), r.get("description", ""))
                           for r in records]
            if text_preprocessor is not None:
                image_texts = text_preprocessor.process_all(image_texts, cache_dir=cache_dir)
            self.image_texts = image_texts
        self._bad_images = 0

    def __len__(self):
        return len(self.records)

    @property
    def labels(self):
        return [r["label"] for r in self.records]

    def _open_image(self, path):
        """Trả về (ảnh, có lỗi hay không). Ảnh lỗi được thay bằng ảnh một màu `missing_image_color`
        (mean của encoder, sau normalize gần 0) và đánh cờ `img_missing` để đếm được số mẫu bị ảnh hưởng."""
        try:
            return open_image_rgb(path), False
        except (OSError, ValueError) as e:
            self._bad_images += 1
            if self._bad_images <= 5:
                logger.warning("không đọc được ảnh %s (%s), thay bằng ảnh một màu và đặt img_missing=1", path, e)
            return Image.new("RGB", (224, 224), self.missing_image_color), True

    def __getitem__(self, idx):
        rec = self.records[idx]
        item = {"id": rec["id"], "text": self.texts[idx], "label": rec["label"]}
        if self.separate_image_text:
            item["ocr"] = self.ocrs[idx]
            item["description"] = self.descriptions[idx]
        elif self.image_texts is not None:
            item["image_text"] = self.image_texts[idx]
        if self.rationale_dim is not None:
            axis = rec.get("teacher_axis")
            item["axis_teacher"] = axis if axis is not None else (0.0, 0.0)
            item["axis_teacher_present"] = float(axis is not None)
            if self.rationale_dim:
                emb = rec.get("rationale_emb")
                item["rationale_emb"] = emb if emb is not None else np.zeros(self.rationale_dim, dtype=np.float32)
                item["rationale_present"] = float(emb is not None)
        if self.load_image:
            img, missing = self._open_image(rec["image_path"])
            if self.image_transform is not None:
                img = self.image_transform(img)
            item["image"] = img
            item["img_missing"] = int(missing)
        return item


class ViMMSDCollator:
    """Tokenize text + xử lý ảnh theo batch (padding động theo câu dài nhất trong batch).

    Khi có text từ ảnh (2 segment), caption được cắt trước còn tối đa `max_caption_length` token, phần còn
    thiếu chỗ chỉ cắt ở segment 2 (`only_second`). Không dùng `only_second` một mình: nếu riêng caption đã
    dài hơn `max_length`, tokenizer không cắt gì và trả về chuỗi dài hơn giới hạn của PhoBERT."""

    def __init__(self, tokenizer=None, image_processor=None, max_length=128, max_caption_length=None,
                 aux_max_length=96):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.max_caption_length = max_caption_length or max_length
        self.aux_max_length = aux_max_length

    def _clip_caption(self, text):
        # only_second chỉ cắt được khi segment 2 dài hơn số token cần bỏ, tức caption + token đặc biệt < max_length
        limit = min(self.max_caption_length, self.max_length - self.tokenizer.num_special_tokens_to_add(pair=True) - 1)
        tokens = self.tokenizer.tokenize(text)
        return text if len(tokens) <= limit else self.tokenizer.convert_tokens_to_string(tokens[:limit])

    def __call__(self, items):
        batch = {
            "ids": [it["id"] for it in items],
            "labels": torch.tensor([it["label"] for it in items], dtype=torch.long),
        }
        if self.tokenizer is not None:
            texts = [it["text"] for it in items]
            if "ocr" in items[0]:
                # caption một mình. OCR và mô tả VLM tokenize riêng, không ghép vào caption.
                enc = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
                for field in ("ocr", "description"):
                    aux = self.tokenizer([it[field] for it in items], padding=True, truncation=True,
                                         max_length=self.aux_max_length, return_tensors="pt")
                    batch[f"{field}_input_ids"] = aux["input_ids"]
                    batch[f"{field}_attention_mask"] = aux["attention_mask"]
                    batch[f"{field}_present"] = torch.tensor(
                        [float(bool(it[field].strip())) for it in items], dtype=torch.float)
            elif "image_text" in items[0]:
                # 2 segment: <s> caption </s></s> chữ trong ảnh + mô tả ảnh </s>
                enc = self.tokenizer([self._clip_caption(t) for t in texts], [it["image_text"] for it in items],
                                     padding=True, truncation="only_second", max_length=self.max_length,
                                     return_tensors="pt")
            else:
                enc = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
            batch["input_ids"] = enc["input_ids"]
            batch["attention_mask"] = enc["attention_mask"]
        if self.image_processor is not None:
            images = [it["image"] for it in items]
            batch["pixel_values"] = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        if "img_missing" in items[0]:
            batch["img_missing"] = torch.tensor([it["img_missing"] for it in items], dtype=torch.long)
        if "axis_teacher" in items[0]:
            batch["axis_teacher"] = torch.tensor([it["axis_teacher"] for it in items], dtype=torch.float)
            batch["axis_teacher_mask"] = torch.tensor([it["axis_teacher_present"] for it in items])
        if "rationale_emb" in items[0]:
            batch["rationale_emb"] = torch.from_numpy(np.stack([it["rationale_emb"] for it in items]).astype(np.float32))
            batch["rationale_mask"] = torch.tensor([it["rationale_present"] for it in items])
        return batch


def text_cache_dir(cfg):
    """Thư mục chứa file cache caption. None trong config thì dùng paths.cache_dir."""
    text_cfg = cfg.data.get("text") or {}
    return text_cfg.get("cache_dir") or cfg.paths.get("cache_dir")


def stage_preprocessed_cache(cfg, search_root=None):
    """Copy JSON đã preprocess từ dataset Kaggle sang thư mục cache ghi được.

    Dataset `funalee/vimmsd-data-preprocessed`: `text_preprocessed.json` cho caption,
    `image_preprocessed.json` cho OCR và mô tả VLM. `image_preprocessed.csv` là bảng xem
    cùng nội dung đó, không đưa vào model.
    """
    root = Path(search_root) if search_root is not None else Path("/kaggle/input")
    if not root.exists():
        return []
    text_cfg = cfg.data.get("text") or {}
    it_cfg = cfg.data.get("image_text") or {}
    names = []
    if text_cfg.get("cache_file"):
        names.append(text_cfg["cache_file"])
    if it_cfg.get("file") and (it_cfg.get("use_ocr") or it_cfg.get("use_description")):
        names.append(it_cfg["file"])
    if not names:
        return []
    dest_dir = Path(text_cfg.get("cache_dir") or cfg.paths.cache_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in names:
        dest = dest_dir / name
        if dest.exists():
            logger.info("dùng cache có sẵn %s", dest)
            copied.append(dest)
            continue
        hits = sorted(p for p in root.rglob(name) if p.is_file())
        if not hits:
            raise FileNotFoundError(
                f"không thấy {name} trong {root}. Add Input dataset funalee/vimmsd-data-preprocessed."
            )
        shutil.copy(hits[0], dest)
        logger.info("copy %s -> %s", hits[0], dest)
        copied.append(dest)
    return copied


def collect_captions(cfg):
    """Mọi caption trong train / public test / private test. File không có thì bỏ qua."""
    data_dir = Path(cfg.paths.data_dir)
    names = [cfg.data.train_json]
    for key in ("public_test_json", "private_test_json"):
        name = cfg.data.get(key)
        if name:
            names.append(name)
    captions, used = [], []
    for name in names:
        path = data_dir / name
        if not path.exists():
            logger.warning("bỏ qua %s: không thấy file", path)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.values() if isinstance(raw, dict) else raw
        captions.extend(str(item.get("caption", "")) for item in items)
        used.append(name)
    return captions, used


def build_text_cache(cfg):
    """Chuẩn hóa toàn bộ caption và ghi data.text.cache_file vào text_cache_dir(cfg).

    Train đọc lại file này qua TextPreprocessor.process_all, nên không xử lý lại từ đầu.
    """
    pre = TextPreprocessor.from_config(cfg.data.text)
    out_dir = Path(text_cache_dir(cfg))
    captions, used = collect_captions(cfg)
    pre.process_all(captions, cache_dir=out_dir)
    return pre._cache_file(out_dir), len(captions), used


_IMAGE_TEXT_FIELDS = {
    "ocr": ("ocr", "text"),
    "description": ("description", "vlm", "vlm_description"),
}


def _image_field(entry, field):
    """Text của field theo tên đầu tiên có trong entry. None = bước đó chưa chạy cho ảnh này, "" = không có chữ."""
    if not isinstance(entry, dict):
        return None
    values = [entry[name] for name in _IMAGE_TEXT_FIELDS[field] if entry.get(name) is not None]
    return str(next((v for v in values if v), values[0])) if values else None


def _load_combined_image_text(path, it_cfg):
    """Đọc `image_preprocessed.json` của dataset Kaggle: {"meta": {...}, "images": {image_key: {ocr, description}}}."""
    if not path.exists():
        raise FileNotFoundError(
            f"thiếu {path}: attach funalee/vimmsd-data-preprocessed, hoặc chạy scripts/build_image_text_dataset.py "
            "(cuối notebooks/01b); đặt data.image_text.file=null để đọc từng cache"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    images = raw.get("images") if isinstance(raw, dict) else None
    if not isinstance(images, dict):
        raise ValueError(f"{path} cần object images (image_key -> {{ocr, description}})")
    sample = next(iter(images.values()), {})
    if not isinstance(sample, dict):
        raise ValueError(f"{path}: mỗi ảnh phải là object có ocr và description, gặp {type(sample).__name__}")
    texts = {}
    for field, flag in (("ocr", "use_ocr"), ("description", "use_description")):
        if it_cfg.get(flag):
            # bỏ ảnh có giá trị null để load_records cảnh báo thiếu, thay vì coi như ảnh không có chữ
            values = {key: _image_field(entry, field) for key, entry in images.items()}
            texts[field] = {key: v for key, v in values.items() if v is not None}
    meta = raw.get("meta") or {}
    logger.info("đọc %d ảnh từ %s (missing=%s)", len(images), path.name, meta.get("missing"))
    return texts


def load_image_texts(cfg):
    """Đọc OCR / mô tả VLM theo `data.image_text` trong config, trả về {"ocr": {image_key: text}, ...}.
    Ưu tiên file gộp `file` (image_preprocessed.json); không có thì đọc từng cache trung gian. Trả về {} nếu không dùng."""
    it_cfg = cfg.data.get("image_text") or {}
    if not (it_cfg.get("use_ocr") or it_cfg.get("use_description")):
        return {}
    cache_dir = Path(it_cfg.get("dir") or cfg.paths.cache_dir)
    if it_cfg.get("file"):
        return _load_combined_image_text(cache_dir / it_cfg["file"], it_cfg)
    texts = {}
    for field, flag, name in (("ocr", "use_ocr", "ocr_cache"), ("description", "use_description", "description_cache")):
        if it_cfg.get(flag):
            path = cache_dir / it_cfg[name]
            if not path.exists():
                raise FileNotFoundError(f"thiếu {path}: chạy scripts/extract_image_text.py (notebooks/01b) trước")
            texts[field] = json.loads(path.read_text(encoding="utf-8"))
    return texts


def uses_image_text(cfg):
    it_cfg = cfg.data.get("image_text") or {}
    return bool(it_cfg.get("use_ocr") or it_cfg.get("use_description"))


SEPARATE_IMAGE_TEXT_MODALITIES = ("text_aux", "multistream", "multistream_clip")


def separates_image_text(cfg):
    """Model 3.2 và fusion nhiều luồng: OCR và mô tả là feature riêng, không gộp vào caption."""
    return cfg.model.get("modality") in SEPARATE_IMAGE_TEXT_MODALITIES and uses_image_text(cfg)


def _load_test_split(cfg, key, label2id, image_texts):
    """Đọc public_test / private_test kèm file nhãn nếu có. Trả về None nếu không thấy json."""
    data_dir = Path(cfg.paths.data_dir)
    name = cfg.data.get(f"{key}_json")
    if not name or not (data_dir / name).exists():
        return None
    labels_name = cfg.data.get(f"{key}_labels")
    labels_path = data_dir / labels_name if labels_name else None
    if labels_path is not None and not labels_path.exists():
        logger.warning("không thấy %s, %s không có nhãn", labels_path, key)
        labels_path = None
    return load_records(data_dir / name, data_dir / cfg.data[f"{key}_image_dir"], label2id, image_texts,
                        labels_path=labels_path)


def load_all_records(cfg):
    """Trả về dict split -> records.

    data.split=official: train = toàn bộ tập train, val = public test, test = private test (đều có nhãn).
    data.split=random: train/val/test chia stratified từ tập train như trước.
    Luôn kèm public_test / private_test (nếu có file) để xuất dự đoán theo định dạng nộp bài."""
    data_dir = Path(cfg.paths.data_dir)
    label2id = {l: i for i, l in enumerate(cfg.data.labels)}
    image_texts = load_image_texts(cfg)

    labeled = load_records(data_dir / cfg.data.train_json, data_dir / cfg.data.train_image_dir,
                           label2id, image_texts)
    public = _load_test_split(cfg, "public_test", label2id, image_texts)
    private = _load_test_split(cfg, "private_test", label2id, image_texts)

    mode = cfg.data.get("split", "official")
    if mode == "official":
        for key, recs in (("public_test", public), ("private_test", private)):
            if recs is None:
                raise FileNotFoundError(f"data.split=official cần {key}: không thấy "
                                        f"{data_dir / str(cfg.data.get(key + '_json'))}")
            unlabeled = sum(r["label"] == IGNORE_LABEL for r in recs)
            if unlabeled:
                raise ValueError(f"data.split=official nhưng {unlabeled}/{len(recs)} mẫu {key} không có nhãn: "
                                 f"kiểm tra data.{key}_labels, hoặc đặt data.split=random")
        splits = {"train": labeled, "val": public, "test": private}
    elif mode == "random":
        split_seed = cfg.data.get("split_seed")  # cố định cách chia khi chạy nhiều seed train
        train, val, test = split_records(labeled, cfg.data.val_ratio, cfg.data.test_ratio,
                                         cfg.seed if split_seed is None else split_seed)
        splits = {"train": train, "val": val, "test": test}
    else:
        raise ValueError(f"data.split không hợp lệ: {mode!r}, chọn official | random")

    if public is not None:
        splits["public_test"] = public
    if private is not None:
        splits["private_test"] = private
    fold = cfg.data.get("fold") or {}
    if fold.get("manifest"):
        if cfg.train.get("eval_test", True):
            raise ValueError("data.fold.manifest là chế độ phát triển nội bộ: đặt train.eval_test=false "
                             "(docs/empirical_analysis_plan.md mục 4.3)")
        manifest = json.loads(Path(fold["manifest"]).read_text(encoding="utf-8"))
        from vimmsd.data.splits import file_sha256

        splits = apply_fold_manifest(splits, manifest, fold.get("index"), file_sha256(data_dir / cfg.data.train_json))
    elif fold.get("index") is not None:
        held_out = fold_of(splits["train"], fold.get("k", 5), fold.get("seed", 42))
        splits["train"] = [r for r in splits["train"] if held_out[r["id"]] != fold["index"]]
    aug = cfg.data.get("augmentation") or {}
    if aug.get("manifest"):
        variants = [json.loads(line) for line in Path(aug["manifest"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        splits = apply_augmentation(splits, variants, label2id)
    return apply_caption_filter(splits, cfg.data.get("caption_filter"))


def apply_augmentation(splits, variants, label2id):
    """Thêm biến thể đã duyệt (augmentation_manifest.jsonl, scripts/augment_minority.py) vào train: chỉ biến thể có
    mẫu gốc nằm trong phần train hiện tại (sau khi tách fold), nên không biến thể nào rơi vào val/test hay đi kèm
    mẫu gốc đang bị giữ lại. Biến thể chép ảnh, OCR, mô tả của mẫu gốc, chỉ thay caption; parent_id = id mẫu gốc."""
    by_id = {r["id"]: r for r in splits["train"]}
    added = []
    for v in variants:
        parent = by_id.get(str(v["parent_id"]))
        if parent is None:
            continue
        if label2id[v["label"]] != parent["label"]:
            raise ValueError(f"biến thể {v['variant_id']} mang nhãn {v['label']} khác nhãn mẫu gốc {v['parent_id']}")
        added.append({**parent, "id": v["variant_id"], "caption": v["caption"], "parent_id": parent["id"]})
    out = dict(splits)
    out["train"] = splits["train"] + added
    logger.info("augmentation: thêm %d/%d biến thể (mẫu gốc thuộc phần train)", len(added), len(variants))
    return out


def apply_fold_manifest(splits, manifest, index, train_sha256):
    """Fold nội bộ có nhóm (scripts/build_empirical_splits.py): train = các fold khác, val = fold `index` của train.
    Báo lỗi nếu manifest được tạo từ file train khác, thiếu id, hoặc index ngoài khoảng."""
    meta, folds = manifest["meta"], manifest["folds"]
    expected = meta["checksums"]["train"]["json_sha256"]
    if expected != train_sha256:
        raise ValueError(f"manifest fold tạo từ train sha256 {expected[:12]}, file train hiện tại {train_sha256[:12]}")
    if index is None or index not in range(meta["k"]):
        raise ValueError(f"data.fold.index phải trong 0..{meta['k'] - 1}, nhận {index!r}")
    missing = [r["id"] for r in splits["train"] if r["id"] not in folds]
    if missing:
        raise ValueError(f"{len(missing)} mẫu train không có trong manifest fold, vd {missing[:3]}")
    out = dict(splits)
    out["train"] = [r for r in splits["train"] if folds[r["id"]] != index]
    out["val"] = [r for r in splits["train"] if folds[r["id"]] == index]
    logger.info("fold %d/%d từ manifest (seed %s): train %d | val (fold giữ lại) %d", index, meta["k"], meta["seed"],
                len(out["train"]), len(out["val"]))
    return out


def fold_of(records, k=5, seed=42):
    """{id: fold} chia stratified theo nhãn, cố định theo seed (B2: data.fold.index = fold bị giữ lại khi train).
    Chỉ phụ thuộc id và nhãn nên script dự đoán out-of-fold tính lại được đúng cách chia."""
    rng = np.random.default_rng(seed)
    out = {}
    by_label = {}
    for rec in sorted(records, key=lambda r: r["id"]):
        by_label.setdefault(rec["label"], []).append(rec["id"])
    for label in sorted(by_label):
        ids = by_label[label]
        for j, idx in enumerate(rng.permutation(len(ids))):
            out[ids[idx]] = j % k
    return out


def apply_caption_filter(splits, name):
    """data.caption_filter: xoá thẻ nguồn khỏi caption ở mọi split (preprocessing.CAPTION_FILTERS). null = giữ nguyên."""
    if not name:
        return splits
    if name not in CAPTION_FILTERS:
        raise ValueError(f"data.caption_filter không hợp lệ: {name!r}, chọn một trong {sorted(CAPTION_FILTERS)}")
    edit = CAPTION_FILTERS[name]
    seen = {}  # val = public_test cùng một list: giữ nguyên quan hệ này
    return {key: seen.setdefault(id(recs), [{**r, "caption": edit(r["caption"])} for r in recs]) for key, recs in splits.items()}


def build_image_transform(cfg, train, fill):
    """Đường A trước image processor: pad vuông (A2, mặc định) rồi augment nhẹ (A3, chỉ khi train).
    `data.image_resize: crop` bỏ bước pad, để processor center-crop như code cũ (dùng cho ablation)."""
    mode = cfg.data.get("image_resize", "pad")
    if mode not in ("pad", "crop"):
        raise ValueError(f"data.image_resize không hợp lệ: {mode!r}, chọn pad | crop")
    steps = [PadToSquare(fill)] if mode == "pad" else []
    augment = build_image_augment(cfg.data.get("image_augment", False)) if train else None
    if augment is not None:
        steps.append(augment)
    return transforms.Compose(steps) if steps else None


def uses_rationale(cfg):
    return bool((cfg.data.get("rationale") or {}).get("use"))


def rationale_dir(cfg):
    """Thư mục reasoning: data.rationale.dir, mặc định <paths.cache_dir>/rationales."""
    r = cfg.data.get("rationale") or {}
    return Path(r.get("dir") or Path(cfg.paths.cache_dir) / "rationales")


def attach_rationales(records, cfg):
    """Gắn nhãn phụ của teacher (`teacher_axis` = (text_sarcastic, image_sarcastic)) và embedding reasoning
    (`rationale_emb`) vào record train có reasoning đạt rubric. Trả về (records mới, số chiều embedding, số mẫu gắn được).
    Embedding chỉ đọc khi loss.lambda_rat > 0."""
    from vimmsd.data.rationale import load_rationale_dataset

    r = cfg.data.rationale
    root = rationale_dir(cfg)
    samples, _ = load_rationale_dataset(root / r.get("file", "rationale_dataset.json"),
                                        exclude_groups=r.get("exclude_groups") or [])
    embeddings, dim = {}, 0
    if cfg.loss.get("lambda_rat"):
        emb_path = root / r.get("embeddings", "rationale_embeddings.npy")
        matrix = np.load(emb_path).astype(np.float32)
        ids = json.loads(emb_path.with_name(emb_path.stem + "_ids.json").read_text(encoding="utf-8"))
        if len(ids) != len(matrix):
            raise ValueError(f"{emb_path}: {len(matrix)} vector nhưng {len(ids)} id")
        embeddings, dim = {str(sid): matrix[k] for k, sid in enumerate(ids)}, matrix.shape[1]
        expected = (cfg.model.get("head") or {}).get("rationale_dim")
        if expected and int(expected) != dim:
            raise ValueError(f"model.head.rationale_dim={expected} nhưng embedding reasoning có {dim} chiều")
    out, n = [], 0
    for rec in records:
        s = samples.get(rec["id"])
        if s is None:
            out.append(rec)
            continue
        extra = {"teacher_axis": (float(s["text_sarcastic"]), float(s["image_sarcastic"]))}
        if rec["id"] in embeddings:
            extra["rationale_emb"] = embeddings[rec["id"]]
        out.append({**rec, **extra})
        n += 1
    logger.info("reasoning của teacher: %d/%d mẫu train có nhãn phụ, %d có embedding (%d chiều)",
                n, len(records), sum("rationale_emb" in r for r in out), dim)
    return out, dim, n


def split_label_counts(cfg, splits=("train", "val")):
    """Số mẫu mỗi lớp của từng split (để tính prior cho hiệu chỉnh, xem heads.compute_prior_shift)."""
    records = load_all_records(cfg)
    n = len(cfg.data.labels)
    out = {}
    for split in splits:
        counts = Counter(r["label"] for r in records.get(split, []) if r["label"] != IGNORE_LABEL)
        out[split] = [counts.get(i, 0) for i in range(n)]
    return out


def build_datasets(cfg, needs_text=True, needs_image=True, splits=("train", "val", "test")):
    records = load_all_records(cfg)
    rationale_dim = None
    if uses_rationale(cfg) and "train" in splits:
        records["train"], rationale_dim, _ = attach_rationales(records["train"], cfg)
    text_pre = TextPreprocessor.from_config(cfg.data.text) if needs_text else None
    # màu viền khi pad và màu ảnh thay thế ảnh lỗi: mean của encoder, sau normalize gần 0
    fill = encoder_mean_color(cfg.model.image_encoder.name) if needs_image else (0, 0, 0)
    datasets = {}
    for split in splits:
        if split not in records:
            continue
        datasets[split] = ViMMSDDataset(
            records[split],
            text_preprocessor=text_pre,
            image_transform=build_image_transform(cfg, split == "train", fill) if needs_image else None,
            load_image=needs_image,
            cache_dir=text_cache_dir(cfg),
            use_image_text=uses_image_text(cfg) and needs_text,
            separate_image_text=separates_image_text(cfg) and needs_text,
            missing_image_color=fill,
            rationale_dim=rationale_dim if split == "train" else None,
        )
    return datasets


def class_counts(dataset, num_classes):
    counts = Counter(dataset.labels)
    return [counts.get(i, 0) for i in range(num_classes)]


class EpochBudgetSampler(Sampler):
    """Lấy đúng `num_samples` chỉ số mỗi epoch, không lặp trong một lượt hoán vị: hết lượt thì sang hoán vị mới,
    phần dư nối sang epoch sau. Dùng cho train.steps_per_epoch khi lấy mẫu tự nhiên (sampler: random), để số bước
    tối ưu mỗi epoch không phụ thuộc cỡ tập train (vd sau khi thêm mẫu augmentation)."""

    def __init__(self, n, num_samples, generator=None):
        self.n, self.num_samples, self.generator = n, num_samples, generator
        self._queue = []

    def __iter__(self):
        out = []
        while len(out) < self.num_samples:
            if not self._queue:
                self._queue = torch.randperm(self.n, generator=self.generator).tolist()
            take = min(self.num_samples - len(out), len(self._queue))
            out += self._queue[:take]
            self._queue = self._queue[take:]
        return iter(out)

    def __len__(self):
        return self.num_samples


def sampling_weights(labels, power):
    """Trọng số lấy mẫu mỗi mẫu = (số mẫu cùng lớp)^-power. power=1: cân bằng lớp hoàn toàn; 0.5: căn bậc hai
    nghịch tần suất (docs/empirical_analysis_plan.md mục 8.1, B2); 0: tự nhiên. Đếm trên chính tập train đưa vào."""
    counts = Counter(labels)
    return [float(counts[l]) ** -power for l in labels]


def build_train_sampler(dataset, num_classes, mode, power=None, num_samples=None):
    """mode: random (tự nhiên) | weighted (power = 1, như trước) | power (cần data.sampler_power).
    num_samples: số mẫu rút mỗi epoch (train.steps_per_epoch × batch × grad_accum); None = cỡ tập train."""
    if mode == "random":
        return None if num_samples is None else EpochBudgetSampler(len(dataset), num_samples)
    if mode == "weighted":
        power = 1.0
    elif mode == "power":
        if power is None:
            raise ValueError("data.sampler=power cần data.sampler_power (vd 0.5)")
    else:
        raise ValueError(f"sampler không hợp lệ: {mode}")
    weights = sampling_weights(dataset.labels, power)
    return WeightedRandomSampler(weights, num_samples=num_samples or len(weights), replacement=True)


def build_dataloaders(cfg, datasets, collator):
    num_classes = len(cfg.data.labels)
    loaders = {}
    for split, ds in datasets.items():
        is_train = split == "train"
        sampler = None
        if is_train:
            steps = cfg.train.get("steps_per_epoch")
            num_samples = steps * cfg.train.batch_size * cfg.train.grad_accum if steps else None
            sampler = build_train_sampler(ds, num_classes, cfg.data.sampler, cfg.data.get("sampler_power"), num_samples)
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.train.batch_size if is_train else cfg.train.eval_batch_size,
            shuffle=is_train and sampler is None,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
