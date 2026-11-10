"""Phần không có HTTP của web xem dữ liệu: đọc mẫu, tìm các run đã train, suy luận một mẫu bằng từng run.

Suy luận đi qua đúng các lớp của scripts/evaluate.py (build_model, ViMMSDDataset, ViMMSDCollator) để đầu vào
giống hệt lúc đánh giá. Khác biệt duy nhất: tiền xử lý caption không bao giờ ghi vào file cache của nhóm."""
import gc
import json
import logging
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path

import torch

from vimmsd.data.dataset import (IGNORE_LABEL, ViMMSDCollator, ViMMSDDataset, build_image_transform, load_all_records,
                                 load_image_texts, separates_image_text, text_cache_dir, uses_image_text)
from vimmsd.data.image_text import image_key
from vimmsd.data.preprocessing import TextPreprocessor
from vimmsd.models.classifier import MODALITY_NEEDS, build_model
from vimmsd.models.heads import compute_prior_shift
from vimmsd.training.trainer import move_to_device
from vimmsd.utils.checkpoint import load_checkpoint
from vimmsd.utils.config import _to_config

logger = logging.getLogger(__name__)

PUBLIC_SPLITS = ("public_test", "train")  # public_test đứng đầu: mặc định xem trên val, không xem trên dữ liệu đã train
PRIVATE_SPLIT = "private_test"


class DataStore:
    """Mọi mẫu theo split, đọc bằng load_all_records để trùng với lúc train. Nhãn lưu bằng tên, không phải số,
    vì thứ tự nhãn thuộc về từng config."""

    def __init__(self, cfg, allow_private=False):
        self.warnings = []
        self.labels = list(cfg.data.labels)
        records = load_all_records(self._display_config(cfg))
        names = [*PUBLIC_SPLITS, *([PRIVATE_SPLIT] if allow_private else [])]
        self.splits, self._index = {}, {}
        for name in names:
            if name not in records:
                continue
            rows = [{**r, "label": self.labels[r["label"]] if r["label"] != IGNORE_LABEL else None}
                    for r in records[name]]
            self.splits[name] = rows
            self._index[name] = {r["id"]: r for r in rows}

    def _display_config(self, cfg):
        """Bật OCR + mô tả để hiện cho người xem. Thiếu file cache thì vẫn xem được ảnh và caption."""
        plain = cfg.to_dict()
        it = plain["data"].setdefault("image_text", {})
        path = Path(it.get("dir") or plain["paths"]["cache_dir"]) / (it.get("file") or "image_preprocessed.json")
        available = path.exists()
        if not available:
            self.warnings.append(f"không thấy {path}: không hiện được OCR và mô tả ảnh")
        it.update(use_ocr=available, use_description=available)
        return _to_config(plain)

    def get(self, split, sample_id):
        return self._index.get(split, {}).get(str(sample_id))

    def label_counts(self, split):
        return Counter(r["label"] for r in self.splits.get(split, []) if r["label"] is not None)

    def search(self, split, label=None, query=None):
        rows = self.splits.get(split, [])
        if label:
            rows = [r for r in rows if r["label"] == label]
        if query:
            q = query.casefold()
            rows = [r for r in rows if r["id"] == query or q in r["caption"].casefold()]
        return rows


def _read_metrics(path):
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    return {"macro_f1": m.get("macro_f1"), "weighted_f1": m.get("weighted_f1"), "accuracy": m.get("accuracy"),
            "per_class_f1": {k: v.get("f1") for k, v in (m.get("per_class") or {}).items()}}


def _run_params(raw):
    """Thông số hiện trên web, lấy từ config.json mà train.py ghi lúc train."""
    model, loss, train, data = (raw.get(k) or {} for k in ("model", "loss", "train", "data"))
    needs_text, needs_image = MODALITY_NEEDS.get(model.get("modality"), (True, True))
    encoder = lambda e: f"{e.get('name')} (freeze={e.get('freeze')}, unfreeze_last_n={e.get('unfreeze_last_n', 0)})"
    head = model.get("head") or {}
    image_text = data.get("image_text") or {}
    params = {
        "modality": model.get("modality"),
        "text_encoder": encoder(model.get("text_encoder") or {}) if needs_text else None,
        "image_encoder": encoder(model.get("image_encoder") or {}) if needs_image else None,
        "fusion": model.get("fusion") or None,
        "head": head.get("type", "flat"),
        "axis_heads": head.get("axis_heads") or None,
        "axis_targets": head.get("axis_targets") if head.get("axis_heads") else None,
        "rationale_dim": head.get("rationale_dim") or None,
        "head_hidden": model.get("head_hidden"),
        "dropout": model.get("dropout"),
        "loss": loss.get("name"),
        "class_weight": loss.get("class_weight"),
        "lambda_s": loss.get("lambda_s") or None,
        "lambda_axis": loss.get("lambda_axis") or None,
        "lambda_rat": loss.get("lambda_rat") or None,
        "epochs": train.get("epochs"),
        "batch_size": train.get("batch_size"),
        "lr_encoder": train.get("lr_encoder"),
        "lr_head": train.get("lr_head"),
        "sampler": data.get("sampler"),
        "data_split": data.get("split"),
        "use_ocr": image_text.get("use_ocr"),
        "use_description": image_text.get("use_description"),
        "seed": raw.get("seed"),
        "commit": raw.get("commit"),
    }
    return {k: v for k, v in params.items() if v is not None}


class RunRegistry:
    """Run là một thư mục có best.pt và config.json, đúng bố cục train.py ghi ra. Tìm đệ quy vì output tải từ
    Kaggle thường nằm lồng thêm vài cấp."""

    def __init__(self, runs_dirs, show_test_metrics=False):
        self.runs = {}
        for root in map(Path, runs_dirs):
            for ckpt in sorted(root.rglob("best.pt")):
                if not (ckpt.parent / "config.json").exists():
                    logger.warning("bỏ qua %s: thiếu config.json", ckpt)
                    continue
                run_id = ckpt.parent.relative_to(root).as_posix()
                if run_id == ".":
                    run_id = root.name
                if run_id in self.runs:
                    run_id = f"{root.name}/{run_id}"
                self.runs[run_id] = self._describe(run_id, ckpt.parent, show_test_metrics)

    @staticmethod
    def _describe(run_id, run_dir, show_test_metrics):
        raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = {"val": _read_metrics(run_dir / "best_val_metrics.json")}
        if show_test_metrics:  # điểm private test chỉ dùng ở bước cuối, không hiện khi đang chọn model
            metrics["test"] = _read_metrics(run_dir / "test_metrics.json")
        return {"id": run_id, "name": raw.get("name", run_dir.name), "dir": str(run_dir),
                "params": _run_params(raw), "metrics": metrics}

    def public_list(self):
        return [{k: r[k] for k in ("id", "name", "params", "metrics")} for r in self.runs.values()]


class ReadOnlyPreprocessor:
    """Thay TextPreprocessor.process_all: đọc cache một lần và không ghi lại. Bản gốc đọc cả file JSON ở mỗi lần gọi,
    và khi lệch pipeline_key sẽ ghi đè cache của nhóm bằng đúng vài câu vừa xử lý."""

    def __init__(self, preprocessor, cache_dir):
        self.preprocessor = preprocessor
        self.cache = preprocessor.load_cache(preprocessor._cache_file(cache_dir)) if cache_dir else {}
        self.misses = 0

    def process_all(self, texts, cache_dir=None):
        out = []
        for text in texts:
            if text in self.cache:
                out.append(self.cache[text])
            else:
                self.misses += 1
                out.append(self.preprocessor(text))
        return out


class InferenceEngine:
    """Giữ tối đa `max_loaded` model trong bộ nhớ (LRU). Mỗi lúc chỉ suy luận một mẫu, tránh tràn VRAM và tránh hai
    request cùng đổi prior_shift của một model."""

    def __init__(self, store, registry, base_paths, device=None, max_loaded=2):
        self.store = store
        self.registry = registry
        self.base_paths = dict(base_paths)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_loaded = max(1, max_loaded)
        self._models = OrderedDict()
        self._preprocessors = {}
        self._image_texts = {}
        self._lock = threading.Lock()

    def run_config(self, run_dir):
        """config.json của run, với đường dẫn của máy đang chạy web (run có thể đã train trên Kaggle)."""
        raw = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
        raw["paths"] = self.base_paths
        data = raw["data"]
        (data.get("text") or {})["cache_dir"] = None
        if data.get("image_text"):
            data["image_text"]["dir"] = None
        if data.get("rationale"):
            data["rationale"]["use"] = False  # reasoning chỉ dùng lúc train
        return _to_config(raw)

    def _load(self, run_id):
        if run_id in self._models:
            self._models.move_to_end(run_id)
            return self._models[run_id]
        run = self.registry.runs[run_id]
        while len(self._models) >= self.max_loaded:
            self._models.popitem(last=False)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        cfg = self.run_config(run["dir"])
        model = build_model(cfg)
        extra = load_checkpoint(Path(run["dir"]) / "best.pt", model, map_location="cpu")
        model.to(self.device).eval()
        info = {"num_params": sum(p.numel() for p in model.parameters()),
                "num_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "best_epoch": extra.get("epoch")}
        self._models[run_id] = (model, cfg, info)
        return self._models[run_id]

    def _preprocessor(self, cfg):
        pre = TextPreprocessor.from_config(cfg.data.text)
        key = (pre.pipeline_key(), pre.cache_file)
        if key not in self._preprocessors:
            self._preprocessors[key] = ReadOnlyPreprocessor(pre, text_cache_dir(cfg))
        return self._preprocessors[key]

    def _record(self, cfg, rec):
        """Chỉ gắn OCR / mô tả mà config của run dùng: dataset ghép mọi field có trong record vào đầu vào."""
        it = cfg.data.get("image_text") or {}
        key = json.dumps([it.get(k) for k in ("use_ocr", "use_description", "file", "ocr_cache", "description_cache")])
        if key not in self._image_texts:
            self._image_texts[key] = load_image_texts(cfg)
        path = Path(rec["image_path"])
        row = {"id": rec["id"], "image_path": rec["image_path"], "caption": rec["caption"], "label": IGNORE_LABEL}
        for field, cache in self._image_texts[key].items():
            row[field] = cache.get(image_key(path.parent, path.name), "")
        return row

    def _batch(self, model, cfg, rec):
        needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
        # cùng giá trị với dataset.encoder_mean_color, nhưng không đọc lại processor từ đĩa ở mỗi lần suy luận
        fill = tuple(round(m * 255) for m in model.image_processor.image_mean) if needs_image else (0, 0, 0)
        pre = self._preprocessor(cfg) if needs_text else None
        misses = pre.misses if pre else 0
        dataset = ViMMSDDataset(
            [self._record(cfg, rec)],
            text_preprocessor=pre,
            image_transform=build_image_transform(cfg, False, fill) if needs_image else None,
            load_image=needs_image,
            use_image_text=uses_image_text(cfg) and needs_text,
            separate_image_text=separates_image_text(cfg) and needs_text,
            missing_image_color=fill,
        )
        item = dataset[0]
        aux_max_length = (cfg.data.get("image_text") or {}).get("aux_max_length") or 96
        collator = ViMMSDCollator(model.tokenizer, model.image_processor, cfg.data.max_length,
                                  cfg.data.get("max_caption_length"), aux_max_length)
        shown = {k: item[k] for k in ("text", "image_text", "ocr", "description") if item.get(k)}
        warnings = []
        if pre and pre.misses > misses:
            warnings.append("Text không có trong cache hoặc lệch pipeline_key, đã tiền xử lý lại nên có thể khác lúc train.")
        if item.get("img_missing"):
            warnings.append("Không đọc được ảnh, model nhận ảnh một màu thay thế.")
        return collator([item]), shown, warnings

    def _prior_shift(self, model, labels):
        train, target = self.store.label_counts("train"), self.store.label_counts("public_test")
        return compute_prior_shift(model.head_type, [train[l] for l in labels], [target[l] for l in labels], labels)

    @torch.no_grad()
    def predict(self, run_id, split, sample_id, prior=False):
        rec = self.store.get(split, sample_id)
        if rec is None:
            raise KeyError(f"không có mẫu {sample_id} trong {split}")
        with self._lock:
            start = time.perf_counter()
            model, cfg, info = self._load(run_id)
            labels = list(cfg.data.labels)
            batch, shown, warnings = self._batch(model, cfg, rec)
            batch = move_to_device(batch, self.device)
            model.set_prior_shift(self._prior_shift(model, labels) if prior else None)
            try:
                use_amp = bool(cfg.train.get("amp")) and self.device.type == "cuda"
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=use_amp):
                    out = model.forward_outputs(batch)
            finally:
                model.set_prior_shift(None)
            probs = out["logits"].float().softmax(-1)[0].cpu().tolist()
            pred = labels[max(range(len(probs)), key=probs.__getitem__)]
            axis = None
            if "axis_logits" in out:
                p_text, p_image = torch.sigmoid(out["axis_logits"].float())[0].cpu().tolist()
                axis = {"text_alone": p_text, "image_alone": p_image}
            return {"run": run_id, "pred": pred, "probs": dict(zip(labels, probs)), "gold": rec["label"],
                    "correct": None if rec["label"] is None else pred == rec["label"], "axis": axis, "prior": prior,
                    "input": shown, "warnings": warnings, "seconds": round(time.perf_counter() - start, 3),
                    "device": str(self.device), "model": info}
