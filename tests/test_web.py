"""Test web xem dữ liệu: model tí hon thật, dữ liệu giả, không cần GPU, mạng hay checkpoint thật."""
import json

import pytest
import torch
from PIL import Image

pytest.importorskip("flask")

from vimmsd.data.dataset import ViMMSDCollator, build_dataloaders, build_datasets  # noqa: E402
from vimmsd.models.classifier import MODALITY_NEEDS, build_model  # noqa: E402
from vimmsd.training.trainer import predict  # noqa: E402
from vimmsd.utils.checkpoint import save_checkpoint  # noqa: E402
from vimmsd.utils.config import load_config  # noqa: E402
from vimmsd.web.app import create_app  # noqa: E402
from vimmsd.web.inference import DataStore, InferenceEngine, RunRegistry  # noqa: E402

LABELS = ["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"]


class FakeTokenizer:
    """Mỗi ký tự một token, đủ để nội dung text làm đổi kết quả của model."""

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def tokenize(self, text):
        return list(text)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def __call__(self, texts, pairs=None, padding=True, truncation=True, max_length=32, return_tensors="pt"):
        pairs = pairs or [""] * len(texts)
        rows = [[1 + ord(c) % 45 for c in (a + b)][:30] or [1] for a, b in zip(texts, pairs)]
        n = max(map(len, rows))
        return {"input_ids": torch.tensor([r + [0] * (n - len(r)) for r in rows]),
                "attention_mask": torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in rows])}


@pytest.fixture
def tiny_encoders(monkeypatch):
    """build_model thật, chỉ thay pretrained bằng encoder tí hon. Seed cố định để weight đóng băng (không lưu trong
    checkpoint) giống nhau giữa lúc lưu và lúc nạp, như pretrained thật."""
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, RobertaConfig, RobertaModel

    from vimmsd.models import image_encoder, text_encoder

    def text_model(name):
        torch.manual_seed(1)
        return RobertaModel(RobertaConfig(vocab_size=50, max_position_embeddings=40, hidden_size=16,
                                          num_hidden_layers=1, num_attention_heads=2, intermediate_size=32))

    def image_model(name):
        torch.manual_seed(2)
        return CLIPVisionModel(CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                                num_attention_heads=2, image_size=16, patch_size=8))

    processor = CLIPImageProcessor(size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16})
    monkeypatch.setattr(text_encoder.AutoModel, "from_pretrained", text_model)
    monkeypatch.setattr(text_encoder.AutoTokenizer, "from_pretrained", lambda name: FakeTokenizer())
    monkeypatch.setattr(image_encoder.AutoConfig, "from_pretrained", lambda name: CLIPVisionConfig())
    monkeypatch.setattr(image_encoder.CLIPVisionModel, "from_pretrained", image_model)
    monkeypatch.setattr(image_encoder.AutoImageProcessor, "from_pretrained", lambda name: processor)
    monkeypatch.setattr("transformers.AutoImageProcessor.from_pretrained", lambda name: processor)


@pytest.fixture
def workspace(tmp_path):
    data, cache = tmp_path / "data", tmp_path / "cache"
    cache.mkdir()
    image_cache = {}

    def split(json_name, image_dir, n, labels_name=None):
        folder = data / image_dir
        folder.mkdir(parents=True)
        items, ref = {}, {}
        for i in range(n):
            name = f"{json_name[:5]}{i}.jpg"
            Image.new("RGB", (20 + i, 24), (40 * i % 255, 90, 160)).save(folder / name)
            label = LABELS[i % 4]
            items[str(i)] = {"image": name, "caption": f"caption {json_name[7:12]} số {i}",
                             "label": None if labels_name else label}
            ref[str(i)] = label
            image_cache[f"{folder.name}/{name}"] = {"ocr": f"chữ {i}", "description": f"mô tả {i}"}
        (data / json_name).write_text(json.dumps(items), encoding="utf-8")
        if labels_name:
            (data / labels_name).write_text(json.dumps({"results": ref}), encoding="utf-8")

    data.mkdir()
    split("vimmsd-train.json", "training-images/train-images", 8)
    split("vimmsd-public-test.json", "public-test-images/dev-images", 6, "vimmsd-public-test-ref.json")
    split("vimmsd-private-test.json", "private-test-images/test-images", 4, "vimmsd-private-test-ref.json")
    (cache / "image_preprocessed.json").write_text(json.dumps({"meta": {}, "images": image_cache}), encoding="utf-8")
    return tmp_path


def _config(workspace, path="configs/base.yaml", overrides=()):
    return load_config(path, [f"paths.local.data_dir={workspace / 'data'}", f"paths.local.cache_dir={workspace / 'cache'}",
                              f"paths.local.output_dir={workspace / 'runs'}", "data.text.word_segment=none",
                              "train.num_workers=0", "train.amp=false", *overrides], env="local")


def _save_run(workspace, name, path, overrides=(), kaggle_paths=True):
    """Lưu run đúng bố cục train.py ghi ra. kaggle_paths: config.json mang đường dẫn của máy đã train."""
    cfg = _config(workspace, path, [f"name={name}", *overrides])
    model = build_model(cfg)
    run_dir = workspace / "runs" / name
    save_checkpoint(run_dir / "best.pt", model, epoch=3, best_f1=0.5)
    plain = {**cfg.to_dict(), "commit": "abc1234"}
    if kaggle_paths:
        plain["paths"] = {"data_dir": "/kaggle/input/x", "output_dir": "/kaggle/working/outputs", "cache_dir": "/kaggle/working/cache"}
        (plain["data"].get("image_text") or {})["dir"] = "/kaggle/input/y"
    (run_dir / "config.json").write_text(json.dumps(plain), encoding="utf-8")
    metrics = {"macro_f1": 0.5, "weighted_f1": 0.6, "accuracy": 0.7, "per_class": {"not-sarcasm": {"f1": 0.8}}}
    (run_dir / "best_val_metrics.json").write_text(json.dumps(metrics))
    (run_dir / "test_metrics.json").write_text(json.dumps({**metrics, "macro_f1": 0.11}))
    return cfg, model


def _engine(workspace, allow_private=False, **kw):
    cfg = _config(workspace)
    store = DataStore(cfg, allow_private=allow_private)
    registry = RunRegistry([workspace / "runs"], show_test_metrics=allow_private)
    return InferenceEngine(store, registry, cfg.paths, device="cpu", **kw), store, registry


RUN_CONFIGS = [
    ("configs/text_only_frozen.yaml", []),
    ("configs/image_only_frozen.yaml", []),
    ("configs/fusion_concat_frozen.yaml", []),
    ("configs/text_only_imgtext.yaml", []),
    ("configs/fusion_multistream_clip_frozen.yaml", []),
    ("configs/fusion_multistream_clip_frozen.yaml", ["model.head.type=hierarchical", "model.head.axis_heads=true"]),
]


@pytest.mark.parametrize("path,overrides", RUN_CONFIGS)
def test_predict_matches_evaluate_pipeline(tiny_encoders, workspace, path, overrides):
    """Xác suất trên web phải trùng đường evaluate.py cho mọi loại modality, nếu không người chấm đang xem sai model."""
    cfg, model = _save_run(workspace, "run", path, overrides, kaggle_paths=False)
    needs_text, needs_image = MODALITY_NEEDS[cfg.model.modality]
    datasets = build_datasets(cfg, needs_text, needs_image, splits=("val",))
    aux = (cfg.data.get("image_text") or {}).get("aux_max_length") or 96
    collator = ViMMSDCollator(model.tokenizer, model.image_processor, cfg.data.max_length,
                              cfg.data.get("max_caption_length"), aux)
    out = predict(model, build_dataloaders(cfg, datasets, collator)["val"], torch.device("cpu"))

    engine, _, _ = _engine(workspace)
    for k, sample_id in enumerate(out["ids"]):
        r = engine.predict("run", "public_test", sample_id)
        assert list(r["probs"].values()) == pytest.approx(out["probs"][k].tolist(), abs=1e-5)
        assert r["pred"] == LABELS[out["preds"][k]]
        assert r["correct"] == (r["pred"] == r["gold"])
    assert ("axis_heads=true" in " ".join(overrides)) == (r["axis"] is not None)


def test_kaggle_run_uses_local_paths(tiny_encoders, workspace):
    _save_run(workspace, "kaggle_run", "configs/text_only_imgtext.yaml")
    engine, _, registry = _engine(workspace)
    r = engine.predict("kaggle_run", "public_test", "0")
    assert "chữ 0" in r["input"]["ocr"] and sum(r["probs"].values()) == pytest.approx(1)  # text_aux: OCR là field riêng
    assert registry.runs["kaggle_run"]["params"]["commit"] == "abc1234"
    assert r["model"]["best_epoch"] == 3 and r["model"]["num_params"] >= r["model"]["num_trainable"] > 0


def test_text_cache_of_team_is_never_overwritten(tiny_encoders, workspace):
    """Cache lệch pipeline_key: TextPreprocessor.process_all gốc sẽ ghi đè bằng vài câu, web không được làm vậy."""
    _save_run(workspace, "run", "configs/text_only_frozen.yaml")
    cache_file = workspace / "cache" / "text_preprocessed.json"
    original = json.dumps({"meta": {"pipeline_key": "khac"}, "texts": {"a": "b"}})
    cache_file.write_text(original, encoding="utf-8")
    engine, _, _ = _engine(workspace)
    r = engine.predict("run", "public_test", "1")
    assert cache_file.read_text(encoding="utf-8") == original
    assert any("pipeline_key" in w for w in r["warnings"])


def test_prior_changes_probs_and_is_switched_off_afterwards(tiny_encoders, workspace):
    (workspace / "data" / "vimmsd-public-test-ref.json").write_text(
        json.dumps({"results": {str(i): "multi-sarcasm" for i in range(6)}}))
    _save_run(workspace, "run", "configs/text_only_frozen.yaml")
    engine, _, _ = _engine(workspace)
    plain = engine.predict("run", "public_test", "0")
    shifted = engine.predict("run", "public_test", "0", prior=True)
    assert shifted["probs"]["multi-sarcasm"] > plain["probs"]["multi-sarcasm"]
    assert engine.predict("run", "public_test", "0")["probs"] == pytest.approx(plain["probs"])


def test_keeps_at_most_max_loaded_models(tiny_encoders, workspace):
    for name in ("a", "b", "c"):
        _save_run(workspace, name, "configs/text_only_frozen.yaml")
    engine, _, _ = _engine(workspace, max_loaded=2)
    for name in ("a", "b", "c", "a"):
        engine.predict(name, "public_test", "0")
    assert list(engine._models) == ["c", "a"]


def test_private_test_hidden_unless_allowed(tiny_encoders, workspace):
    _save_run(workspace, "run", "configs/text_only_frozen.yaml")
    engine, store, registry = _engine(workspace)
    client = create_app(store, registry, engine).test_client()
    assert set(client.get("/api/meta").json["splits"]) == {"public_test", "train"}
    assert client.get("/api/samples?split=private_test").status_code == 404
    assert client.get("/image?split=private_test&id=0").status_code == 404
    assert client.get("/api/predict?split=private_test&id=0&run=run").status_code == 404
    assert "test" not in client.get("/api/runs").json["runs"][0]["metrics"]

    engine, store, registry = _engine(workspace, allow_private=True)
    client = create_app(store, registry, engine).test_client()
    assert client.get("/api/samples?split=private_test").json["total"] == 4
    assert client.get("/api/runs").json["runs"][0]["metrics"]["test"]["macro_f1"] == 0.11


def test_api_browse_and_predict(tiny_encoders, workspace):
    _save_run(workspace, "good", "configs/text_only_frozen.yaml")
    _save_run(workspace, "broken", "configs/text_only_frozen.yaml")
    (workspace / "runs" / "broken" / "best.pt").write_bytes(b"hong")
    engine, store, registry = _engine(workspace)
    client = create_app(store, registry, engine).test_client()

    assert b"ViMMSD Viewer" in client.get("/").data
    meta = client.get("/api/meta").json
    assert meta["labels"] == LABELS and meta["splits"]["public_test"]["size"] == 6
    assert meta["splits"]["public_test"]["counts"]["multi-sarcasm"] == 1

    assert client.get("/api/samples?split=public_test&label=not-sarcasm").json["total"] == 2
    assert [i["id"] for i in client.get("/api/samples?split=public_test&q=SỐ 3").json["items"]] == ["3"]
    assert client.get("/api/samples?split=train&random=1").json["item"]["id"] in {str(i) for i in range(8)}
    assert client.get("/api/samples?split=train&q=khongco&random=1").json["item"] is None

    sample = client.get("/api/sample?split=public_test&id=2").json
    assert sample["label"] == "image-sarcasm" and sample["ocr"] == "chữ 2" and sample["has_image"]
    assert client.get("/image?split=public_test&id=2").mimetype == "image/jpeg"
    assert client.get("/image?split=public_test&id=../../x").status_code == 404
    assert client.get("/api/sample?split=public_test&id=99").status_code == 404

    run = next(r for r in client.get("/api/runs").json["runs"] if r["id"] == "good")
    assert run["params"]["modality"] == "text" and "image_encoder" not in run["params"]
    assert run["metrics"]["val"]["macro_f1"] == 0.5

    assert client.get("/api/predict?split=public_test&id=2&run=khongco").status_code == 404
    assert client.get("/api/predict?split=public_test&id=99&run=good").status_code == 404
    bad = client.get("/api/predict?split=public_test&id=2&run=broken")
    assert bad.status_code == 500 and bad.json["error"]
    good = client.get("/api/predict?split=public_test&id=2&run=good")  # run hỏng không ảnh hưởng run khác
    assert good.status_code == 200 and good.json["gold"] == "image-sarcasm"
    assert list(json.loads(good.data)["probs"]) == LABELS  # giao diện vẽ cột theo đúng thứ tự này


def test_missing_image_is_reported_not_fatal(tiny_encoders, workspace):
    _save_run(workspace, "run", "configs/fusion_concat_frozen.yaml")
    engine, store, registry = _engine(workspace)
    next((workspace / "data" / "public-test-images" / "dev-images").glob("*0.jpg")).unlink()
    client = create_app(store, registry, engine).test_client()
    assert client.get("/api/sample?split=public_test&id=0").json["has_image"] is False
    assert client.get("/image?split=public_test&id=0").status_code == 404
    r = client.get("/api/predict?split=public_test&id=0&run=run").json
    assert any("ảnh" in w for w in r["warnings"])


def test_missing_image_text_cache_still_browses(tiny_encoders, workspace):
    (workspace / "cache" / "image_preprocessed.json").unlink()
    store = DataStore(_config(workspace))
    assert store.get("public_test", "0")["caption"] and "ocr" not in store.get("public_test", "0")
    assert any("image_preprocessed.json" in w for w in store.warnings)
