"""Flask app của web xem dữ liệu và thử model. Chạy bằng scripts/serve_web.py."""
import logging
import random
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
PAGE_SIZE = 30


def _sample_summary(rec):
    return {"id": rec["id"], "caption": rec["caption"], "label": rec["label"]}


def create_app(store, registry, engine):
    app = Flask(__name__, static_folder=None)
    app.json.sort_keys = False  # giữ thứ tự nhãn và thứ tự thông số, Flask mặc định xếp theo bảng chữ cái

    def _split():
        split = request.args.get("split", "")
        if split not in store.splits:
            abort(404, f"không có split {split!r}")
        return split

    @app.get("/")
    def index():
        return send_file(STATIC_DIR / "index.html")

    @app.get("/api/meta")
    def meta():
        return jsonify(labels=store.labels, warnings=store.warnings, device=str(engine.device),
                       splits={name: {"size": len(rows), "counts": store.label_counts(name)}
                               for name, rows in store.splits.items()})

    @app.get("/api/runs")
    def runs():
        return jsonify(runs=registry.public_list())

    @app.get("/api/samples")
    def samples():
        rows = store.search(_split(), request.args.get("label") or None, request.args.get("q", "").strip() or None)
        if request.args.get("random") == "1":
            return jsonify(item=_sample_summary(random.choice(rows)) if rows else None, total=len(rows))
        page = max(request.args.get("page", 0, type=int), 0)
        chunk = rows[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
        return jsonify(items=[_sample_summary(r) for r in chunk], total=len(rows), page=page, page_size=PAGE_SIZE)

    @app.get("/api/sample")
    def sample():
        rec = store.get(_split(), request.args.get("id", ""))
        if rec is None:
            abort(404, "không có mẫu này")
        return jsonify(**_sample_summary(rec), ocr=rec.get("ocr", ""), description=rec.get("description", ""),
                       has_image=Path(rec["image_path"]).is_file())

    @app.get("/image")
    def image():
        # đường dẫn lấy từ record theo id, không nhận đường dẫn từ client
        rec = store.get(_split(), request.args.get("id", ""))
        if rec is None or not Path(rec["image_path"]).is_file():
            abort(404, "không có ảnh")
        return send_file(Path(rec["image_path"]).resolve(), max_age=3600)

    @app.get("/api/predict")
    def predict():
        split = _split()
        run_id = request.args.get("run", "")
        if run_id not in registry.runs:
            abort(404, f"không có run {run_id!r}")
        sample_id = request.args.get("id", "")
        if store.get(split, sample_id) is None:
            abort(404, "không có mẫu này")
        try:
            return jsonify(engine.predict(run_id, split, sample_id, prior=request.args.get("prior") == "1"))
        except Exception as e:  # checkpoint lệch code hay thiếu file chỉ làm hỏng run đó, các run khác vẫn chạy
            logger.exception("run %s lỗi", run_id)
            return jsonify(error=f"{type(e).__name__}: {e}"), 500

    @app.errorhandler(404)
    def not_found(e):
        return jsonify(error=e.description), 404

    return app
