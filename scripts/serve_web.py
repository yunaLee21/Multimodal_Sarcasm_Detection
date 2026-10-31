"""Web xem dữ liệu ViMMSD và cho từng model đã train suy luận trực tiếp một mẫu.

Chạy trên máy có dữ liệu + checkpoint (GPU server), mở từ máy mình qua SSH tunnel:
    python scripts/serve_web.py --runs-dir data/server/outputs
    ssh -L 7860:localhost:7860 <server>     # rồi mở http://localhost:7860

--runs-dir (lặp lại được): thư mục chứa các run, mỗi run có best.pt + config.json. Output tải từ Kaggle để đâu cũng được.
--allow-private: hiện private test và điểm test của từng run. Tắt mặc định vì private test chỉ chấm ở bước cuối."""
import argparse
import logging

from vimmsd.utils.config import load_config
from vimmsd.web.app import create_app
from vimmsd.web.inference import DataStore, InferenceEngine, RunRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="lấy đường dẫn dữ liệu và danh sách nhãn")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--runs-dir", action="append", default=None, help="mặc định: paths.output_dir")
    parser.add_argument("--allow-private", action="store_true")
    parser.add_argument("--device", help="mặc định: cuda nếu có")
    parser.add_argument("--max-loaded", type=int, default=2, help="số model giữ trong bộ nhớ cùng lúc")
    parser.add_argument("--host", default="127.0.0.1", help="mặc định chỉ nhận kết nối từ chính máy này")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config, args.override)
    store = DataStore(cfg, allow_private=args.allow_private)
    registry = RunRegistry(args.runs_dir or [cfg.paths.output_dir], show_test_metrics=args.allow_private)
    engine = InferenceEngine(store, registry, cfg.paths, device=args.device, max_loaded=args.max_loaded)
    logging.info("splits %s | %d run | device %s", {k: len(v) for k, v in store.splits.items()},
                 len(registry.runs), engine.device)
    create_app(store, registry, engine).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
