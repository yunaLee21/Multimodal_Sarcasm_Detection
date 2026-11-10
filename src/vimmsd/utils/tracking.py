"""Theo dõi run bằng wandb (bật bằng wandb.enabled trong config, mặc định tắt).

Tracking chỉ là phần phụ: thiếu thư viện, không có mạng hay lỗi từ wandb đều chỉ ghi cảnh báo, không làm dừng train.
Kết quả chính vẫn nằm ở output_dir và reports/results.md."""
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

RUN_ID_FILE = "wandb_run_id.txt"


def metric_scalars(metrics, prefix):
    """Metric cần theo dõi từ compute_metrics, dạng phẳng <prefix>/<tên>, bỏ confusion matrix."""
    scalars = {f"{prefix}/{k}": metrics[k] for k in ("macro_f1", "weighted_f1", "accuracy") if k in metrics}
    scalars.update({f"{prefix}/f1_{name}": m["f1"] for name, m in (metrics.get("per_class") or {}).items()})
    return scalars


def run_names(cfg):
    """(tên run, tên nhóm): các seed của cùng một config vào chung một nhóm."""
    suffix = f"_s{cfg.seed}"
    name = cfg.name if cfg.name.endswith(suffix) else cfg.name + suffix
    return name, re.sub(r"_s\d+$", "", name)


class NullTracker:
    def log(self, data, step=None):
        pass

    def summary(self, data):
        pass

    def finish(self):
        pass


class WandbTracker:
    def __init__(self, run):
        self.run = run

    def _safe(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.warning("wandb lỗi, bỏ qua: %s", e)

    def log(self, data, step=None):
        self._safe(self.run.log, data, step=step)

    def summary(self, data):
        self._safe(self.run.summary.update, data)

    def finish(self):
        self._safe(self.run.finish)


def start_tracking(cfg, commit, output_dir):
    """Tạo run wandb, config tắt hoặc không tạo được thì trả NullTracker.
    train.resume_from có giá trị và còn run id cũ trong output_dir thì ghi tiếp vào run đó."""
    settings = cfg.get("wandb") or {}
    if not settings.get("enabled"):
        return NullTracker()
    output_dir = Path(output_dir)
    id_file = output_dir / RUN_ID_FILE
    run_id = id_file.read_text().strip() if cfg.train.get("resume_from") and id_file.exists() else None
    name, group = run_names(cfg)
    try:
        import wandb

        run = wandb.init(project=settings.get("project"), entity=settings.get("entity"), name=name, group=group,
                         config={**cfg.to_dict(), "commit": commit}, dir=str(output_dir),
                         id=run_id, resume="allow" if run_id else None)
    except Exception as e:  # ImportError, thiếu API key, không có mạng...
        logger.warning("không tạo được run wandb, train tiếp không tracking: %s", e)
        return NullTracker()
    id_file.write_text(run.id)
    return WandbTracker(run)
