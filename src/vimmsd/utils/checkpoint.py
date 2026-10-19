from pathlib import Path

import torch


def frozen_param_names(model):
    return {name for name, p in model.named_parameters() if not p.requires_grad}


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, **extra):
    """Không lưu parameter đóng băng (requires_grad=False): chúng giống hệt weight pretrained mà `build_model`
    tải lại từ config. Concat đóng băng đầy đủ khoảng 0,9 GB mỗi file, bỏ encoder chỉ còn vài MB;
    lưu đầy đủ thì vài chục run làm đầy 20 GB của /kaggle/working."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen = frozen_param_names(model)
    weights = {k: v for k, v in model.state_dict().items() if k not in frozen}
    state = {"model": weights, "frozen_omitted": bool(frozen), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    # ghi ra file tạm rồi rename: session free tier bị ngắt giữa lúc ghi sẽ không làm hỏng checkpoint cũ
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    """Checkpoint thiếu parameter đóng băng (xem `save_checkpoint`) thì giữ weight pretrained đang có trong `model`,
    nên `model` phải được dựng từ cùng config (cùng encoder, cùng chế độ đóng băng). Checkpoint cũ lưu đầy đủ vẫn load được."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    not_frozen = set(missing) - frozen_param_names(model)
    if unexpected or not_frozen:
        raise RuntimeError(f"checkpoint {path} không khớp model: thiếu {sorted(not_frozen)[:5]}, thừa {unexpected[:5]}")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return {k: v for k, v in state.items() if k not in {"model", "optimizer", "scheduler", "scaler", "frozen_omitted"}}


def resolve_resume_path(resume_from):
    """`resume_from` có thể là file .pt hoặc thư mục (ví dụ output version trước được attach
    làm input trên Kaggle) — với thư mục thì lấy last.pt bên trong, tìm đệ quy."""
    if not resume_from:
        return None
    p = Path(resume_from)
    if p.is_file():
        return p
    if p.is_dir():
        candidates = sorted(p.rglob("last.pt"), key=lambda f: f.stat().st_mtime)
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"không tìm thấy checkpoint tại {resume_from}")
