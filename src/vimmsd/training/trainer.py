import json
import logging
import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from vimmsd.training.metrics import compute_metrics, format_metrics
from vimmsd.utils.checkpoint import load_checkpoint, resolve_resume_path, save_checkpoint
from vimmsd.utils.tracking import NullTracker, metric_scalars

logger = logging.getLogger(__name__)


def selection_score(metrics, metric="macro_f1", exclude=()):
    """Điểm chọn checkpoint trên val. macro_f1 với `exclude` (vd [text-sarcasm], chỉ 1 mẫu trong val) tính
    trung bình F1 các lớp còn lại; weighted_f1 lấy thẳng từ metrics."""
    if metric == "weighted_f1":
        return metrics["weighted_f1"]
    if metric != "macro_f1":
        raise ValueError(f"train.select_metric phải là macro_f1 | weighted_f1, nhận được {metric!r}")
    if not exclude:
        return metrics["macro_f1"]
    f1 = [m["f1"] for name, m in metrics["per_class"].items() if name not in set(exclude)]
    return sum(f1) / len(f1)


def move_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device, use_amp=False):
    model.eval()
    ids, preds, labels, probs = [], [], [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch = move_to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(batch)
        p = logits.float().softmax(-1)
        ids.extend(batch["ids"])
        probs.append(p.cpu())
        preds.extend(p.argmax(-1).tolist())
        labels.extend(batch["labels"].tolist())
    return {"ids": ids, "preds": preds, "labels": labels, "probs": torch.cat(probs) if probs else None}


class Trainer:
    """Vòng lặp train/eval dùng chung cho mọi thí nghiệm.

    Mỗi epoch lưu `last.pt` (để resume khi session bị ngắt) và `best.pt` theo macro-F1 trên val."""

    def __init__(self, model, loss_fn, loaders, cfg, output_dir, tracker=None):
        self.model = model
        self.tracker = tracker or NullTracker()
        self.loss_fn = loss_fn
        self.loaders = loaders
        self.cfg = cfg
        self.labels = list(cfg.data.labels)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        t = cfg.train
        self.epochs = t.epochs
        self.grad_accum = t.grad_accum
        self.max_grad_norm = t.max_grad_norm
        self.patience = t.early_stop_patience
        self.use_amp = t.amp and self.device.type == "cuda"
        self.select_metric = t.get("select_metric", "macro_f1")
        self.select_exclude = list(t.get("select_exclude_labels") or [])

        self.optimizer = torch.optim.AdamW(model.param_groups(t.lr_encoder, t.lr_head, t.weight_decay))
        steps = math.ceil(len(loaders["train"]) / self.grad_accum) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(steps * t.warmup_ratio), steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.start_epoch = 0
        self.best_f1 = -1.0
        self.bad_epochs = 0
        self.history = []

        resume_path = resolve_resume_path(t.get("resume_from"))
        if resume_path:
            state = load_checkpoint(resume_path, model, self.optimizer, self.scheduler, self.scaler,
                                    map_location=self.device)
            self.start_epoch = state["epoch"] + 1
            self.best_f1 = state["best_f1"]
            self.bad_epochs = state.get("bad_epochs", 0)
            self.history = state.get("history", [])
            logger.info("resume từ %s (epoch %d, best macro-F1 %.4f)", resume_path, state["epoch"], self.best_f1)

    def train_epoch(self, epoch):
        self.model.train()
        total, n = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        loader = self.loaders["train"]
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{self.epochs}", leave=False)
        part_sums = {}
        uses_outputs = getattr(self.loss_fn, "uses_outputs", False)
        for step, batch in enumerate(pbar):
            batch = move_to_device(batch, self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                out = self.model.forward_outputs(batch) if uses_outputs else self.model(batch)
            if uses_outputs:
                loss, parts = self.loss_fn(out, batch)
                for k, v in parts.items():
                    part_sums[k] = part_sums.get(k, 0.0) + v * len(batch["labels"])
            else:
                loss = self.loss_fn(out.float(), batch["labels"])
            self.scaler.scale(loss / self.grad_accum).backward()

            if (step + 1) % self.grad_accum == 0 or step + 1 == len(loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            total += loss.item() * len(batch["labels"])
            n += len(batch["labels"])
            pbar.set_postfix(loss=f"{total / n:.4f}")
        self.last_loss_parts = {k: v / max(n, 1) for k, v in part_sums.items()} if len(part_sums) > 1 else {}
        if len(part_sums) > 1:  # chỉ log khi có loss phụ
            logger.info("epoch %d | loss thành phần %s", epoch + 1,
                        " ".join(f"{k}={v / max(n, 1):.4f}" for k, v in part_sums.items()))
        return total / max(n, 1)

    def predict(self, loader):
        return predict(self.model, loader, self.device, self.use_amp)

    def evaluate(self, split="val", prior_shift=None):
        """prior_shift: dict từ heads.compute_prior_shift để đánh giá có hiệu chỉnh prior (tắt lại sau khi xong)."""
        if prior_shift is not None:
            self.model.set_prior_shift(prior_shift)
        try:
            out = self.predict(self.loaders[split])
        finally:
            if prior_shift is not None:
                self.model.set_prior_shift(None)
        return compute_metrics(out["labels"], out["preds"], self.labels), out

    def fit(self):
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_epoch(epoch)
            metrics, _ = self.evaluate("val")
            score = selection_score(metrics, self.select_metric, self.select_exclude)
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": metrics["macro_f1"],
                                 "val_select": score})
            logger.info("epoch %d | train loss %.4f | val chọn checkpoint (%s) %.4f | val %s", epoch + 1, train_loss,
                        self.select_metric + (f" bỏ {self.select_exclude}" if self.select_exclude else ""), score,
                        format_metrics(metrics))

            improved = score > self.best_f1
            if improved:
                self.best_f1 = score
                self.best_val_metrics = metrics
                self.bad_epochs = 0
                save_checkpoint(self.output_dir / "best.pt", self.model, epoch=epoch, best_f1=self.best_f1,
                                config=self.cfg.to_dict())
                (self.output_dir / "best_val_metrics.json").write_text(json.dumps(metrics, indent=2))
            else:
                self.bad_epochs += 1

            self.tracker.log({"epoch": epoch + 1, "train/loss": train_loss,
                              **{f"train/loss_{k}": v for k, v in self.last_loss_parts.items()},
                              "train/lr_head": self.optimizer.param_groups[0]["lr"],
                              **metric_scalars(metrics, "val"), "val/select": score, "val/best_select": self.best_f1},
                             step=epoch + 1)

            save_checkpoint(self.output_dir / "last.pt", self.model, self.optimizer, self.scheduler, self.scaler,
                            epoch=epoch, best_f1=self.best_f1, bad_epochs=self.bad_epochs,
                            history=self.history, config=self.cfg.to_dict())

            if self.patience and self.bad_epochs >= self.patience:
                logger.info("early stopping: val không cải thiện sau %d epoch", self.patience)
                break
        peak_vram_gb = None
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        best = getattr(self, "best_val_metrics", None)
        if best is None and (self.output_dir / "best_val_metrics.json").exists():  # resume: best từ lần chạy trước
            best = json.loads((self.output_dir / "best_val_metrics.json").read_text())
        return {"best_val_macro_f1": best["macro_f1"] if best else self.best_f1, "best_val_select": self.best_f1,
                "train_minutes": (time.time() - start) / 60,
                "peak_vram_gb": peak_vram_gb, "history": self.history}

    def load_best(self):
        load_checkpoint(self.output_dir / "best.pt", self.model, map_location=self.device)
