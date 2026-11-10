"""Test tracking wandb: dùng wandb giả, không cần mạng hay tài khoản."""
import runpy
import sys
import types

import pytest
import torch

from test_cot_distill import _tiny_classifier
from vimmsd.utils.config import load_config
from vimmsd.utils.tracking import NullTracker, WandbTracker, metric_scalars, run_names, start_tracking


class FakeRun:
    id = "abc123"

    def __init__(self):
        self.logs, self.summary, self.finished = [], {}, False

    def log(self, data, step=None):
        self.logs.append((step, data))

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    module = types.ModuleType("wandb")
    module.run, module.kwargs = FakeRun(), None

    def init(**kwargs):
        module.kwargs = kwargs
        return module.run

    module.init = init
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def _cfg(*overrides):
    return load_config("configs/base.yaml", list(overrides), env="local")


def test_disabled_by_default_does_not_touch_wandb(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import wandb sẽ lỗi nếu bị gọi
    assert isinstance(start_tracking(_cfg(), "c", tmp_path), NullTracker)
    assert not (tmp_path / "wandb_run_id.txt").exists()


def test_missing_wandb_or_init_error_keeps_training(monkeypatch, fake_wandb, tmp_path):
    def fail(**kwargs):
        raise RuntimeError("no network")

    fake_wandb.init = fail
    assert isinstance(start_tracking(_cfg("wandb.enabled=true"), "c", tmp_path), NullTracker)
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert isinstance(start_tracking(_cfg("wandb.enabled=true"), "c", tmp_path), NullTracker)


def test_log_error_does_not_raise():
    run = FakeRun()
    run.log = lambda *a, **k: 1 / 0
    WandbTracker(run).log({"a": 1}, step=1)


def test_run_names_group_seeds():
    assert run_names(_cfg("name=hier_a_s7", "seed=7")) == ("hier_a_s7", "hier_a")
    assert run_names(_cfg("name=hier_a", "seed=7")) == ("hier_a_s7", "hier_a")


def test_resume_reuses_run_id_only_when_resuming(fake_wandb, tmp_path):
    start_tracking(_cfg("wandb.enabled=true", "name=x", "seed=1"), "abc", tmp_path)
    kwargs = fake_wandb.kwargs
    assert kwargs["id"] is None and kwargs["resume"] is None
    assert (kwargs["name"], kwargs["group"]) == ("x_s1", "x") and kwargs["config"]["commit"] == "abc"
    assert (tmp_path / "wandb_run_id.txt").read_text() == "abc123"
    start_tracking(_cfg("wandb.enabled=true", "train.resume_from=last.pt"), "abc", tmp_path)
    assert fake_wandb.kwargs["id"] == "abc123" and fake_wandb.kwargs["resume"] == "allow"


def test_metric_scalars_skip_confusion_matrix():
    scalars = metric_scalars({"macro_f1": 0.5, "confusion_matrix": [[1]], "per_class": {"multi": {"f1": 0.7}}}, "val")
    assert scalars == {"val/macro_f1": 0.5, "val/f1_multi": 0.7}


def test_trainer_logs_each_epoch_with_loss_parts(monkeypatch, tmp_path):
    from vimmsd.training.losses import build_training_loss
    from vimmsd.training.trainer import Trainer

    torch.manual_seed(0)
    cfg = load_config("configs/distill_hier_b.yaml", ["train.epochs=2", "train.amp=false", "train.early_stop_patience=0",
                                                       f"paths.local.output_dir={tmp_path}"], env="local")
    model = _tiny_classifier(monkeypatch, dict(cfg.model.head))

    def batch(n):
        return {"ids": [str(i) for i in range(n)], "input_ids": torch.randint(5, 50, (n, 6)),
                "attention_mask": torch.ones(n, 6, dtype=torch.long), "labels": torch.arange(n) % 4}

    run = FakeRun()
    loaders = {"train": [batch(8)], "val": [batch(8)]}
    Trainer(model, build_training_loss(cfg, [10, 3, 4, 8]), loaders, cfg, tmp_path, tracker=WandbTracker(run)).fit()
    assert [step for step, _ in run.logs] == [1, 2]
    data = run.logs[0][1]
    assert data["epoch"] == 1 and {"train/loss", "train/lr_head", "val/macro_f1", "val/select", "val/best_select"} <= set(data)
    assert "val/f1_multi-sarcasm" in data and any(k.startswith("train/loss_") for k in data)


def test_run_experiments_wandb_flag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--plan", "configs/experiments/smoke.yaml", "--rerun",
                                      "--dry-run", "--wandb"])
    monkeypatch.chdir(".")
    runpy.run_path("scripts/run_experiments.py", run_name="__main__")
    assert "wandb.enabled=true" in capsys.readouterr().out
