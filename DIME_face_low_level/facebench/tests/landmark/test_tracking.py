import math
import sys
from types import SimpleNamespace

from facebench.tasks.landmark.tracking import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    HeatmapCollapseMonitor,
    WandbTracker,
    epoch_log_payload,
    format_epoch_line,
    new_run_identity,
)


def _row():
    return {
        "epoch": 4,
        "train": {
            "loss": 0.0189,
            "coordinate_loss": 0.0167,
            "heatmap_loss": 0.0022,
        },
        "validation": None,
        "official_test": {
            "nme_interocular": 4.21,
            "nme_interpupil_diagnostic": 5.8,
            "fr_0.10": 2.0,
            "auc_0.10": 61.5,
            "subsets": {
                "largepose": {"nme_interocular": 7.1},
                "blur": {"nme_interocular": 4.8},
            },
        },
        "encoder_lr": 1e-4,
        "head_lr": 1e-2,
        "seconds": 71.2,
    }


def test_disabled_tracker_requires_no_credentials(tmp_path):
    tracker = WandbTracker.start(
        method="dime",
        backbone="dime",
        stage="benchmark",
        output_dir=tmp_path,
        config={"wandb": {"enabled": False}},
        metadata={},
        resume_checkpoint="",
        entity="",
        project="",
        api_key="",
    )
    tracker.log({"train/loss": 1.0}, step=1)
    tracker.finish()
    assert tracker.run_id == ""


def test_wandb_identity_is_unique_and_readable(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    first_id, first_name = new_run_identity("farl_ep64", "benchmark")
    second_id, second_name = new_run_identity("farl_ep64", "benchmark")
    assert first_id != second_id
    assert first_name != second_name
    assert "farl_ep64-benchmark-12345" in first_name


def test_epoch_payload_contains_losses_and_all_test_metrics():
    payload = epoch_log_payload(_row(), best_epoch=4, best_nme=4.21, is_best=True)
    assert payload["train/loss"] == 0.0189
    assert payload["test/nme_interocular"] == 4.21
    assert payload["test/subsets/largepose/nme_interocular"] == 7.1
    assert payload["selection/is_best"] == 1
    assert payload["best/epoch"] == 4


def test_epoch_console_output_is_exactly_one_concise_line():
    line = format_epoch_line(
        _row(), total_epochs=150, best_epoch=4, best_nme=4.21, is_best=True
    )
    assert "\n" not in line
    assert line.startswith("[004/150] | ")
    assert "train loss=0.01890 (coord=0.01670, heat=0.00220)" in line
    assert "TEST NME=4.210 FR10=2.00 AUC10=61.50" in line
    assert "best NME=4.210@004 NEW_BEST" in line
    assert "largepose" not in line
    assert line.endswith("71.2s")


def test_payload_omits_non_finite_uninitialized_best():
    payload = epoch_log_payload(_row(), best_epoch=0, best_nme=math.inf, is_best=False)
    assert "best/epoch" not in payload
    assert "best/nme_interocular" not in payload


def test_route_a_collapse_monitor_honors_warmup_and_patience():
    monitor = HeatmapCollapseMonitor.from_objective(
        {
            "name": "route_a",
            "collapse_detection": {
                "enabled": True,
                "warmup_epochs": 2,
                "patience": 2,
                "fraction_threshold": 0.95,
            },
        }
    )
    collapsed = {"heatmap_collapsed_fraction": 1.0}
    assert monitor.update(epoch=1, train_metrics=collapsed) is None
    assert monitor.update(epoch=2, train_metrics=collapsed) is None
    assert monitor.update(epoch=3, train_metrics=collapsed) is None
    message = monitor.update(epoch=4, train_metrics=collapsed)
    assert message is not None
    assert "collapsed" in message


def test_route_a_diagnostics_reach_console_and_wandb():
    row = _row()
    row["train"].update(
        {
            "heatmap_max": 0.82,
            "heatmap_std": 0.014,
            "heatmap_peak_response": 0.73,
            "heatmap_collapsed_fraction": 0.0,
        }
    )
    row["collapse"] = {"consecutive_epochs": 0, "triggered": False}
    payload = epoch_log_payload(row, best_epoch=4, best_nme=4.21, is_best=True)
    assert payload["diagnostics/heatmap_max"] == 0.82
    assert payload["diagnostics/heatmap_std"] == 0.014
    assert payload["diagnostics/heatmap_peak_response"] == 0.73
    assert payload["diagnostics/heatmap_collapsed_fraction"] == 0.0
    assert payload["collapse/triggered"] == 0
    line = format_epoch_line(
        row, total_epochs=150, best_epoch=4, best_nme=4.21, is_best=True
    )
    assert "heatmap max=0.8200 std=0.014000 GT=0.7300 flat=0.0%" in line


def test_tracker_starts_unique_online_run_and_logs_epoch(monkeypatch, tmp_path):
    calls = {}

    class FakeRun:
        url = "https://wandb.invalid/run"

        def __init__(self):
            self.summary = {}
            self.metrics = []
            self.finished = None

        def define_metric(self, *args, **kwargs):
            return None

        def log(self, payload, step):
            self.metrics.append((payload, step))

        def finish(self, exit_code=0):
            self.finished = exit_code

    fake_run = FakeRun()

    def fake_init(**kwargs):
        calls.update(kwargs)
        return fake_run

    fake_wandb = SimpleNamespace(
        init=fake_init,
        Settings=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    tracker = WandbTracker.start(
        method="farl_ep64",
        backbone="farl",
        stage="benchmark",
        output_dir=tmp_path,
        config={"protocol": {"epochs": 150}},
        metadata={"git_commit": "test"},
        resume_checkpoint="",
        entity="test-entity",
        project=WANDB_PROJECT,
        api_key="test-api-key",
    )
    assert calls["entity"] == "test-entity"
    assert calls["project"] == WANDB_PROJECT
    assert calls["mode"] == "online"
    assert calls["resume"] == "never"
    assert calls["force"] is True
    assert calls["settings"]["console"] == "wrap"
    assert calls["id"] == tracker.run_id
    assert calls["name"] == tracker.name

    tracker.log_epoch(_row(), best_epoch=4, best_nme=4.21, is_best=True)
    assert fake_run.metrics[0][1] == 4
    assert fake_run.metrics[0][0]["train/loss"] == 0.0189
    tracker.finish()
    assert fake_run.finished == 0
