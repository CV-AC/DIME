from types import SimpleNamespace

import pytest
import torch

from facebench.tasks.landmark.optim import build_optimizer, build_scheduler


def _model():
    return SimpleNamespace(
        backbone=torch.nn.Linear(2, 2),
        pyramid=torch.nn.Linear(2, 2),
        head=torch.nn.Linear(2, 2),
    )


def _protocol():
    return {
        "encoder_lr": 1e-4,
        "head_lr": 1e-2,
        "optimizer": {
            "name": "adamw",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-5,
            "amsgrad": False,
        },
        "scheduler": {
            "name": "multistep",
            "warmup_epochs": 0,
            "warmup_start_factor": 0.01,
            "milestones": [200],
            "gamma": 0.1,
            "min_lr_factor": 0.0,
        },
    }


def test_base_optimizer_is_adamw_with_two_learning_rates():
    optimizer = build_optimizer(_model(), _protocol())
    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["name"] for group in optimizer.param_groups] == ["encoder", "head"]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-2]
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["weight_decay"] == 1e-5


def test_default_scheduler_preserves_original_multistep_behavior():
    optimizer = build_optimizer(_model(), _protocol())
    scheduler = build_scheduler(optimizer, _protocol(), total_epochs=150)
    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
    for _ in range(150):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-2]


def test_legacy_flat_optimizer_and_scheduler_keys_still_work():
    protocol = {
        "encoder_lr": 1e-4,
        "head_lr": 1e-2,
        "betas": [0.8, 0.95],
        "weight_decay": 2e-5,
        "lr_milestones": [2],
        "lr_gamma": 0.5,
    }
    optimizer = build_optimizer(_model(), protocol)
    scheduler = build_scheduler(optimizer, protocol, total_epochs=4)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["betas"] == (0.8, 0.95)
    assert optimizer.defaults["weight_decay"] == 2e-5
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [5e-5, 5e-3]
    )


def test_cosine_scheduler_applies_warmup_to_both_parameter_groups():
    protocol = _protocol()
    protocol["scheduler"] = {
        "name": "cosine",
        "warmup_epochs": 2,
        "warmup_start_factor": 0.1,
        "min_lr_factor": 0.01,
    }
    optimizer = build_optimizer(_model(), protocol)
    scheduler = build_scheduler(optimizer, protocol, total_epochs=6)

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-5, 1e-3]
    )
    optimizer.step()
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [5.5e-5, 5.5e-3]
    )
    optimizer.step()
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-4, 1e-2]
    )

    for _ in range(3):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-6, 1e-4]
    )


def test_cosine_optimizer_and_scheduler_state_resume_exactly():
    protocol = _protocol()
    protocol["scheduler"] = {
        "name": "cosine",
        "warmup_epochs": 2,
        "warmup_start_factor": 0.1,
        "min_lr_factor": 0.01,
    }
    first_optimizer = build_optimizer(_model(), protocol)
    first_scheduler = build_scheduler(first_optimizer, protocol, total_epochs=10)
    for _ in range(4):
        first_optimizer.step()
        first_scheduler.step()

    second_optimizer = build_optimizer(_model(), protocol)
    second_scheduler = build_scheduler(second_optimizer, protocol, total_epochs=10)
    second_optimizer.load_state_dict(first_optimizer.state_dict())
    second_scheduler.load_state_dict(first_scheduler.state_dict())
    assert [group["lr"] for group in second_optimizer.param_groups] == pytest.approx(
        [group["lr"] for group in first_optimizer.param_groups]
    )

    first_optimizer.step()
    first_scheduler.step()
    second_optimizer.step()
    second_scheduler.step()
    assert [group["lr"] for group in second_optimizer.param_groups] == pytest.approx(
        [group["lr"] for group in first_optimizer.param_groups]
    )


def test_optimizer_and_scheduler_names_are_validated():
    protocol = _protocol()
    protocol["optimizer"]["name"] = "invalid"
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer(_model(), protocol)

    protocol = _protocol()
    optimizer = build_optimizer(_model(), protocol)
    protocol["scheduler"]["name"] = "invalid"
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        build_scheduler(optimizer, protocol, total_epochs=150)
