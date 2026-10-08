import os
from pathlib import Path
import subprocess
import sys

import pytest

from facebench.tasks.head_pose.config import (
    CONFIG_ROOT,
    HEAD_POSE_ROOT,
    load_config,
    public_config,
    resolve_path,
)


def test_dime_config_inherits_protocol():
    config = load_config(CONFIG_ROOT / "dime.yaml")
    assert config["protocol"]["epochs"] == 80
    assert config["protocol"]["effective_batch_size"] == 64
    assert config["protocol"]["optimizer"]["name"] == "adamw"
    assert config["protocol"]["scheduler"]["name"] == "cosine"
    assert config["model"]["pretrained_checkpoint"] == ""


def test_repvgg_config_uses_synced_root_checkpoint():
    config = load_config(CONFIG_ROOT / "repvgg_b1g2.yaml")
    assert config["model"]["pretrained_checkpoint"] == "RepVGG-B1g2-train.pth"
    assert len(config["model"]["expected_pretrained_sha256"]) == 64
    assert config["wandb"]["api_key"] == ""
    assert not config["wandb"]["enabled"]


def test_public_config_redacts_embedded_wandb_key():
    config = {"wandb": {"api_key": "private-test-key"}, "value": 1}
    output = public_config(config)
    assert output["wandb"]["api_key"] == "<redacted>"
    assert config["wandb"]["api_key"] == "private-test-key"


def test_embedded_wandb_key_configures_auth_environment(monkeypatch):
    from facebench.tasks.head_pose.tracking import _wandb_settings

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    options = {
        "wandb": {
            "enabled": True,
            "entity": "entity",
            "project": "project",
            "mode": "online",
            "api_key": "private-test-key",
        }
    }
    assert _wandb_settings(options) == (True, "entity", "project", "online")
    assert os.environ["WANDB_API_KEY"] == "private-test-key"


def test_resolve_path_expands_environment(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "dime.pth"
    checkpoint.touch()
    monkeypatch.setenv("DIME_CHECKPOINT", str(checkpoint))
    assert resolve_path("${DIME_CHECKPOINT}", must_exist=True) == checkpoint.resolve()
    monkeypatch.delenv("DIME_CHECKPOINT")
    with pytest.raises(ValueError, match="Unresolved environment"):
        resolve_path("${DIME_CHECKPOINT}", must_exist=True)


def test_rocm_runtime_uses_rank_local_miopen_cache(monkeypatch, tmp_path: Path):
    import torch

    from facebench.tasks.head_pose.utils import setup_runtime

    monkeypatch.setattr(torch.version, "hip", "6.3.4")
    monkeypatch.setenv("LOCAL_RANK", "6")
    monkeypatch.setenv("DIME_MIOPEN_CACHE_ROOT", str(tmp_path))
    monkeypatch.delenv("MIOPEN_USER_DB_PATH", raising=False)
    monkeypatch.delenv("MIOPEN_CUSTOM_CACHE_DIR", raising=False)
    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
    monkeypatch.delenv("MIOPEN_FIND_ENFORCE", raising=False)

    setup_runtime()

    assert Path(os.environ["MIOPEN_USER_DB_PATH"]) == tmp_path / "rank_006" / "user-db"
    assert (
        Path(os.environ["MIOPEN_CUSTOM_CACHE_DIR"])
        == tmp_path / "rank_006" / "kernel-cache"
    )
    assert os.environ["MIOPEN_FIND_MODE"] == "FAST"
    assert os.environ["MIOPEN_FIND_ENFORCE"] == "NONE"
    assert torch.backends.cudnn.benchmark is False


def test_portable_submission_uses_one_four_gpu_allocation():
    script = Path(__file__).resolve().parents[4] / "scripts/submit_slurm.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--account",
            "test",
            "--partition",
            "gpu",
            "--nodes",
            "1",
            "--dry-run",
            "--",
            "--data",
            "data/pretrain.lmdb",
            "--gpus",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.count("sbatch") == 1
    assert "--nodes=1" in result.stdout
    assert "--gres=gpu:4" in result.stdout
    assert "run_pretrain.py" in result.stdout
