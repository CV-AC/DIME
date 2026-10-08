import os
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.distributed.run import get_args_parser as torchrun_parser

from DIME_VIT.config import load_config
from DIME_VIT.train import setup_distributed
from scripts import run_pretrain, submit_slurm


@pytest.fixture(autouse=True)
def clear_distributed_environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SLURM_") or key in {
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        }:
            monkeypatch.delenv(key)


@pytest.mark.parametrize(
    "name", ["vit_small_patch16", "vit_base_patch16", "vit_large_patch16", "smoke_4gpu"]
)
def test_all_pretraining_configs_default_to_eight_nodes_and_gpus(name):
    config = load_config(run_pretrain.ROOT / "DIME_VIT/configs" / f"{name}.yaml")
    assert config.distributed.nodes == 8
    assert config.distributed.gpus_per_node == 8
    assert config.loss.edds_version == "v2"
    assert config.data.pair_sampling == "identity_uniform"


@pytest.mark.parametrize("field", ["nodes", "gpus_per_node"])
@pytest.mark.parametrize("value", ["0", "-1", "true", "1.5"])
def test_config_rejects_invalid_resource_counts(field, value):
    with pytest.raises(ValueError, match=f"distributed.{field}"):
        load_config(
            run_pretrain.ROOT / "DIME_VIT/configs/vit_base_patch16.yaml",
            [f"distributed.{field}={value}"],
        )


def test_slurm_dry_run_launches_one_controller_per_node(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit_slurm.py",
            "--account",
            "example",
            "--partition",
            "gpu",
            "--dry-run",
            "--",
            "--data",
            "/path/to/pretrain.lmdb",
        ],
    )
    with patch.object(submit_slurm.subprocess, "run") as execute:
        submit_slurm.main()
        execute.assert_not_called()
    command = shlex.split(capsys.readouterr().out)
    assert command[0] == "sbatch"
    assert "--nodes=8" in command
    assert "--ntasks=8" in command
    assert "--ntasks-per-node=1" in command
    assert "--gres=gpu:8" in command
    wrapped = command[command.index("--wrap") + 1]
    step = shlex.split(wrapped.splitlines()[-1])
    assert step[0] == "srun"
    assert "--nodes=8" in step
    assert "--ntasks=8" in step
    assert "--ntasks-per-node=1" in step
    assert step[step.index("--gpus") + 1] == "8"
    assert "export MASTER_PORT=" in wrapped


@pytest.mark.parametrize("rank", range(8))
def test_each_manual_node_launches_eight_workers(monkeypatch, capsys, rank):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pretrain.py",
            "--data",
            "/path/to/pretrain.lmdb",
            "--node-rank",
            str(rank),
            "--master-addr",
            "node0.example",
            "--master-port",
            "29601",
            "--dry-run",
        ],
    )
    with patch.object(run_pretrain.subprocess, "run") as execute:
        run_pretrain.main()
        execute.assert_not_called()
    command = shlex.split(capsys.readouterr().out)
    options = torchrun_parser().parse_args(command[3:])
    assert options.nnodes == "8"
    assert options.nproc_per_node == "8"
    assert options.node_rank == rank
    assert options.master_addr == "node0.example"
    assert options.master_port == 29601
    assert options.rdzv_backend == "static"
    assert options.module and options.training_script == "DIME_VIT.train"
    assert "distributed.nodes=8" in command
    assert "distributed.gpus_per_node=8" in command


def test_slurm_rank_and_master_are_resolved_from_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_NNODES", "8")
    monkeypatch.setenv("SLURM_NODEID", "7")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[0-7]")
    monkeypatch.setenv("MASTER_PORT", "29602")
    parser = run_pretrain.get_args_parser()
    args = parser.parse_args(["--data", "/path/to/pretrain.lmdb"])
    _, config = run_pretrain.read_config(args)
    with patch.object(
        run_pretrain.subprocess,
        "run",
        return_value=SimpleNamespace(stdout="node0\nnode1\n"),
    ) as execute:
        nodes, gpus, options = run_pretrain.distributed_options(args, config, parser)
    execute.assert_called_once_with(
        ["scontrol", "show", "hostnames", "node[0-7]"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (nodes, gpus) == (8, 8)
    assert "--node-rank=7" in options
    assert "--master-addr=node0" in options
    assert "--master-port=29602" in options


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--node-rank", "0"],
        ["--node-rank", "8", "--master-addr", "node0"],
        ["--node-rank", "0", "--master-addr", "node0", "--master-port", "0"],
        ["--nodes", "0"],
        ["--nodes", "1", "--node-rank", "1"],
    ],
)
def test_invalid_or_incomplete_launch_is_rejected(options):
    parser = run_pretrain.get_args_parser()
    args = parser.parse_args(["--data", "/path/to/pretrain.lmdb", *options])
    _, config = run_pretrain.read_config(args)
    with pytest.raises(SystemExit):
        run_pretrain.distributed_options(args, config, parser)


def test_global_rank_uses_local_device_without_initializing_real_ddp(monkeypatch):
    monkeypatch.setenv("RANK", "63")
    monkeypatch.setenv("WORLD_SIZE", "64")
    monkeypatch.setenv("LOCAL_RANK", "7")
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.set_device") as set_device,
        patch("torch.distributed.init_process_group") as initialize,
        patch("torch.distributed.barrier") as barrier,
    ):
        result = setup_distributed("nccl")
    assert result == (63, 64, 7, torch.device("cuda", 7))
    set_device.assert_called_once_with(7)
    initialize.assert_called_once_with(
        backend="nccl", init_method="env://", device_id=torch.device("cuda", 7)
    )
    barrier.assert_called_once_with()
