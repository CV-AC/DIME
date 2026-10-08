from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def setup_runtime() -> None:
    torch.backends.cudnn.benchmark = torch.version.hip is None
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass


def ddp_options(config: dict[str, Any]) -> dict[str, Any]:
    options = config["runtime"].get("ddp", {})
    return {
        "find_unused_parameters": bool(options.get("find_unused_parameters", False)),
        "broadcast_buffers": bool(options.get("broadcast_buffers", True)),
        "gradient_as_bucket_view": bool(options.get("gradient_as_bucket_view", True)),
        "static_graph": bool(options.get("static_graph", True)),
        "bucket_cap_mb": int(options.get("bucket_cap_mb", 50)),
    }


def init_distributed() -> tuple[int, int, int, torch.device]:

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires ROCm/CUDA PyTorch.")
        visible_devices = torch.cuda.device_count()
        if visible_devices == 1:

            device_index = 0
        elif local_rank < visible_devices:

            device_index = local_rank
        else:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {visible_devices} GPU(s) "
                "are visible to this process."
            )
        torch.cuda.set_device(device_index)
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        timeout = int(os.environ.get("DIME_PARSING_DIST_TIMEOUT_SECONDS", "7200"))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=timeout),
        )
        device = torch.device("cuda", device_index)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def broadcast_module(module: torch.nn.Module, source: int = 0) -> None:

    if not dist.is_initialized():
        return
    for tensor in module.state_dict().values():
        dist.broadcast(tensor, src=source)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def autocast_context(enabled: bool, dtype_name: str):
    if not enabled or not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name.lower() == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def git_commit(path: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def runtime_versions() -> dict[str, str | None]:
    try:
        import timm

        timm_version = timm.__version__
    except ImportError:
        timm_version = "not-installed"
    return {
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_cuda": torch.version.cuda,
        "timm": timm_version,
        "numpy": np.__version__,
    }


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def gather_objects(value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    output: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(output, value)
    return output
