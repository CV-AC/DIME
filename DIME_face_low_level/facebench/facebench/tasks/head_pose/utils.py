from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
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


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def setup_runtime() -> None:
    is_rocm = torch.version.hip is not None
    if is_rocm:

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        cache_root = Path(
            os.environ.get(
                "DIME_MIOPEN_CACHE_ROOT",
                f"/tmp/dime-headpose-miopen-{os.environ.get('SLURM_JOB_ID', 'local')}",
            )
        )
        rank_root = cache_root / f"rank_{local_rank:03d}"
        user_db = rank_root / "user-db"
        kernel_cache = rank_root / "kernel-cache"
        user_db.mkdir(parents=True, exist_ok=True)
        kernel_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MIOPEN_USER_DB_PATH"] = str(user_db)
        os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(kernel_cache)
        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
        os.environ.setdefault("MIOPEN_FIND_ENFORCE", "NONE")

    torch.backends.cudnn.benchmark = not is_rocm
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass


def init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA/NCCL.")
        torch.cuda.set_device(local_rank)
        timeout_seconds = int(
            os.environ.get("DIME_HEADPOSE_DIST_TIMEOUT_SECONDS", "1800")
        )
        if timeout_seconds <= 0:
            raise ValueError("DIME_HEADPOSE_DIST_TIMEOUT_SECONDS must be positive.")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def autocast_context(enabled: bool, dtype_name: str):
    if not enabled or not torch.cuda.is_available():
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if dtype_name.lower() == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def git_commit(path: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def runtime_versions() -> dict[str, str | None]:
    try:
        import timm

        timm_version: str | None = timm.__version__
    except ImportError:
        timm_version = None
    return {
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "timm": timm_version,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
        "miopen_find_enforce": os.environ.get("MIOPEN_FIND_ENFORCE"),
        "cudnn_benchmark": str(torch.backends.cudnn.benchmark),
    }


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
