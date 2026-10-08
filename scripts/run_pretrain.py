import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small", "base", "large"], default="base")
    parser.add_argument("--data", required=True)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--gpus", "--gpus-per-node", dest="gpus", type=int)
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return parser


def read_config(args):
    from DIME_VIT.config import load_config

    path = args.config or ROOT / "DIME_VIT/configs" / (
        "smoke_4gpu.yaml" if args.verify_resume else f"vit_{args.model}_patch16.yaml"
    )
    return path.resolve(), load_config(path, args.opts)


def distributed_options(args, config, parser):
    nodes = config.distributed.nodes if args.nodes is None else args.nodes
    gpus = config.distributed.gpus_per_node if args.gpus is None else args.gpus
    if nodes < 1 or gpus < 1:
        parser.error("--nodes and --gpus must be positive")
    if os.environ.get("SLURM_NNODES") and int(os.environ["SLURM_NNODES"]) != nodes:
        parser.error("--nodes does not match the Slurm allocation")
    if nodes == 1:
        if args.node_rank not in (None, 0):
            parser.error("a single-node launch requires node rank 0")
        return nodes, gpus, ["--standalone"]

    rank = args.node_rank
    if rank is None:
        if "SLURM_NODEID" not in os.environ:
            parser.error("multi-node launches require --node-rank or SLURM_NODEID")
        rank = int(os.environ["SLURM_NODEID"])
    if not 0 <= rank < nodes:
        parser.error("--node-rank must be within [0, nodes)")

    address = args.master_addr or os.environ.get("MASTER_ADDR")
    if not address and os.environ.get("SLURM_JOB_NODELIST"):
        result = subprocess.run(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            check=True,
            capture_output=True,
            text=True,
        )
        hosts = result.stdout.splitlines()
        address = hosts[0].strip() if hosts else None
    if not address:
        parser.error("multi-node launches require --master-addr or a Slurm node list")
    port = args.master_port
    if port is None:
        port = int(os.environ.get("MASTER_PORT", "29500"))
    if not 1 <= port <= 65535:
        parser.error("--master-port must be within [1, 65535]")
    return (
        nodes,
        gpus,
        [
            f"--nnodes={nodes}",
            f"--node-rank={rank}",
            "--rdzv-backend=static",
            f"--master-addr={address}",
            f"--master-port={port}",
        ],
    )


def main():
    sys.path.insert(0, str(ROOT))
    parser = get_args_parser()
    args = parser.parse_args()
    config_path, config = read_config(args)
    nodes, gpus, launch_options = distributed_options(args, config, parser)
    data = Path(args.data).expanduser().resolve()
    if not args.dry_run and not data.exists():
        parser.error(f"LMDB does not exist: {data}")
    default_output = (
        "smoke_4gpu" if args.verify_resume else f"dime_vit_{args.model}_patch16"
    )
    output = (args.output or ROOT / "outputs" / default_output).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("DIME_DISABLE_CUDNN_SDPA", "1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *launch_options,
        f"--nproc-per-node={gpus}",
        "--max-restarts=0",
        "-m",
        "DIME_VIT.train",
        "--config",
        str(config_path),
        "--opts",
        f"data.path={data}",
        f"checkpoint.output_dir={output}",
        "checkpoint.resume=auto",
        *args.opts,
        f"distributed.nodes={nodes}",
        f"distributed.gpus_per_node={gpus}",
    ]
    for stop_epoch in [2, 3] if args.verify_resume else [None]:
        stage_command = (
            command
            if stop_epoch is None
            else command + [f"train.stop_epoch={stop_epoch}"]
        )
        if args.dry_run:
            print(shlex.join(stage_command))
        else:
            subprocess.run(stage_command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
