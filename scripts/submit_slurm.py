import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_pretrain import get_args_parser, read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--time", default="36:00:00")
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--gres")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--job-name", default="dime_pretrain")
    parser.add_argument("--setup", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    training_args = args.training_args
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    if not training_args:
        parser.error("provide run_pretrain.py arguments after --")
    training = get_args_parser().parse_args(training_args)
    _, config = read_config(training)
    for outer, inner, name in [
        (args.nodes, training.nodes, "nodes"),
        (args.gpus_per_node, training.gpus, "gpus-per-node"),
        (args.master_port, training.master_port, "master-port"),
    ]:
        if outer is not None and inner is not None and outer != inner:
            parser.error(f"conflicting --{name} values")
    nodes = (
        args.nodes
        if args.nodes is not None
        else (config.distributed.nodes if training.nodes is None else training.nodes)
    )
    gpus = (
        args.gpus_per_node
        if args.gpus_per_node is not None
        else (
            config.distributed.gpus_per_node if training.gpus is None else training.gpus
        )
    )
    if min(nodes, gpus, args.cpus) < 1:
        parser.error("nodes, gpus-per-node, and cpus must be positive")
    if training.node_rank is not None:
        parser.error("Slurm assigns node ranks; do not pass --node-rank")
    gres = args.gres or f"gpu:{gpus}"
    if gres.rsplit(":", 1)[-1] != str(gpus):
        parser.error("--gres GPU count must match --gpus-per-node")
    port = args.master_port if args.master_port is not None else training.master_port
    if port is not None and not 1 <= port <= 65535:
        parser.error("--master-port must be within [1, 65535]")
    executable = "python" if args.setup else sys.executable
    run = [
        executable,
        str(ROOT / "scripts/run_pretrain.py"),
        "--nodes",
        str(nodes),
        "--gpus",
        str(gpus),
        *training_args,
    ]
    step = [
        "srun",
        f"--nodes={nodes}",
        f"--ntasks={nodes}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={args.cpus}",
        "--cpu-bind=none",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        *run,
    ]
    port_value = (
        str(port)
        if port is not None
        else ("${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}")
    )
    wrapped = "set -eu\n" + (args.setup + "\n" if args.setup else "")
    wrapped += f'export MASTER_PORT="{port_value}"\n' + shlex.join(step)
    command = [
        "sbatch",
        f"--account={args.account}",
        f"--partition={args.partition}",
        f"--nodes={nodes}",
        f"--ntasks={nodes}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={args.cpus}",
        f"--gres={gres}",
        f"--time={args.time}",
        f"--job-name={args.job_name}",
        f"--chdir={ROOT}",
        "--output=slurm-%x-%j.out",
        "--error=slurm-%x-%j.err",
        "--wrap",
        wrapped,
    ]
    if args.dry_run:
        print(shlex.join(command))
    else:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
