import ast
import io
import os
from pathlib import Path
import re
import subprocess
import tokenize


CACHE_DIRECTORIES = {".git", "__pycache__", ".pytest_cache"}
ARTIFACT_DIRECTORIES = {"outputs", "results", "logs", "weights", ".trash"}
ARTIFACT_SUFFIXES = {
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".lmdb",
    ".pyc",
    ".out",
    ".err",
    ".csv",
    ".jsonl",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".zip",
    ".gz",
    ".pdf",
}
IMAGE_SUFFIXES = {".png", ".webp", ".gif", ".jpg", ".jpeg"}
PRIVATE_PATH = re.compile(
    r"/(?:scratch|flash|projappl)/project_[0-9]+|/(?:home|Users)/[^/\s\"']+/"
)
CREDENTIAL = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"
    r"|sk-[A-Za-z0-9_-]{32,}|AKIA[A-Z0-9]{16})\b"
)
CONVERSATION = re.compile(
    r"^\s*(?:USER|ASSISTANT|SYSTEM|用户|助手):|<\|(?:im_start|im_end)\|>", re.M
)


def release_files(root):
    physical = set()
    for directory, children, names in os.walk(root, followlinks=False):
        children[:] = [name for name in children if name not in CACHE_DIRECTORIES]
        for name in children:
            path = Path(directory) / name
            if path.is_symlink():
                physical.add(path.relative_to(root))
        for name in names:
            physical.add((Path(directory) / name).relative_to(root))
    violations = []
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        )
        tracked = {Path(name.decode()) for name in result.stdout.split(b"\0") if name}
        violations.extend(
            f"{name}: untracked release file" for name in sorted(physical - tracked)
        )
        violations.extend(
            f"{name}: missing tracked file" for name in sorted(tracked - physical)
        )
    return [root / name for name in sorted(physical)], violations


def check_python(source):
    violations = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    if any(token.type == tokenize.COMMENT for token in tokens):
        violations.append("Python comment")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and ast.get_docstring(node):
            violations.append("docstring")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [
                target.id.upper() for target in targets if isinstance(target, ast.Name)
            ]
            if (
                any(
                    name in {"WANDB_API_KEY", "API_KEY", "PASSWORD", "ACCESS_TOKEN"}
                    for name in names
                )
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and node.value.value
            ):
                violations.append("embedded credential")
    return violations


def main():
    root = Path(__file__).resolve().parents[1]
    files, violations = release_files(root)
    python_files = 0
    for path in files:
        relative = path.relative_to(root)
        if path.is_symlink():
            violations.append(f"{relative}: symbolic link")
            continue
        if path.suffix.lower() in ARTIFACT_SUFFIXES:
            violations.append(f"{relative}: data or runtime artifact")
            continue
        if ARTIFACT_DIRECTORIES.intersection(relative.parts):
            violations.append(f"{relative}: runtime directory")
        if (
            "lapa_to_wflw" in relative.parts
            or path.name.endswith(("_route_a.yaml", "_recommended.yaml"))
            or path.name.startswith("dime_frozen_")
            or path.name == "icr_encoder.py"
        ):
            violations.append(f"{relative}: historical experiment configuration")
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if relative.parts[0] != "assets":
                violations.append(f"{relative}: image outside assets")
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            violations.append(f"{relative}: unreadable or unexpected binary file")
            continue
        for pattern, label in [
            (PRIVATE_PATH, "private machine path"),
            (CREDENTIAL, "credential"),
            (CONVERSATION, "conversation record"),
        ]:
            if pattern.search(source):
                violations.append(f"{relative}: {label}")
        if path.suffix == ".py":
            python_files += 1
            violations.extend(
                f"{relative}: {reason}" for reason in check_python(source)
            )
        if path.suffix in {".yaml", ".yml", ".toml"}:
            if re.search(r"^\s*#", source, flags=re.M):
                violations.append(f"{relative}: configuration comment")
            for match in re.finditer(r"^\s*api_key:\s*(.*)$", source, flags=re.M):
                if match.group(1).strip() not in {"", "''", '""', "null"}:
                    violations.append(f"{relative}: configured credential")
    if violations:
        raise SystemExit("\n".join(violations))
    print(
        f"Checked {len(files)} release files, including {python_files} Python files: PASS"
    )


if __name__ == "__main__":
    main()
