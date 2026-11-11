from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def config_fingerprint(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_torch_save(payload: object, path: str | Path) -> None:
    """Durably replace a checkpoint after its complete temporary copy exists."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        # Persist the directory entry on POSIX/NFS. Windows does not support
        # opening directories this way; the fully fsynced file plus atomic
        # replace remains the strongest portable contract there.
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            directory_fd = os.open(destination.parent, directory_flags)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_dump(payload: object, path: str | Path) -> None:
    """Atomically publish a JSON completion/result marker."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={project_root.resolve().as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_output(config: dict, project_root: Path) -> Path:
    output = Path(config["experiment"]["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    info = {
        "experiment": config["experiment"]["name"],
        "config_fingerprint": config_fingerprint(config),
        "source_revision": git_revision(project_root),
        "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME"),
        "pid": os.getpid(),
    }
    with (output / "run_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    return output
