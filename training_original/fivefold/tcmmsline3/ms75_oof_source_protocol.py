from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .runtime import git_revision


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_oof_source_binding_config(train: Mapping[str, Any]) -> None:
    path = str(train.get("oof_source_protocol_manifest", "")).strip()
    expected = str(
        train.get("oof_source_protocol_manifest_expected_sha256", "")
    ).lower()
    if not path:
        raise ValueError("clean OOF training requires a source protocol manifest")
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(
            "clean OOF source protocol manifest expected SHA256 is invalid"
        )


def _require_clean_git_tree(project_root: Path) -> None:
    process = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={project_root.resolve().as_posix()}",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    dirty = [line for line in process.stdout.splitlines() if line]
    if dirty:
        raise ValueError(
            "clean OOF startup requires a completely clean git tree; "
            f"first_dirty_entries={dirty[:10]}"
        )


def load_oof_source_protocol_binding(
    config: Mapping[str, Any], *, project_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate a clean OOF source protocol against the live checkout."""

    root = Path(project_root).resolve(strict=True)
    _require_clean_git_tree(root)
    train = config["train"]
    validate_oof_source_binding_config(train)
    manifest_path = Path(str(train["oof_source_protocol_manifest"])).resolve(
        strict=True
    )
    file_sha = sha256_file(manifest_path)
    expected_file_sha = str(
        train["oof_source_protocol_manifest_expected_sha256"]
    ).lower()
    if file_sha != expected_file_sha:
        raise ValueError("clean OOF source protocol file SHA256 mismatch")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("clean OOF source protocol is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("clean OOF source protocol must be a mapping")
    role = str(payload.get("role", ""))
    stage = str(config["data"].get("protocol_stage", ""))
    architecture = str(config["model"].get("architecture", ""))
    if role == "ms75_clean_aligned_swin_oof":
        from .ms75_aligned_protocol import (
            current_software_dependencies,
            validate_aligned_source_protocol,
        )

        protocol = validate_aligned_source_protocol(payload)
        if stage != "oof_aligned_proxy" or architecture != (
            "ms75_oof_aligned_swin"
        ):
            raise ValueError("aligned OOF source protocol/config role mismatch")
        if protocol.get("software_dependencies") != (
            current_software_dependencies()
        ):
            raise ValueError("aligned OOF software dependency identity mismatch")
    elif role == "ms75_clean_neutral_oof":
        from .ms75_neutral_oof import (
            current_neutral_software_dependencies,
            validate_neutral_source_protocol,
        )

        protocol = validate_neutral_source_protocol(payload)
        if stage not in {"oof_neutral", "oof_neutral_proxy"}:
            raise ValueError("neutral OOF source protocol/config role mismatch")
        if protocol.get("software_dependencies") != (
            current_neutral_software_dependencies()
        ):
            raise ValueError("neutral OOF software dependency identity mismatch")
    else:
        raise ValueError(f"unsupported clean OOF source protocol role: {role!r}")

    revision = git_revision(root)
    if revision is None or revision != protocol["source_revision"]:
        raise ValueError("clean OOF source revision differs from the live checkout")
    config_path = root / str(protocol["config_path"])
    if sha256_file(config_path) != protocol["config_sha256"]:
        raise ValueError("clean OOF static config SHA256 mismatch")
    for relative, expected_sha in protocol["implementation_sha256"].items():
        path = (root / str(relative)).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("clean OOF implementation path escapes project root") from error
        if sha256_file(path) != expected_sha:
            raise ValueError(
                f"clean OOF implementation SHA256 mismatch: {relative}"
            )
    binding = {
        "path": str(manifest_path),
        "file_sha256": file_sha,
        "source_protocol_sha256": protocol["source_protocol_sha256"],
        "source_revision": protocol["source_revision"],
        "role": role,
    }
    return dict(protocol), binding


def validate_checkpoint_oof_source_protocol_binding(
    checkpoint: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    observed = checkpoint.get("oof_source_protocol_binding")
    exact_keys = {
        "path",
        "file_sha256",
        "source_protocol_sha256",
        "source_revision",
        "role",
    }
    if (
        not isinstance(observed, Mapping)
        or set(observed) != exact_keys
        or dict(observed) != dict(expected)
    ):
        raise RuntimeError(
            "clean OOF checkpoint source protocol binding mismatch"
        )


__all__ = [
    "load_oof_source_protocol_binding",
    "sha256_file",
    "validate_checkpoint_oof_source_protocol_binding",
    "validate_oof_source_binding_config",
]
