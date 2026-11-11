"""Safe, provenance-preserving import of the complete fbponlyms anchor.

This module deliberately has no dependency on the fbponlyms source tree.  A
checkpoint is treated as user-supplied data: it is opened read-only, reduced to
its tensor model state, and published into a Line3-owned directory.  Training
code can then load all 596 CASTR model tensors from that independent copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn


# The byte-preserving import/key-normalization algorithm is unchanged from v1;
# complete-vs-local semantics are enforced by the strict loader below rather
# than by invalidating already published import receipts.
MAPPING_VERSION = "fbponlyms-to-line3-v1"
MANIFEST_SCHEMA_VERSION = 1
ARCHIVED_ANCHOR_TENSOR_COUNT = 596
ANCHOR_PREFIXES = (
    "local_encoder.",
    "decoder.",
    "local_head.",
    "context_encoder.",
    "context_projection.",
    "global_head.",
    "structure_input.",
    "structure_blocks.",
    "context_fusion.",
    "structure_fusion.",
    "context_refine.",
    "structure_refine.",
    "context_head.",
    "structure_head.",
    "router.",
)
_STATE_CONTAINER_KEYS = ("model", "state_dict", "model_state_dict", "network", "net")
_PER_KEY_WRAPPERS = ("module.", "_orig_mod.")
_UNIFORM_ROOT_WRAPPERS = ("model.", "network.", "net.", "state_dict.")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CheckpointCompatibilityError(RuntimeError):
    """Raised before mutation when an anchor cannot be loaded strictly."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 of *path* without opening it for writing."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_expected_hash(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not _HASH_PATTERN.fullmatch(normalized):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    return normalized


def _load_checkpoint(path: Path) -> object:
    """Load on CPU while avoiding a second in-memory copy when PyTorch supports mmap."""

    try:
        # PyTorch 2.1 accepts mmap only when ``f`` is a string filename, while
        # newer releases also accept pathlib.Path.  Always pass str so the
        # authenticated loader behaves identically in the server environment.
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older supported torch builds
        return torch.load(str(path), map_location="cpu", weights_only=False)


def extract_tensor_state(checkpoint: object) -> tuple[OrderedDict[str, torch.Tensor], str]:
    """Extract a tensor-only state dict from common checkpoint containers.

    The returned tensors are not cloned.  Importing preserves their values and
    dtypes exactly; ``torch.save`` writes them into an independent file.
    """

    candidates: list[tuple[str, object]] = []
    if isinstance(checkpoint, Mapping):
        for key in _STATE_CONTAINER_KEYS:
            if key in checkpoint:
                candidates.append((key, checkpoint[key]))
        candidates.append(("<root>", checkpoint))

    for container, candidate in candidates:
        if not isinstance(candidate, Mapping) or not candidate:
            continue
        if not all(isinstance(key, str) for key in candidate):
            continue
        if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
            continue
        return OrderedDict(candidate.items()), container

    raise ValueError(
        "checkpoint does not contain a non-empty tensor state under model, state_dict, "
        "model_state_dict, network, net, or the checkpoint root"
    )


def normalize_state_dict_keys(
    state: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Remove serialization wrappers while preserving architectural key names.

    DDP/compiled wrappers are stripped per key.  Ambiguous roots such as
    ``model.`` are stripped only when every key shares the prefix, so a genuine
    nested module (for example ``local_encoder.model``) is never altered.
    Collisions are rejected rather than silently choosing one tensor.
    """

    if not state:
        raise ValueError("model state must not be empty")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError("model state must map string keys to tensors")

    stripped_counts: dict[str, int] = {}
    prepared: list[tuple[str, str, torch.Tensor]] = []
    for source_key, tensor in state.items():
        key = source_key
        changed = True
        while changed:
            changed = False
            for prefix in _PER_KEY_WRAPPERS:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    stripped_counts[prefix] = stripped_counts.get(prefix, 0) + 1
                    changed = True
                    break
        if not key:
            raise ValueError(f"wrapper removal produced an empty key from {source_key!r}")
        prepared.append((source_key, key, tensor))

    uniform_prefixes: list[str] = []
    while prepared:
        prefix = next(
            (
                candidate
                for candidate in _UNIFORM_ROOT_WRAPPERS
                if all(key.startswith(candidate) for _, key, _ in prepared)
            ),
            None,
        )
        if prefix is None:
            break
        prepared = [
            (source_key, key[len(prefix) :], tensor)
            for source_key, key, tensor in prepared
        ]
        uniform_prefixes.append(prefix)
        stripped_counts[prefix] = stripped_counts.get(prefix, 0) + len(prepared)

    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    origins: dict[str, str] = {}
    collisions: list[dict[str, str]] = []
    renamed = 0
    for source_key, destination_key, tensor in prepared:
        if not destination_key:
            raise ValueError(f"wrapper removal produced an empty key from {source_key!r}")
        if destination_key in normalized:
            collisions.append(
                {
                    "destination_key": destination_key,
                    "first_source_key": origins[destination_key],
                    "second_source_key": source_key,
                }
            )
            continue
        normalized[destination_key] = tensor
        origins[destination_key] = source_key
        renamed += int(destination_key != source_key)

    mapping_report = {
        "version": MAPPING_VERSION,
        "source_tensor_count": len(state),
        "mapped_tensor_count": len(normalized),
        "renamed_tensor_count": renamed,
        "stripped_prefix_counts": dict(sorted(stripped_counts.items())),
        "uniform_root_prefixes": uniform_prefixes,
        "collisions": collisions,
    }
    if collisions:
        raise CheckpointCompatibilityError(
            "state-dict key normalization produced collisions", mapping_report
        )
    return normalized, mapping_report


def compare_state_dicts(
    candidate: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Report strict name/shape compatibility without modifying a model."""

    candidate_keys = set(candidate)
    target_keys = set(target)
    common = candidate_keys & target_keys
    shape_mismatches = []
    dtype_mismatches = []
    for key in sorted(common):
        source_tensor = candidate[key]
        target_tensor = target[key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
        elif source_tensor.dtype != target_tensor.dtype:
            dtype_mismatches.append(
                {
                    "key": key,
                    "source_dtype": str(source_tensor.dtype),
                    "target_dtype": str(target_tensor.dtype),
                }
            )

    missing = sorted(target_keys - candidate_keys)
    unexpected = sorted(candidate_keys - target_keys)
    compatible = not missing and not unexpected and not shape_mismatches
    return {
        "compatible": compatible,
        "candidate_tensor_count": len(candidate),
        "target_tensor_count": len(target),
        "matched_tensor_count": len(common) - len(shape_mismatches),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        # Loading floating weights of a different dtype is supported by
        # PyTorch, but provenance still records the conversion risk.
        "dtype_mismatches": dtype_mismatches,
    }


def _anchor_subset(state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key, value) for key, value in state.items() if key.startswith(ANCHOR_PREFIXES)
    )


def _prefix_counts(state: Mapping[str, torch.Tensor]) -> dict[str, int]:
    return {
        prefix: sum(key.startswith(prefix) for key in state)
        for prefix in ANCHOR_PREFIXES
    }


def load_fbponlyms_local_anchor(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_source_sha256: str | None = None,
    require_archived_tensor_count: bool = True,
) -> dict[str, Any]:
    """Strictly load the complete CASTR graph into an MS75 model.

    The historical function name is retained for API compatibility.  Every
    model tensor in the authenticated source must belong to one of the exact
    CASTR prefixes, and every target tensor under those prefixes must exist
    with an identical shape.  Validation finishes before model mutation.
    """

    source = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint is not a file: {source}")
    expected = _validated_expected_hash(expected_sha256)
    expected_original = _validated_expected_hash(expected_source_sha256)
    source_hash = sha256_file(source)
    if expected is not None and source_hash != expected:
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected {expected}, observed {source_hash}"
        )

    import_manifest = None
    if expected_original is not None:
        manifest_path = source.with_suffix(source.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "source SHA256 pinning requires the imported-anchor manifest: "
                f"{manifest_path}"
            )
        try:
            import_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid imported-anchor manifest: {manifest_path}") from error
        if not isinstance(import_manifest, Mapping):
            raise ValueError("imported-anchor manifest must be a JSON object")
        manifest_source = import_manifest.get("source")
        manifest_destination = import_manifest.get("destination")
        if not isinstance(manifest_source, Mapping) or not isinstance(
            manifest_destination, Mapping
        ):
            raise ValueError(
                "imported-anchor manifest must contain source and destination objects"
            )
        recorded_source_hash = str(manifest_source.get("sha256", "")).lower()
        recorded_destination_hash = str(
            manifest_destination.get("sha256", "")
        ).lower()
        if recorded_source_hash != expected_original:
            raise ValueError(
                "imported-anchor source SHA256 mismatch: expected "
                f"{expected_original}, manifest records {recorded_source_hash or '<missing>'}"
            )
        if recorded_destination_hash != source_hash:
            raise ValueError(
                "imported-anchor destination SHA256 mismatch: checkpoint is "
                f"{source_hash}, manifest records {recorded_destination_hash or '<missing>'}"
            )
        recorded_name = Path(
            str(manifest_destination.get("checkpoint_path", ""))
        ).name
        if recorded_name != source.name:
            raise ValueError(
                "imported-anchor manifest destination basename does not match checkpoint"
            )

    checkpoint = _load_checkpoint(source)
    if sha256_file(source) != source_hash:
        raise RuntimeError("source checkpoint changed while it was being loaded")
    raw_state, container = extract_tensor_state(checkpoint)
    source_state, mapping = normalize_state_dict_keys(raw_state)
    source_anchor = _anchor_subset(source_state)

    target_model = model
    target_state = target_model.state_dict()
    if not _anchor_subset(target_state) and hasattr(model, "module"):
        possible = getattr(model, "module")
        if isinstance(possible, nn.Module) and _anchor_subset(possible.state_dict()):
            target_model = possible
            target_state = target_model.state_dict()
    target_anchor = _anchor_subset(target_state)

    source_prefix_counts = _prefix_counts(source_anchor)
    target_prefix_counts = _prefix_counts(target_anchor)
    missing_source_modules = [
        prefix for prefix, count in source_prefix_counts.items() if count == 0
    ]
    missing_target_modules = [
        prefix for prefix, count in target_prefix_counts.items() if count == 0
    ]
    compatibility = compare_state_dicts(source_anchor, target_anchor)
    ignored_source_keys = sorted(set(source_state) - set(source_anchor))
    report: dict[str, Any] = {
        "loaded": False,
        "source_path": str(source),
        "source_sha256": source_hash,
        "original_source_sha256": (
            str(import_manifest["source"]["sha256"]).lower()
            if import_manifest is not None
            else source_hash
        ),
        "import_manifest_path": (
            str(source.with_suffix(source.suffix + ".manifest.json"))
            if import_manifest is not None
            else None
        ),
        "source_container": container,
        "mapping": mapping,
        "anchor_prefixes": list(ANCHOR_PREFIXES),
        "source_prefix_counts": source_prefix_counts,
        "target_prefix_counts": target_prefix_counts,
        "missing_source_modules": missing_source_modules,
        "missing_target_modules": missing_target_modules,
        "compatibility": compatibility,
        "archived_anchor_tensor_count": ARCHIVED_ANCHOR_TENSOR_COUNT,
        "archived_tensor_count_required": bool(require_archived_tensor_count),
        "expected_anchor_tensor_count": (
            ARCHIVED_ANCHOR_TENSOR_COUNT
            if require_archived_tensor_count
            else len(target_anchor)
        ),
        "source_anchor_tensor_count": len(source_anchor),
        "target_anchor_tensor_count": len(target_anchor),
        "loaded_anchor_tensor_count": 0,
        "complete_anchor": False,
        "ignored_source_tensor_count": len(ignored_source_keys),
        "ignored_source_keys": ignored_source_keys,
    }
    archived_count_mismatch = bool(require_archived_tensor_count) and (
        len(source_anchor) != ARCHIVED_ANCHOR_TENSOR_COUNT
        or len(target_anchor) != ARCHIVED_ANCHOR_TENSOR_COUNT
    )
    report["archived_tensor_count_mismatch"] = archived_count_mismatch
    if (
        archived_count_mismatch
        or
        missing_source_modules
        or missing_target_modules
        or ignored_source_keys
        or compatibility["dtype_mismatches"]
        or not compatibility["compatible"]
    ):
        raise CheckpointCompatibilityError(
            "fbponlyms complete CASTR anchor is not strictly compatible with the target model",
            report,
        )

    # Load a complete target state rather than relying on the meaning of
    # ``missing_keys`` from a partial ``strict=False`` load.  Timm modules may
    # register load hooks for generated Swin buffers; those hooks can suppress
    # or add missing-key diagnostics even when the explicitly compared source
    # and target anchor dictionaries are identical.  Merging the audited
    # anchor into the model's own complete state lets PyTorch apply every hook
    # under ``strict=True`` while all non-anchor tensors are supplied with
    # their current values and therefore remain unchanged.
    merged_state = OrderedDict(target_state)
    merged_state.update(source_anchor)
    incompatible = target_model.load_state_dict(merged_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        report["pytorch_missing_keys"] = sorted(incompatible.missing_keys)
        report["pytorch_unexpected_keys"] = sorted(incompatible.unexpected_keys)
        raise RuntimeError("PyTorch reported an inconsistency after strict anchor validation")

    # A load hook is allowed to consume or regenerate derived buffers, but the
    # resulting reusable state must still equal the authenticated source after
    # the normal PyTorch dtype/device conversion.  This postcondition protects
    # every trainable anchor weight instead of trusting diagnostics alone.
    loaded_anchor = _anchor_subset(target_model.state_dict())
    post_load_mismatches: list[str] = []
    for key, source_value in source_anchor.items():
        loaded_value = loaded_anchor[key]
        expected_value = source_value.to(
            device=loaded_value.device, dtype=loaded_value.dtype
        )
        if not torch.equal(loaded_value, expected_value):
            post_load_mismatches.append(key)
    if post_load_mismatches:
        report["post_load_mismatches"] = post_load_mismatches
        raise RuntimeError("anchor tensors changed or were not loaded exactly")
    report["loaded"] = True
    report["loaded_tensor_count"] = len(source_anchor)
    report["loaded_anchor_tensor_count"] = len(source_anchor)
    report["complete_anchor"] = True
    return report


def load_fbponlyms_complete_anchor(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_source_sha256: str | None = None,
    require_archived_tensor_count: bool = True,
) -> dict[str, Any]:
    """Explicit complete-anchor alias for the legacy loader name."""

    return load_fbponlyms_local_anchor(
        model,
        checkpoint_path,
        expected_sha256=expected_sha256,
        expected_source_sha256=expected_source_sha256,
        require_archived_tensor_count=require_archived_tensor_count,
    )


def audit_ms75_initialization_gates(model: nn.Module) -> dict[str, Any]:
    """Require the complete MS75 model to remain on its exact CASTR path."""

    context_scale = getattr(model, "context_residual_scale", None)
    hierarchy_base = getattr(model, "hierarchy_mix_base", None)
    hierarchy_gate = getattr(model, "hierarchy_mix_gate", None)
    if not isinstance(context_scale, torch.Tensor):
        raise ValueError("MS75 model does not expose context_residual_scale")
    if not isinstance(hierarchy_base, torch.Tensor):
        raise ValueError("MS75 model does not expose hierarchy_mix_base")
    if not isinstance(hierarchy_gate, nn.Module):
        raise ValueError("MS75 model does not expose hierarchy_mix_gate")
    gate_state = hierarchy_gate.state_dict()
    if not gate_state:
        raise ValueError("MS75 hierarchy_mix_gate has no state")

    values = {
        "context_residual_scale": float(context_scale.detach().cpu()),
        "hierarchy_mix_base": float(hierarchy_base.detach().cpu()),
        "hierarchy_mix_gate_max_abs": max(
            float(value.detach().abs().max().cpu())
            for value in gate_state.values()
        ),
    }
    nonzero = [name for name, value in values.items() if value != 0.0]
    if nonzero:
        raise CheckpointCompatibilityError(
            "initialization-only MS75 checkpoint requires exact zero "
            "context/hierarchy gates",
            {"gate_values": values, "nonzero_gates": nonzero},
        )
    return values


def materialize_ms75_initialization(
    model: nn.Module,
    anchor_path: str | Path,
    output_path: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
    config_fingerprint_value: str,
    expected_anchor_sha256: str | None = None,
    expected_anchor_source_sha256: str | None = None,
    require_archived_tensor_count: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create a complete, directly evaluable P0 checkpoint inside Line3.

    The imported anchor itself must already be owned by the Line3 project.
    Only the in-memory target model is modified; neither the imported copy nor
    the original fbponlyms project is opened for writing.
    """

    root = Path(project_root).expanduser().resolve(strict=True)
    anchor = Path(anchor_path).expanduser().resolve(strict=True)
    if not anchor.is_file():
        raise FileNotFoundError(f"anchor checkpoint is not a file: {anchor}")
    if not anchor.is_relative_to(root):
        raise ValueError("P0 materialization requires a Line3-owned imported anchor")
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve(strict=False)
    if not output.is_relative_to(root) or output == root:
        raise ValueError("P0 output must remain inside the Line3 project")
    if output == anchor:
        raise ValueError("P0 output must not overwrite the imported anchor")
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("P0 output must end in .pt or .pth")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        existing = output if output.exists() else manifest_path
        raise FileExistsError(
            f"refusing to overwrite existing P0 artifact: {existing}"
        )

    devices = {
        value.device.type
        for value in list(model.parameters()) + list(model.buffers())
    }
    if devices - {"cpu"}:
        raise ValueError("P0 materialization must run on CPU")
    anchor_report = load_fbponlyms_local_anchor(
        model,
        anchor,
        expected_sha256=expected_anchor_sha256,
        expected_source_sha256=expected_anchor_source_sha256,
        require_archived_tensor_count=require_archived_tensor_count,
    )
    gate_values = audit_ms75_initialization_gates(model)
    model.eval()
    model_state = model.state_dict()
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    config_absolute = str(Path(config_path).expanduser().resolve(strict=True))
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "ms75_initialization_only_p0",
        "created_at_utc": created_at,
        "dry_run": bool(dry_run),
        "project_root": str(root),
        "config_path": config_absolute,
        "config_fingerprint": str(config_fingerprint_value),
        "anchor": anchor_report,
        "zero_gate_audit": gate_values,
        "destination": {
            "checkpoint_path": str(output),
            "manifest_path": str(manifest_path),
            "sha256": None,
            "tensor_count": len(model_state),
        },
    }
    if dry_run:
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("P0 output directory resolves outside the Line3 project")
    provenance = {
        "artifact_kind": "ms75_initialization_only_p0",
        "created_at_utc": created_at,
        "config_path": config_absolute,
        "config_fingerprint": str(config_fingerprint_value),
        "anchor_path": str(anchor),
        "anchor_file_sha256": anchor_report["source_sha256"],
        "anchor_original_source_sha256": anchor_report[
            "original_source_sha256"
        ],
        "anchor_loaded_tensor_count": anchor_report["loaded_tensor_count"],
        "mapping_version": MAPPING_VERSION,
        "zero_gate_audit": gate_values,
    }
    payload = {
        "model": model_state,
        "checkpoint_kind": "ms75_initialization_only_p0",
        "config_fingerprint": str(config_fingerprint_value),
        "ms75_initialization": provenance,
    }
    checkpoint_temporary = output.parent / (
        f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    manifest_temporary = output.parent / (
        f".{manifest_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    published_checkpoint = False
    try:
        torch.save(payload, checkpoint_temporary)
        with checkpoint_temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        output_hash = sha256_file(checkpoint_temporary)
        manifest["destination"]["sha256"] = output_hash
        manifest["dry_run"] = False
        _write_json_temporary(manifest, manifest_temporary)
        _publish_no_replace(checkpoint_temporary, output)
        published_checkpoint = True
        _publish_no_replace(manifest_temporary, manifest_path)
    except Exception:
        checkpoint_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
        if published_checkpoint:
            output.unlink(missing_ok=True)
        raise
    if sha256_file(output) != manifest["destination"]["sha256"]:
        raise RuntimeError("published P0 checkpoint failed SHA256 verification")
    return manifest


def _safe_project_output(
    output_dir: str | Path,
    project_root: str | Path | None,
) -> tuple[Path, Path]:
    root = (
        Path(project_root).expanduser().resolve(strict=False)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    output = Path(output_dir).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve(strict=False)
    if output == root or not output.is_relative_to(root):
        raise ValueError(
            f"output_dir must be a dedicated directory inside Line3 project {root}, got {output}"
        )
    return root, output


def _destination_name(source: Path, source_hash: str, requested: str | None) -> str:
    if requested is None:
        # The formal config points at the original checkpoint basename.  The
        # full hash remains mandatory in the manifest, while no-replace
        # publication prevents two different sources from sharing this name.
        return source.name
    if Path(requested).name != requested or requested in {".", ".."}:
        raise ValueError("destination_name must be a plain file name, not a path")
    suffix = Path(requested).suffix.lower()
    if suffix not in {".pt", ".pth"}:
        raise ValueError("destination_name must end in .pt or .pth")
    return requested


def _publish_no_replace(temporary: Path, destination: Path) -> None:
    """Atomically publish a complete file and refuse to replace any existing path."""

    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing import: {destination}") from None
    except OSError as error:
        raise OSError(
            f"could not atomically publish {destination} without overwrite; "
            "the output filesystem must support same-directory hard links"
        ) from error
    temporary.unlink()


def _write_json_temporary(payload: object, temporary: Path) -> None:
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def import_fbponlyms_checkpoint(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    expected_sha256: str | None = None,
    dry_run: bool = False,
    project_root: str | Path | None = None,
    destination_name: str | None = None,
    target_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Import a checkpoint into a Line3-owned directory without touching its source."""

    source = Path(source_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint is not a file: {source}")
    root, output = _safe_project_output(output_dir, project_root)
    if source == output or output in source.parents:
        raise ValueError("source checkpoint must not be the Line3 import directory")
    if source.is_relative_to(output):
        raise ValueError("refusing to re-import a checkpoint from its own output directory")

    expected = _validated_expected_hash(expected_sha256)
    source_stat = source.stat()
    source_hash = sha256_file(source)
    if expected is not None and source_hash != expected:
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected {expected}, observed {source_hash}"
        )

    checkpoint = _load_checkpoint(source)
    raw_state, container = extract_tensor_state(checkpoint)
    normalized_state, mapping = normalize_state_dict_keys(raw_state)
    # Detect a concurrently replaced/truncated source before publishing any
    # result.  Stat metadata is diagnostic; the content hash is authoritative.
    verified_hash = sha256_file(source)
    source_stat_after = source.stat()
    if source_hash != verified_hash:
        raise RuntimeError("source checkpoint changed while it was being imported")

    compatibility = None
    if target_state is not None:
        normalized_target, target_mapping = normalize_state_dict_keys(target_state)
        compatibility = compare_state_dicts(normalized_state, normalized_target)
        compatibility["target_mapping"] = target_mapping

    file_name = _destination_name(source, source_hash, destination_name)
    destination = output / file_name
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        existing = destination if destination.exists() else manifest_path
        raise FileExistsError(f"refusing to overwrite existing import: {existing}")

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "mapping_version": MAPPING_VERSION,
        "created_at_utc": created_at,
        "dry_run": bool(dry_run),
        "project_root": str(root),
        "source": {
            "absolute_path": str(source),
            "sha256": source_hash,
            "size_bytes": source_stat.st_size,
            "mtime_ns_before": source_stat.st_mtime_ns,
            "mtime_ns_after": source_stat_after.st_mtime_ns,
            "state_container": container,
            "tensor_count": len(raw_state),
        },
        "destination": {
            "checkpoint_path": str(destination),
            "manifest_path": str(manifest_path),
            "sha256": None,
            "tensor_count": len(normalized_state),
        },
        "mapping": mapping,
        "compatibility": compatibility,
        "preserved_checkpoint_fields": (
            sorted(str(key) for key in checkpoint.keys() if key != container)
            if isinstance(checkpoint, Mapping) and container != "<root>"
            else []
        ),
    }
    if dry_run:
        return manifest

    output.mkdir(parents=True, exist_ok=True)
    # Resolve again after creation so a pre-existing symlink cannot redirect
    # writes outside the Line3 project.
    resolved_output = output.resolve(strict=True)
    if not resolved_output.is_relative_to(root.resolve(strict=False)):
        raise ValueError("created output directory resolves outside the Line3 project")

    checkpoint_temporary = output / f".{file_name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    manifest_temporary = output / (
        f".{manifest_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    published_checkpoint = False
    try:
        # Preserve the exact source bytes.  The strict loader normalizes common
        # state-dict wrappers in memory, so re-serializing here adds no value and
        # makes a pre-registered source hash impossible to verify at launch.
        # A byte-preserving independent copy keeps source and destination hashes
        # equal while the manifest still records the key-mapping audit above.
        with source.open("rb") as source_handle, checkpoint_temporary.open(
            "xb"
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, 8 * 1024 * 1024)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        destination_hash = sha256_file(checkpoint_temporary)
        if destination_hash != source_hash:
            raise RuntimeError(
                "byte-preserving anchor import failed SHA256 verification"
            )
        manifest["destination"]["sha256"] = destination_hash
        manifest["dry_run"] = False
        _write_json_temporary(manifest, manifest_temporary)
        _publish_no_replace(checkpoint_temporary, destination)
        published_checkpoint = True
        _publish_no_replace(manifest_temporary, manifest_path)
    except Exception:
        checkpoint_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
        if published_checkpoint:
            destination.unlink(missing_ok=True)
        raise

    if sha256_file(destination) != manifest["destination"]["sha256"]:
        raise RuntimeError("published checkpoint failed its SHA256 verification")
    return manifest


__all__ = [
    "ARCHIVED_ANCHOR_TENSOR_COUNT",
    "ANCHOR_PREFIXES",
    "MAPPING_VERSION",
    "CheckpointCompatibilityError",
    "compare_state_dicts",
    "extract_tensor_state",
    "import_fbponlyms_checkpoint",
    "audit_ms75_initialization_gates",
    "load_fbponlyms_local_anchor",
    "load_fbponlyms_complete_anchor",
    "materialize_ms75_initialization",
    "normalize_state_dict_keys",
    "sha256_file",
]
