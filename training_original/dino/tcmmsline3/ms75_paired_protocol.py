from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .ms75_checkpoint import sha256_file
from .ms75_data import (
    BalancedPointIndex,
    canonical_oof_provenance_json,
    validate_oof_provenance_manifest,
)
from .ms75_neutral_oof import (
    NEUTRAL_OOF_ROLE,
    validate_neutral_source_protocol,
)
from .ms75_paired_decision import PairedDecisionThresholds
from .ms75_protocol import MS75_OFFICIAL_TEST_SCENE_IDS


SCHEMA_VERSION = 1
NUM_CLASSES = 24
EXPECTED_BASE_CHECKPOINT_SHA256 = (
    "dc52e3c98a9624cb5bfd9c40ef58402ef7e08cff0556a68e5641d782f77a6bf5"
)
FORMAL_BASE_SOURCE_REVISION = "09d11171141271868828cae6123c59031e3dace5"
_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ARM_SPECIFIC_PATHS = {
    "experiment.name",
    "experiment.output_dir",
    "train.paired_continuation_arm",
    "data.dynamic_sampling.class_probability_manifest",
    "data.dynamic_sampling.class_probability_manifest_sha256",
}
_SELF_REFERENCE_PATHS = {
    "train.paired_protocol_manifest": "<PAIR_PROTOCOL_MANIFEST>",
    "train.paired_protocol_manifest_expected_sha256": (
        "<PAIR_PROTOCOL_MANIFEST_FILE_SHA256>"
    ),
}


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _json_mapping(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _verify_embedded_identity(payload: Mapping[str, Any], *, name: str) -> str:
    supplied = _sha(
        payload.get("manifest_sha256"), name=f"{name}.manifest_sha256"
    )
    body = {
        key: value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if canonical_sha256(body) != supplied:
        raise ValueError(f"{name} embedded manifest SHA mismatch")
    return supplied


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result: dict[str, object] = {}
    for key in sorted(value):
        child = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(value[key], child))
    return result


def _arm_config_differences(
    arm_a0: Mapping[str, Any], arm_a1: Mapping[str, Any]
) -> dict[str, tuple[object, object]]:
    left = _flatten(arm_a0)
    right = _flatten(arm_a1)
    differences: dict[str, tuple[object, object]] = {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            differences[key] = (left.get(key), right.get(key))
    return differences


def normalized_common_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    replacements: dict[str, object] = {
        "experiment.name": "<ARM_NAME>",
        "experiment.output_dir": "<ARM_OUTPUT>",
        "train.paired_continuation_arm": "<ARM>",
        "data.dynamic_sampling.class_probability_manifest": (
            "<CLASS_PROBABILITY_MANIFEST>"
        ),
        "data.dynamic_sampling.class_probability_manifest_sha256": (
            "<CLASS_PROBABILITY_FILE_SHA256>"
        ),
        **_SELF_REFERENCE_PATHS,
    }
    for dotted, replacement in replacements.items():
        cursor: dict[str, Any] = result
        fields = dotted.split(".")
        for field in fields[:-1]:
            nested = cursor.get(field)
            if not isinstance(nested, dict):
                raise ValueError(f"paired config is missing {dotted}")
            cursor = nested
        if fields[-1] not in cursor:
            raise ValueError(f"paired config is missing {dotted}")
        cursor[fields[-1]] = replacement
    return result


def paired_evaluation_protocol(
    config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return only fields capable of changing whole-scene predictions.

    Training-only sampling/loader settings and the train-vs-evaluate role are
    deliberately removed, allowing the common formal evaluation config to be
    used for the Base, A0 and A1 checkpoints while binding every model,
    preprocessing, geometry, AMP and TTA choice.
    """

    data = copy.deepcopy(dict(config.get("data", {})))
    for key in ("dynamic_sampling", "samples_per_epoch", "num_workers"):
        data.pop(key, None)
    inference = copy.deepcopy(dict(config.get("inference", {})))
    inference.pop("protocol_role", None)
    inference.pop("evaluate_test_after_training", None)
    body: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "FBP",
        "num_classes": NUM_CLASSES,
        "data": data,
        "model": copy.deepcopy(dict(config.get("model", {}))),
        "inference": inference,
        "execution": {
            "device": str(config.get("train", {}).get("device", "")),
            "cudnn_benchmark": bool(
                config.get("train", {}).get("cudnn_benchmark", False)
            ),
        },
    }
    return {**body, "protocol_sha256": canonical_sha256(body)}


def _normalized_arm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for dotted, replacement in _SELF_REFERENCE_PATHS.items():
        cursor: dict[str, Any] = result
        fields = dotted.split(".")
        for field in fields[:-1]:
            nested = cursor.get(field)
            if not isinstance(nested, dict):
                raise ValueError(f"paired config is missing {dotted}")
            cursor = nested
        if fields[-1] not in cursor:
            raise ValueError(f"paired config is missing {dotted}")
        cursor[fields[-1]] = replacement
    return result


def _validate_probability_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_arm: str,
    parent_difficulty_sha256: str,
) -> tuple[dict[str, Any], str]:
    source = Path(path).resolve()
    expected_file_sha256 = _sha(
        expected_file_sha256, name=f"{expected_arm} probability file SHA"
    )
    observed_file_sha = sha256_file(source)
    if observed_file_sha != expected_file_sha256:
        raise ValueError(f"{expected_arm} probability file SHA mismatch")
    payload = _json_mapping(
        source, name=f"{expected_arm} probability manifest"
    )
    _verify_embedded_identity(
        payload, name=f"{expected_arm} probability manifest"
    )
    required = {
        "schema_version",
        "source_split",
        "arm",
        "class_ids",
        "probabilities",
        "lower_multiplier",
        "upper_multiplier",
        "parent_difficulty_sha256",
        "manifest_sha256",
    }
    if set(payload) != required:
        raise ValueError(
            f"{expected_arm} probability manifest schema mismatch"
        )
    if int(payload["schema_version"]) != 1:
        raise ValueError("probability manifest schema_version must be one")
    if payload["source_split"] != "official_train":
        raise ValueError("probability manifest must use official_train")
    if payload["arm"] != expected_arm:
        raise ValueError(f"probability manifest must belong to {expected_arm}")
    if payload["class_ids"] != list(range(NUM_CLASSES)):
        raise ValueError(
            "probability class_ids must be the ordered 24 classes"
        )
    if payload["parent_difficulty_sha256"] != parent_difficulty_sha256:
        raise ValueError(
            "probability manifest has the wrong difficulty parent"
        )
    probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
    if probabilities.shape != (NUM_CLASSES,) or not np.all(
        np.isfinite(probabilities) & (probabilities > 0.0)
    ):
        raise ValueError(
            "probabilities must contain 24 finite positive values"
        )
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    lower = float(payload["lower_multiplier"]) / NUM_CLASSES
    upper = float(payload["upper_multiplier"]) / NUM_CLASSES
    if np.any(probabilities < lower - 1e-12) or np.any(
        probabilities > upper + 1e-12
    ):
        raise ValueError("probabilities violate the declared bounds")
    uniform = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
    if expected_arm == "A0" and not np.allclose(
        probabilities, uniform, rtol=0.0, atol=1e-12
    ):
        raise ValueError("A0 probability manifest must be uniform")
    if expected_arm == "A1" and np.allclose(
        probabilities, uniform, rtol=0.0, atol=1e-12
    ):
        raise ValueError("A1 probability manifest must be non-uniform")
    return payload, observed_file_sha


def _config_probability(config: Mapping[str, Any]) -> tuple[str, str]:
    dynamic = config.get("data", {}).get("dynamic_sampling", {})
    if not isinstance(dynamic, Mapping):
        raise ValueError("data.dynamic_sampling must be a mapping")
    path = dynamic.get("class_probability_manifest")
    digest = dynamic.get("class_probability_manifest_sha256")
    if not isinstance(path, str) or not path:
        raise ValueError("class_probability_manifest is required")
    return path, _sha(digest, name="class_probability_manifest_sha256")


def _completion_artifact(
    value: object,
    *,
    name: str,
    work_root: Path,
    verify_file_sha256: bool,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256")
    relative = value.get("path")
    digest = _sha(value.get("sha256"), name=f"{name}.sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError(f"{name}.path must be a portable relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name}.path must stay below the neutral work root")
    resolved = work_root.joinpath(*pure.parts).resolve(strict=True)
    try:
        resolved.relative_to(work_root)
    except ValueError as error:
        raise ValueError(f"{name}.path escapes the neutral work root") from error
    if not resolved.is_file():
        raise ValueError(f"{name}.path is not a regular file")
    if verify_file_sha256 and sha256_file(resolved) != digest:
        raise ValueError(f"{name} file SHA mismatch")
    return {"path": str(resolved), "sha256": digest}


def validate_neutral_oof_completion_for_pair(
    completion_path: str | Path,
) -> dict[str, Any]:
    """Authenticate the sole confusion-index source accepted by the pair.

    The neutral terminal receipt uses work-root-relative artifact references.
    This function re-opens and hashes its source protocol, provenance and OOF
    archive, validates the clean-v2 fold proof, and returns an absolute,
    content-addressed contract suitable for embedding in the pair manifest.
    """

    completion_file = Path(completion_path).resolve(strict=True)
    if completion_file.name != "completed.json":
        raise ValueError("neutral OOF completion must be named completed.json")
    work_root = completion_file.parent
    completion = _json_mapping(completion_file, name="neutral OOF completion")
    expected_fields = {
        "schema_version",
        "status",
        "role",
        "source_split",
        "official_test_opened",
        "full120_checkpoint_opened",
        "source_protocol_sha256",
        "source_protocol",
        "provenance",
        "provenance_manifest_sha256",
        "oof_index",
        "fold_checkpoint_sha256",
        "folds",
    }
    if set(completion) != expected_fields:
        raise ValueError("neutral OOF completion schema is not exact")
    expected_identity = {
        "schema_version": 1,
        "status": "completed",
        "role": NEUTRAL_OOF_ROLE,
        "source_split": "official_train",
        "official_test_opened": False,
        "full120_checkpoint_opened": False,
    }
    for key, expected in expected_identity.items():
        if completion.get(key) != expected:
            raise ValueError(f"neutral OOF completion {key} is invalid")

    source_ref = _completion_artifact(
        completion["source_protocol"],
        name="neutral source_protocol",
        work_root=work_root,
        verify_file_sha256=True,
    )
    provenance_ref = _completion_artifact(
        completion["provenance"],
        name="neutral provenance",
        work_root=work_root,
        verify_file_sha256=True,
    )
    index_ref = _completion_artifact(
        completion["oof_index"],
        name="neutral OOF index",
        work_root=work_root,
        verify_file_sha256=True,
    )
    source_protocol = validate_neutral_source_protocol(
        _json_mapping(source_ref["path"], name="neutral source_protocol")
    )
    source_inner_sha = _sha(
        completion.get("source_protocol_sha256"),
        name="neutral completion source_protocol_sha256",
    )
    if source_protocol["source_protocol_sha256"] != source_inner_sha:
        raise ValueError("neutral completion source-protocol identity mismatch")

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root.joinpath(
        *PurePosixPath(source_protocol["config_path"]).parts
    )
    if sha256_file(config_path) != source_protocol["config_sha256"]:
        raise ValueError("neutral source-protocol config changed")
    for relative, expected_sha in source_protocol[
        "implementation_sha256"
    ].items():
        implementation = project_root.joinpath(*PurePosixPath(relative).parts)
        if sha256_file(implementation) != expected_sha:
            raise ValueError(f"neutral source implementation changed: {relative}")

    provenance = _json_mapping(
        provenance_ref["path"], name="neutral OOF provenance"
    )
    if (
        provenance.get("schema_version") != 2
        or provenance.get("source_split") != "official_train"
        or provenance.get("source_protocol") != source_protocol
    ):
        raise ValueError("neutral OOF provenance is not clean schema v2")
    folds = provenance.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("neutral OOF provenance must contain five folds")
    first = folds[0] if folds else None
    if not isinstance(first, Mapping):
        raise ValueError("neutral OOF provenance fold schema is invalid")
    train_universe = {
        str(value)
        for key in ("train_scene_ids", "heldout_scene_ids")
        for value in first.get(key, [])
    }
    if len(train_universe) != 120:
        raise ValueError("neutral OOF provenance does not cover 120 train scenes")
    validate_oof_provenance_manifest(
        provenance,
        official_train_scene_ids=sorted(train_universe),
        official_test_scene_ids=MS75_OFFICIAL_TEST_SCENE_IDS,
    )
    provenance_inner_sha = hashlib.sha256(
        canonical_oof_provenance_json(provenance).encode("utf-8")
    ).hexdigest()
    if provenance_inner_sha != _sha(
        completion.get("provenance_manifest_sha256"),
        name="neutral provenance_manifest_sha256",
    ):
        raise ValueError("neutral OOF provenance canonical SHA mismatch")
    loaded_index = BalancedPointIndex.from_confusion_npz(
        index_ref["path"],
        expected_sha256=index_ref["sha256"],
        active_scene_ids=sorted(train_universe),
        official_train_scene_ids=sorted(train_universe),
        official_test_scene_ids=MS75_OFFICIAL_TEST_SCENE_IDS,
    )
    if loaded_index.file_sha256 != index_ref["sha256"]:
        raise AssertionError("neutral OOF index identity changed during load")
    with np.load(index_ref["path"], allow_pickle=False) as archive:
        embedded_json = str(np.asarray(archive["provenance_json"]).item())
        embedded_sha = str(np.asarray(archive["provenance_sha256"]).item())
        embedded_scene_ids = [
            str(value) for value in np.asarray(archive["scene_ids"]).tolist()
        ]
        embedded_scene_folds = np.asarray(
            archive["scene_fold_ids"], dtype=np.int64
        )
    expected_scene_folds = {
        str(scene): int(fold["fold_id"])
        for fold in folds
        for scene in fold["heldout_scene_ids"]
    }
    if (
        embedded_json != canonical_oof_provenance_json(provenance)
        or embedded_sha != provenance_inner_sha
        or len(embedded_scene_ids) != 120
        or set(embedded_scene_ids) != train_universe
        or not np.array_equal(
            embedded_scene_folds,
            np.asarray(
                [expected_scene_folds[scene] for scene in embedded_scene_ids],
                dtype=np.int64,
            ),
        )
        or provenance["source_protocol"].get("role") != NEUTRAL_OOF_ROLE
    ):
        raise ValueError(
            "neutral OOF index embedded provenance differs from completion"
        )

    checkpoint_map = completion.get("fold_checkpoint_sha256")
    if not isinstance(checkpoint_map, Mapping) or set(checkpoint_map) != {
        str(fold) for fold in range(5)
    }:
        raise ValueError("neutral completion fold checkpoint map is not exact")
    completion_folds = completion.get("folds")
    if not isinstance(completion_folds, list) or len(completion_folds) != 5:
        raise ValueError("neutral completion must bind five fold artifacts")
    provenance_by_fold = {
        int(fold["fold_id"]): fold for fold in folds if isinstance(fold, Mapping)
    }
    completion_by_fold: dict[int, Mapping[str, Any]] = {}
    for fold in completion_folds:
        if not isinstance(fold, Mapping) or set(fold) != {
            "fold_id",
            "checkpoint",
            "training_completion",
            "prediction_completion",
        }:
            raise ValueError("neutral completion fold schema is not exact")
        fold_id = int(fold.get("fold_id", -1))
        if fold_id in completion_by_fold:
            raise ValueError("neutral completion has duplicate fold IDs")
        completion_by_fold[fold_id] = fold
    if set(provenance_by_fold) != set(range(5)) or set(completion_by_fold) != set(
        range(5)
    ):
        raise ValueError("neutral completion fold IDs must be exactly 0..4")
    for fold_id in range(5):
        fold = completion_by_fold[fold_id]
        checkpoint = _completion_artifact(
            fold["checkpoint"],
            name=f"neutral fold {fold_id} checkpoint",
            work_root=work_root,
            verify_file_sha256=True,
        )
        training_completion_ref = _completion_artifact(
            fold["training_completion"],
            name=f"neutral fold {fold_id} training completion",
            work_root=work_root,
            verify_file_sha256=True,
        )
        prediction_completion_ref = _completion_artifact(
            fold["prediction_completion"],
            name=f"neutral fold {fold_id} prediction completion",
            work_root=work_root,
            verify_file_sha256=True,
        )
        expected_checkpoint_sha = _sha(
            checkpoint_map[str(fold_id)],
            name=f"neutral fold {fold_id} checkpoint SHA",
        )
        if (
            checkpoint["sha256"] != expected_checkpoint_sha
            or provenance_by_fold[fold_id].get("checkpoint")
            != dict(fold["checkpoint"])
        ):
            raise ValueError(f"neutral fold {fold_id} checkpoint identity mismatch")
        training_completion = _json_mapping(
            training_completion_ref["path"],
            name=f"neutral fold {fold_id} training completion",
        )
        if (
            training_completion.get("status") != "completed"
            or training_completion.get("selection_metric") != "last_epoch"
            or int(training_completion.get("epochs", -1)) != 10
        ):
            raise ValueError(
                f"neutral fold {fold_id} training completion is invalid"
            )
        prediction_completion = _json_mapping(
            prediction_completion_ref["path"],
            name=f"neutral fold {fold_id} prediction completion",
        )
        prediction_binding = prediction_completion.get(
            "oof_source_protocol_binding"
        )
        if (
            prediction_completion.get("status") != "completed"
            or prediction_completion.get("fold_id") != fold_id
            or prediction_completion.get("proxy_role") != NEUTRAL_OOF_ROLE
            or prediction_completion.get("checkpoint_sha256")
            != expected_checkpoint_sha
            or not isinstance(prediction_binding, Mapping)
            or prediction_binding.get("source_protocol_sha256")
            != source_inner_sha
        ):
            raise ValueError(
                f"neutral fold {fold_id} prediction completion is invalid"
            )

    builder_path = "tools/build_ms75_oof_confusion_index.py"
    builder_sha = source_protocol["implementation_sha256"].get(builder_path)
    _sha(builder_sha, name="neutral OOF builder implementation SHA")
    return {
        "schema_version": 1,
        "role": "ms75_clean_neutral_oof_confusion_index",
        "source_revision": source_protocol["source_revision"],
        "neutral_completion_path": str(completion_file),
        "neutral_completion_file_sha256": sha256_file(completion_file),
        "source_protocol_path": source_ref["path"],
        "source_protocol_file_sha256": source_ref["sha256"],
        "source_protocol_sha256": source_inner_sha,
        "provenance_path": provenance_ref["path"],
        "provenance_file_sha256": provenance_ref["sha256"],
        "provenance_manifest_sha256": provenance_inner_sha,
        "index_path": index_ref["path"],
        "index_file_sha256": index_ref["sha256"],
        "builder_path": builder_path,
        "builder_implementation_sha256": builder_sha,
    }


def _config_index(
    config: Mapping[str, Any], *, kind: str
) -> tuple[Path, str]:
    dynamic = config.get("data", {}).get("dynamic_sampling", {})
    if not isinstance(dynamic, Mapping):
        raise ValueError("data.dynamic_sampling must be a mapping")
    path = dynamic.get(f"{kind}_index")
    digest = _sha(
        dynamic.get(f"{kind}_index_sha256"),
        name=f"{kind}_index_sha256",
    )
    if not isinstance(path, str) or not path:
        raise ValueError(f"{kind}_index is required")
    resolved = Path(path).resolve(strict=True)
    if sha256_file(resolved) != digest:
        raise ValueError(f"{kind} index file SHA mismatch")
    return resolved, digest


def verify_common_clean_confusion_binding(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = manifest.get("common_clean_confusion_index")
    if not isinstance(contract, Mapping):
        raise ValueError("pair manifest has no common clean confusion index")
    observed = validate_neutral_oof_completion_for_pair(
        str(contract.get("neutral_completion_path", ""))
    )
    if dict(contract) != observed:
        raise ValueError("pair common clean confusion contract changed")
    if manifest.get("source_revision") != observed["source_revision"]:
        raise ValueError("pair and clean confusion source revisions differ")
    if config is not None:
        confusion_path, confusion_sha = _config_index(config, kind="confusion")
        if (
            str(confusion_path) != observed["index_path"]
            or confusion_sha != observed["index_file_sha256"]
        ):
            raise ValueError("runtime config uses another confusion index")
        class_path, class_sha = _config_index(config, kind="class")
        if (
            str(class_path) != manifest.get("common_class_index_path")
            or class_sha != manifest.get("common_class_index_sha256")
        ):
            raise ValueError("runtime config uses another common class index")
    return observed


def build_pair_manifest(
    *,
    arm_a0_config: Mapping[str, Any],
    arm_a1_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    base_checkpoint: str | Path,
    difficulty_manifest: str | Path,
    neutral_oof_completion: str | Path,
    source_revision: str,
    accept_historical_oof_proxy: bool,
    thresholds: PairedDecisionThresholds = PairedDecisionThresholds(),
) -> dict[str, Any]:
    if not _GIT_SHA_PATTERN.fullmatch(str(source_revision)):
        raise ValueError("source_revision must be a lowercase Git SHA")
    if arm_a0_config.get("train", {}).get("paired_continuation_arm") != "A0":
        raise ValueError("arm_a0_config must identify A0")
    if arm_a1_config.get("train", {}).get("paired_continuation_arm") != "A1":
        raise ValueError("arm_a1_config must identify A1")
    differences = _arm_config_differences(arm_a0_config, arm_a1_config)
    unexpected = set(differences) - _ARM_SPECIFIC_PATHS
    if unexpected:
        raise ValueError(
            "A0/A1 config difference is outside the allowlist: "
            + ", ".join(sorted(unexpected))
        )
    if set(differences) != _ARM_SPECIFIC_PATHS:
        missing = _ARM_SPECIFIC_PATHS - set(differences)
        raise ValueError(
            "A0/A1 config does not distinguish every arm field: "
            + ", ".join(sorted(missing))
        )
    normalized_a0 = normalized_common_config(arm_a0_config)
    normalized_a1 = normalized_common_config(arm_a1_config)
    if normalized_a0 != normalized_a1:
        raise AssertionError("normalized paired configs unexpectedly differ")
    evaluation_protocol = paired_evaluation_protocol(evaluation_config)
    arm_prediction_protocol = paired_evaluation_protocol(arm_a0_config)
    for field in ("data", "model", "inference"):
        if evaluation_protocol[field] != arm_prediction_protocol[field]:
            raise ValueError(
                "paired evaluation config changes prediction semantics outside "
                f"the registered execution settings: {field}"
            )
    if evaluation_protocol.get("execution") != {
        "device": "cuda",
        "cudnn_benchmark": True,
    }:
        raise ValueError(
            "paired historical evaluation must explicitly use CUDA with the "
            "original cuDNN benchmark setting"
        )
    evaluator_path = Path(__file__).resolve().parents[1] / "evaluate.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError("paired evaluation implementation is missing")
    evaluator_sha256 = sha256_file(evaluator_path)

    base = Path(base_checkpoint).resolve()
    base_sha = sha256_file(base)
    if base_sha != EXPECTED_BASE_CHECKPOINT_SHA256:
        raise ValueError("paired continuation Base checkpoint SHA mismatch")

    common_confusion = validate_neutral_oof_completion_for_pair(
        neutral_oof_completion
    )
    if common_confusion["source_revision"] != source_revision:
        raise ValueError(
            "clean neutral confusion index uses another source revision"
        )
    a0_confusion_path, a0_confusion_sha = _config_index(
        arm_a0_config, kind="confusion"
    )
    a1_confusion_path, a1_confusion_sha = _config_index(
        arm_a1_config, kind="confusion"
    )
    if (a0_confusion_path, a0_confusion_sha) != (
        a1_confusion_path,
        a1_confusion_sha,
    ):
        raise ValueError("A0/A1 must use one common confusion index")
    if (
        str(a0_confusion_path) != common_confusion["index_path"]
        or a0_confusion_sha != common_confusion["index_file_sha256"]
    ):
        raise ValueError(
            "paired confusion index is not the clean neutral OOF index"
        )
    a0_class_path, a0_class_sha = _config_index(arm_a0_config, kind="class")
    a1_class_path, a1_class_sha = _config_index(arm_a1_config, kind="class")
    if (a0_class_path, a0_class_sha) != (a1_class_path, a1_class_sha):
        raise ValueError("A0/A1 must use one common class index")

    difficulty_path = Path(difficulty_manifest).resolve()
    difficulty = _json_mapping(difficulty_path, name="difficulty manifest")
    difficulty_identity = _verify_embedded_identity(
        difficulty, name="difficulty manifest"
    )
    if difficulty.get("source_split") != "official_train":
        raise ValueError("difficulty manifest must use official_train")
    if difficulty.get("official_test_opened") is not False:
        raise ValueError("difficulty manifest must not open official test")
    tainted = bool(difficulty.get("historical_manual_oof_sampling_taint"))
    source_status = difficulty.get("oof_source_status")
    clean_evidence_sha256: str | None = None
    if tainted:
        if source_status not in {None, "historical_legacy_proxy"}:
            raise ValueError("tainted difficulty declares a clean OOF source status")
        if not accept_historical_oof_proxy:
            raise ValueError(
                "historical OOF proxy is tainted; explicit acceptance is required"
            )
        source_status = "historical_legacy_proxy"
    else:
        if accept_historical_oof_proxy:
            raise ValueError(
                "historical-proxy acceptance is forbidden for clean OOF evidence"
            )
        if source_status != "clean_dual_proxy":
            raise ValueError(
                "untainted paired difficulty must identify clean_dual_proxy evidence"
            )
        if difficulty.get("clean_dual_proxy_required") is not True:
            raise ValueError("clean dual-proxy requirement is not frozen")
        evidence = difficulty.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("clean paired difficulty is missing evidence")
        stability = evidence.get("stability")
        if not isinstance(stability, Mapping) or stability.get("passed") is not True:
            raise ValueError("clean dual-proxy stability gate did not pass")
        gates = stability.get("gates")
        expected_gates = {
            "spearman_at_least_0_60",
            "hard6_overlap_at_least_4",
            "bootstrap_spearman_q05_at_least_0_40",
            "bootstrap_hard6_overlap_rate_at_least_0_80",
        }
        if (
            not isinstance(gates, Mapping)
            or set(gates) != expected_gates
            or any(value is not True for value in gates.values())
        ):
            raise ValueError("clean dual-proxy stability gates are incomplete")
        inputs = difficulty.get("input_sha256_chain")
        if not isinstance(inputs, Mapping):
            raise ValueError("clean paired difficulty is missing its input chain")
        clean_revisions: set[str] = set()
        for source_name, expected_role in (
            ("neutral", "ms75_clean_neutral_oof"),
            ("aligned", "ms75_clean_aligned_swin_oof"),
        ):
            source = inputs.get(source_name)
            protocol = source.get("source_protocol") if isinstance(source, Mapping) else None
            if not isinstance(protocol, Mapping):
                raise ValueError(f"clean {source_name} OOF source protocol is missing")
            supplied = protocol.get("source_protocol_sha256")
            protocol_body = {
                key: value
                for key, value in protocol.items()
                if key != "source_protocol_sha256"
            }
            if not isinstance(supplied, str) or supplied != canonical_sha256(protocol_body):
                raise ValueError(f"clean {source_name} source protocol identity mismatch")
            if (
                protocol.get("role") != expected_role
                or protocol.get("source_split") != "official_train"
                or protocol.get("official_test_opened") is not False
                or protocol.get("full120_checkpoint_opened") is not False
                or protocol.get("manual_class_ids") is not False
                or protocol.get("initialization_scope")
                != "generic_pretraining_only"
            ):
                raise ValueError(f"clean {source_name} OOF contract is invalid")
            revision = protocol.get("source_revision")
            if not isinstance(revision, str) or not _GIT_SHA_PATTERN.fullmatch(revision):
                raise ValueError(f"clean {source_name} source revision is invalid")
            clean_revisions.add(revision)
        if len(clean_revisions) != 1:
            raise ValueError("clean OOF sources do not share one source revision")
        neutral_chain = inputs.get("neutral")
        if not isinstance(neutral_chain, Mapping):
            raise ValueError("clean difficulty has no neutral input chain")
        if (
            Path(str(neutral_chain.get("provenance", ""))).resolve()
            != Path(common_confusion["provenance_path"])
            or neutral_chain.get("provenance_file_sha256")
            != common_confusion["provenance_file_sha256"]
        ):
            raise ValueError(
                "clean difficulty and common confusion index use different "
                "neutral provenance"
            )
        neutral_source = neutral_chain.get("source_protocol")
        if (
            not isinstance(neutral_source, Mapping)
            or neutral_source.get("source_protocol_sha256")
            != common_confusion["source_protocol_sha256"]
            or neutral_source.get("source_revision")
            != common_confusion["source_revision"]
        ):
            raise ValueError(
                "clean difficulty and common confusion index use different "
                "source protocols"
            )
        clean_evidence_sha256 = canonical_sha256(evidence)
    tiers = difficulty.get("tiers")
    if not isinstance(tiers, Mapping):
        raise ValueError("difficulty manifest is missing tiers")
    hard_ids = tiers.get("hard_class_ids")
    if (
        not isinstance(hard_ids, list)
        or len(hard_ids) != 6
        or len(set(int(value) for value in hard_ids)) != 6
    ):
        raise ValueError("difficulty manifest must freeze six hard classes")

    a0_path, a0_expected = _config_probability(arm_a0_config)
    a1_path, a1_expected = _config_probability(arm_a1_config)
    a0, a0_file_sha = _validate_probability_manifest(
        a0_path,
        expected_file_sha256=a0_expected,
        expected_arm="A0",
        parent_difficulty_sha256=difficulty_identity,
    )
    a1, a1_file_sha = _validate_probability_manifest(
        a1_path,
        expected_file_sha256=a1_expected,
        expected_arm="A1",
        parent_difficulty_sha256=difficulty_identity,
    )
    projected = np.asarray(
        tiers.get("projected_probabilities_by_class"), dtype=np.float64
    )
    if projected.shape != (NUM_CLASSES,) or not np.allclose(
        projected,
        np.asarray(a1["probabilities"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "A1 probabilities do not match frozen difficulty tiers"
        )

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "role": "ms75_paired_continuation_contract",
        "protocol_mode": "historical_benchmark_exploratory",
        "official_test_opened": False,
        "source_revision": source_revision,
        "formal_base_source_revision": FORMAL_BASE_SOURCE_REVISION,
        "evaluation_implementation": str(evaluator_path),
        "evaluation_implementation_sha256": evaluator_sha256,
        "base_checkpoint": str(base),
        "base_checkpoint_sha256": base_sha,
        "common_class_index_path": str(a0_class_path),
        "common_class_index_sha256": a0_class_sha,
        "common_clean_confusion_index": common_confusion,
        "common_config_fingerprint": canonical_sha256(normalized_a0),
        "evaluation_protocol": evaluation_protocol,
        "allowed_arm_config_differences": sorted(_ARM_SPECIFIC_PATHS),
        "observed_arm_config_differences": sorted(differences),
        "difficulty_manifest": str(difficulty_path),
        "difficulty_manifest_file_sha256": sha256_file(difficulty_path),
        "difficulty_manifest_sha256": difficulty_identity,
        "oof_source_status": source_status,
        "clean_dual_proxy_evidence_sha256": clean_evidence_sha256,
        "historical_manual_oof_sampling_taint": tainted,
        "historical_oof_proxy_explicitly_accepted": bool(
            accept_historical_oof_proxy
        ),
        "hard_class_ids": [int(value) for value in hard_ids],
        "arms": {
            "A0": {
                "config_fingerprint": canonical_sha256(
                    _normalized_arm_config(arm_a0_config)
                ),
                "class_probability_manifest": str(Path(a0_path).resolve()),
                "class_probability_manifest_file_sha256": a0_file_sha,
                "class_probability_manifest_sha256": a0["manifest_sha256"],
            },
            "A1": {
                "config_fingerprint": canonical_sha256(
                    _normalized_arm_config(arm_a1_config)
                ),
                "class_probability_manifest": str(Path(a1_path).resolve()),
                "class_probability_manifest_file_sha256": a1_file_sha,
                "class_probability_manifest_sha256": a1["manifest_sha256"],
            },
        },
        "decision_thresholds": asdict(thresholds),
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def verify_pair_manifest_identity(manifest: Mapping[str, Any]) -> str:
    return _verify_embedded_identity(manifest, name="pair manifest")


def verify_runtime_pair_binding(
    *,
    config: Mapping[str, Any],
    pair_manifest_path: str | Path,
    expected_pair_file_sha256: str,
    expected_base_checkpoint_sha256: str,
    verify_source_checkout: bool = True,
) -> dict[str, Any]:
    """Bind one runtime arm to every authenticated pair-manifest identity.

    A file-level SHA alone proves only that a particular JSON file was opened;
    it does not prove that the JSON describes this resolved arm config or its
    class-probability artifact.  This validator closes that gap before the
    trainer allocates a model or constructs an optimizer.
    """

    pair_path = Path(pair_manifest_path).resolve()
    expected_pair_file_sha256 = _sha(
        expected_pair_file_sha256,
        name="paired_protocol_manifest_expected_sha256",
    )
    observed_pair_file_sha256 = sha256_file(pair_path)
    if observed_pair_file_sha256 != expected_pair_file_sha256:
        raise ValueError("paired protocol manifest file SHA mismatch")
    manifest = _json_mapping(pair_path, name="pair manifest")
    pair_identity = verify_pair_manifest_identity(manifest)
    if manifest.get("role") != "ms75_paired_continuation_contract":
        raise ValueError("pair manifest has the wrong protocol role")
    if manifest.get("official_test_opened") is not False:
        raise ValueError("pair manifest must not open official test")
    source_revision = str(manifest.get("source_revision", ""))
    if not _GIT_SHA_PATTERN.fullmatch(source_revision):
        raise ValueError("pair manifest has no valid source revision")
    if verify_source_checkout:
        project_root = Path(__file__).resolve().parents[1]
        try:
            current_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("paired runtime cannot inspect source checkout") from error
        if current_revision != source_revision:
            raise ValueError("paired runtime source revision differs from preflight")
        if dirty:
            raise ValueError("paired runtime requires a clean committed source tree")
    expected_base_checkpoint_sha256 = _sha(
        expected_base_checkpoint_sha256,
        name="runtime base checkpoint SHA256",
    )
    if (
        manifest.get("base_checkpoint_sha256")
        != expected_base_checkpoint_sha256
    ):
        raise ValueError("pair manifest is bound to another Base checkpoint")
    common_confusion = verify_common_clean_confusion_binding(
        manifest, config=config
    )

    train = config.get("train")
    if not isinstance(train, Mapping):
        raise ValueError("paired runtime config has no train mapping")
    arm = str(train.get("paired_continuation_arm", ""))
    if arm not in {"A0", "A1"}:
        raise ValueError("paired runtime arm must be A0 or A1")
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"A0", "A1"}:
        raise ValueError("pair manifest must contain exactly A0 and A1 arms")
    arm_contract = arms.get(arm)
    if not isinstance(arm_contract, Mapping):
        raise ValueError(f"pair manifest has no {arm} contract")
    runtime_config_fingerprint = canonical_sha256(
        _normalized_arm_config(config)
    )
    if arm_contract.get("config_fingerprint") != runtime_config_fingerprint:
        raise ValueError(
            f"pair manifest {arm} resolved config fingerprint mismatch"
        )

    probability_path, probability_file_sha256 = _config_probability(config)
    configured_probability_path = Path(probability_path).resolve()
    contracted_probability_path = Path(
        str(arm_contract.get("class_probability_manifest", ""))
    ).resolve()
    if configured_probability_path != contracted_probability_path:
        raise ValueError(
            f"pair manifest {arm} class probability path mismatch"
        )
    if (
        arm_contract.get("class_probability_manifest_file_sha256")
        != probability_file_sha256
    ):
        raise ValueError(
            f"pair manifest {arm} class probability file SHA mismatch"
        )
    parent_difficulty_sha256 = _sha(
        manifest.get("difficulty_manifest_sha256"),
        name="pair difficulty manifest SHA256",
    )
    probability, observed_probability_file_sha256 = (
        _validate_probability_manifest(
            configured_probability_path,
            expected_file_sha256=probability_file_sha256,
            expected_arm=arm,
            parent_difficulty_sha256=parent_difficulty_sha256,
        )
    )
    if observed_probability_file_sha256 != probability_file_sha256:
        raise AssertionError(
            "probability file identity changed during validation"
        )
    if (
        arm_contract.get("class_probability_manifest_sha256")
        != probability["manifest_sha256"]
    ):
        raise ValueError(
            f"pair manifest {arm} class probability inner SHA mismatch"
        )
    return {
        "pair_manifest_file_sha256": observed_pair_file_sha256,
        "pair_manifest_sha256": pair_identity,
        "source_revision": source_revision,
        "arm": arm,
        "arm_config_fingerprint": runtime_config_fingerprint,
        "base_checkpoint_sha256": expected_base_checkpoint_sha256,
        "class_probability_manifest_path": str(configured_probability_path),
        "class_probability_manifest_file_sha256": probability_file_sha256,
        "class_probability_manifest_sha256": probability["manifest_sha256"],
        "difficulty_manifest_sha256": parent_difficulty_sha256,
        "common_class_index_path": manifest["common_class_index_path"],
        "common_class_index_sha256": manifest["common_class_index_sha256"],
        "common_clean_confusion_index": common_confusion,
    }


__all__ = [
    "EXPECTED_BASE_CHECKPOINT_SHA256",
    "FORMAL_BASE_SOURCE_REVISION",
    "PairedDecisionThresholds",
    "build_pair_manifest",
    "canonical_json",
    "canonical_sha256",
    "normalized_common_config",
    "paired_evaluation_protocol",
    "verify_pair_manifest_identity",
    "validate_neutral_oof_completion_for_pair",
    "verify_common_clean_confusion_binding",
    "verify_runtime_pair_binding",
]
