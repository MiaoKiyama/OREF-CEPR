from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from .data import (
    FBPSceneReader,
    _extract_boundless,
    _fbp_resize_multiband,
    read_fbp_split,
    validate_fbp_splits,
)


SAMPLING_KINDS = ("scene", "class", "confusion")
OOF_PROVENANCE_SCHEMA_VERSION = 1
CLEAN_OOF_PROVENANCE_SCHEMA_VERSION = 2
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CLASS_PROBABILITY_SCHEMA_VERSION = 1
CLASS_PROBABILITY_COUNT = 24

# Must remain identical to ``ms75_model.fine_to_family_map()``.  Keeping this
# immutable protocol constant in the data layer avoids importing the model just
# to reject invalid OOF pairs before any large arrays/readers are constructed.
FBP_FINE_TO_FAMILY = (
    0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 0,
    4, 4, 4, 5, 5, 0, 0, 0, 6, 6, 6, 6,
)


def _verified_file_sha256(
    path: str | Path,
    expected_sha256: str | None,
    *,
    artifact_name: str,
) -> str:
    """Hash a sampling artifact before opening it and enforce its pin.

    The index path alone is part of the configuration fingerprint, but a file
    at that path could otherwise be replaced between an interruption and a
    resume.  Pinning the content makes the crop sequence reproducible as well
    as provenance-auditable.
    """

    source = Path(path)
    expected = "" if expected_sha256 is None else str(expected_sha256).lower()
    if not _SHA256_PATTERN.fullmatch(expected):
        raise ValueError(
            f"{artifact_name} requires an expected SHA256 with 64 lowercase "
            "hexadecimal characters"
        )
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected:
        raise ValueError(
            f"{artifact_name} SHA256 mismatch: expected {expected}, "
            f"observed {observed}"
        )
    return observed


def canonical_class_probability_json(manifest: Mapping[str, Any]) -> str:
    """Canonical bytes used by a class-probability manifest's inner pin."""

    body = dict(manifest)
    body.pop("manifest_sha256", None)
    return json.dumps(
        body,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ClassProbabilityManifest:
    """Authenticated A0/A1 probability contract for the 24 class keys."""

    arm: str
    probabilities: tuple[float, ...]
    file_sha256: str
    manifest_sha256: str
    parent_difficulty_sha256: str
    lower_multiplier: float
    upper_multiplier: float


def load_class_probability_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None,
) -> ClassProbabilityManifest:
    """Load and fail closed on a content-addressed A0/A1 probability vector."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing class probability manifest: {source}")
    file_sha256 = _verified_file_sha256(
        source,
        expected_sha256,
        artifact_name="class probability manifest",
    )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("class probability manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("class probability manifest must be a JSON object")
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
    missing = required - set(value)
    if missing:
        raise ValueError(
            "class probability manifest is missing required fields: "
            f"{sorted(missing)}"
        )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != CLASS_PROBABILITY_SCHEMA_VERSION
    ):
        raise ValueError("unsupported class probability manifest schema_version")
    if type(value["source_split"]) is not str or value["source_split"] != (
        "official_train"
    ):
        raise ValueError(
            "class probability manifest must come from official_train"
        )
    arm = value["arm"]
    if type(arm) is not str or arm not in {"A0", "A1"}:
        raise ValueError("class probability manifest arm must be A0 or A1")
    class_ids = value["class_ids"]
    if (
        not isinstance(class_ids, list)
        or any(type(item) is not int for item in class_ids)
        or class_ids != list(range(CLASS_PROBABILITY_COUNT))
    ):
        raise ValueError("class probability manifest class_ids must be 0..23")
    probabilities_value = value["probabilities"]
    if not isinstance(probabilities_value, list) or len(probabilities_value) != 24:
        raise ValueError(
            "class probability manifest probabilities must contain 24 values"
        )
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           for item in probabilities_value):
        raise ValueError("class probabilities must be finite numeric values")
    probabilities = tuple(float(item) for item in probabilities_value)
    if not all(math.isfinite(item) and item > 0 for item in probabilities):
        raise ValueError("class probabilities must be finite and positive")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("class probabilities must sum to one")
    lower_multiplier = value["lower_multiplier"]
    upper_multiplier = value["upper_multiplier"]
    if (
        isinstance(lower_multiplier, bool)
        or isinstance(upper_multiplier, bool)
        or not isinstance(lower_multiplier, (int, float))
        or not isinstance(upper_multiplier, (int, float))
        or not math.isfinite(float(lower_multiplier))
        or not math.isfinite(float(upper_multiplier))
        or not math.isclose(float(lower_multiplier), 0.5, abs_tol=1e-12)
        or not math.isclose(float(upper_multiplier), 3.0, abs_tol=1e-12)
    ):
        raise ValueError("class probability multipliers must be exactly 0.5 and 3.0")
    lower = float(lower_multiplier) / CLASS_PROBABILITY_COUNT
    upper = float(upper_multiplier) / CLASS_PROBABILITY_COUNT
    if any(item < lower - 1e-12 or item > upper + 1e-12 for item in probabilities):
        raise ValueError("class probabilities violate the registered 0.5x/3.0x caps")
    uniform = 1.0 / CLASS_PROBABILITY_COUNT
    if arm == "A0" and any(
        not math.isclose(item, uniform, rel_tol=0.0, abs_tol=1e-12)
        for item in probabilities
    ):
        raise ValueError("A0 class probabilities must be uniform")
    if arm == "A1" and all(
        math.isclose(item, uniform, rel_tol=0.0, abs_tol=1e-12)
        for item in probabilities
    ):
        raise ValueError("A1 class probabilities must differ from uniform")
    parent_sha256 = value["parent_difficulty_sha256"]
    manifest_sha256 = value["manifest_sha256"]
    if not isinstance(parent_sha256, str) or not _SHA256_PATTERN.fullmatch(
        parent_sha256
    ):
        raise ValueError("parent_difficulty_sha256 must be lowercase SHA256")
    if not isinstance(manifest_sha256, str) or not _SHA256_PATTERN.fullmatch(
        manifest_sha256
    ):
        raise ValueError("manifest_sha256 must be lowercase SHA256")
    observed_inner_sha256 = hashlib.sha256(
        canonical_class_probability_json(value).encode("utf-8")
    ).hexdigest()
    if observed_inner_sha256 != manifest_sha256:
        raise ValueError(
            "class probability manifest inner SHA256 mismatch: expected "
            f"{manifest_sha256}, observed {observed_inner_sha256}"
        )
    return ClassProbabilityManifest(
        arm=arm,
        probabilities=probabilities,
        file_sha256=file_sha256,
        manifest_sha256=manifest_sha256,
        parent_difficulty_sha256=parent_sha256,
        lower_multiplier=float(lower_multiplier),
        upper_multiplier=float(upper_multiplier),
    )


def canonical_oof_provenance_json(manifest: Mapping[str, Any]) -> str:
    """Return the portable byte representation embedded in an OOF archive."""

    return json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validated_manifest_scene_list(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"OOF provenance {name} must be a non-empty list")
    scene_ids = tuple(str(item) for item in value)
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError(f"OOF provenance {name} contains duplicate scenes")
    for scene_id in scene_ids:
        if Path(scene_id).name != scene_id or "/" in scene_id or "\\" in scene_id:
            raise ValueError(f"unsafe OOF provenance scene ID: {scene_id!r}")
    return scene_ids


def _validate_manifest_artifact(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(
            f"OOF provenance {name} must contain exactly path and sha256"
        )
    path = value["path"]
    digest = value["sha256"]
    if not isinstance(path, str) or not path:
        raise ValueError(f"OOF provenance {name}.path must be non-empty")
    if "\\" in path:
        raise ValueError(
            f"OOF provenance {name}.path must use portable '/' separators"
        )
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ValueError(f"OOF provenance {name}.path must be a safe relative path")
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"OOF provenance {name}.sha256 must be lowercase SHA256")
    return {"path": path, "sha256": digest}


def validate_oof_provenance_manifest(
    manifest: object,
    *,
    official_train_scene_ids: Sequence[str],
    official_test_scene_ids: Sequence[str],
) -> dict[str, int]:
    """Validate a complete per-fold OOF proof and return ``scene -> fold``.

    Each fold must document the exact complementary training set, its held-out
    scenes, a checkpoint digest, and one prediction digest per held-out scene.
    This validates the internal closure embedded by the builder; the builder
    separately reads each checkpoint's ``data_provenance`` to corroborate that
    the manifest is not merely a renamed flat fold map.
    """

    if not isinstance(manifest, dict):
        raise ValueError("OOF provenance manifest must be a mapping")
    schema_version = manifest.get("schema_version")
    expected_top_level = {
        "schema_version",
        "source_split",
        "folds",
    }
    if schema_version == CLEAN_OOF_PROVENANCE_SCHEMA_VERSION:
        expected_top_level.add("source_protocol")
    if set(manifest) != expected_top_level:
        raise ValueError(
            "OOF provenance manifest has invalid top-level fields for its schema"
        )
    if schema_version not in {
        OOF_PROVENANCE_SCHEMA_VERSION,
        CLEAN_OOF_PROVENANCE_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"unsupported OOF provenance schema_version={manifest['schema_version']!r}"
        )
    if manifest["source_split"] != "official_train":
        raise ValueError("OOF provenance source_split must be official_train")
    clean_source_protocol: dict[str, Any] | None = None
    if schema_version == CLEAN_OOF_PROVENANCE_SCHEMA_VERSION:
        source_protocol = manifest.get("source_protocol")
        if not isinstance(source_protocol, Mapping):
            raise ValueError("clean OOF provenance requires source_protocol")
        role = source_protocol.get("role")
        if role == "ms75_clean_neutral_oof":
            from .ms75_neutral_oof import validate_neutral_source_protocol

            clean_source_protocol = validate_neutral_source_protocol(
                source_protocol
            )
        elif role == "ms75_clean_aligned_swin_oof":
            from .ms75_aligned_protocol import validate_aligned_source_protocol

            clean_source_protocol = validate_aligned_source_protocol(
                source_protocol
            )
        else:
            raise ValueError("unsupported clean OOF source_protocol role")
    folds = manifest["folds"]
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("OOF provenance must contain at least two held-out folds")

    official_train = tuple(str(value) for value in official_train_scene_ids)
    train_set = set(official_train)
    test_set = {str(value) for value in official_test_scene_ids}
    if len(train_set) != len(official_train) or train_set & test_set:
        raise ValueError("invalid official train/test scene universe")

    scene_to_fold: dict[str, int] = {}
    fold_ids: set[int] = set()
    checkpoint_hashes: set[str] = set()
    for fold_number, fold in enumerate(folds):
        expected_fold_fields = {
            "fold_id",
            "train_scene_ids",
            "heldout_scene_ids",
            "checkpoint",
            "predictions",
        }
        if schema_version == CLEAN_OOF_PROVENANCE_SCHEMA_VERSION:
            expected_fold_fields.add("training_protocol")
        if not isinstance(fold, dict) or set(fold) != expected_fold_fields:
            raise ValueError(
                f"OOF provenance fold[{fold_number}] has an invalid schema"
            )
        fold_id = fold["fold_id"]
        if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id < 0:
            raise ValueError("OOF provenance fold_id must be a non-negative integer")
        if fold_id in fold_ids:
            raise ValueError(f"duplicate OOF provenance fold_id={fold_id}")
        fold_ids.add(fold_id)
        training = _validated_manifest_scene_list(
            fold["train_scene_ids"], name=f"fold[{fold_id}].train_scene_ids"
        )
        heldout = _validated_manifest_scene_list(
            fold["heldout_scene_ids"], name=f"fold[{fold_id}].heldout_scene_ids"
        )
        training_set = set(training)
        heldout_set = set(heldout)
        if training_set & heldout_set:
            raise ValueError(
                f"OOF provenance fold {fold_id} has a scene in train and heldout"
            )
        if (training_set | heldout_set) & test_set:
            raise ValueError("official test scene leaked into OOF provenance")
        if training_set | heldout_set != train_set:
            raise ValueError(
                f"OOF provenance fold {fold_id} is not an exact partition of "
                "official training scenes"
            )
        if training_set != train_set - heldout_set:
            raise ValueError(
                f"OOF provenance fold {fold_id} training scenes are not the "
                "complement of heldout scenes"
            )

        checkpoint = _validate_manifest_artifact(
            fold["checkpoint"], name=f"fold[{fold_id}].checkpoint"
        )
        if schema_version == CLEAN_OOF_PROVENANCE_SCHEMA_VERSION:
            training_protocol = fold.get("training_protocol")
            required_training_fields = {
                "schema_version",
                "role",
                "source_protocol_sha256",
                "source_revision",
                "config_fingerprint",
                "resolved_config",
                "run_info",
                "sampling_index",
                "checkpoint_role",
                "selection_metric",
                "validation_enabled",
                "successful_optimizer_updates",
                "sampler_audit",
                "initialization_provenance",
                "execution_provenance",
            }
            if (
                not isinstance(training_protocol, Mapping)
                or set(training_protocol) != required_training_fields
                or training_protocol.get("schema_version") != 1
                or clean_source_protocol is None
                or training_protocol.get("role")
                != clean_source_protocol.get("role")
                or training_protocol.get("source_protocol_sha256")
                != clean_source_protocol.get("source_protocol_sha256")
                or training_protocol.get("source_revision")
                != clean_source_protocol.get("source_revision")
                or training_protocol.get("selection_metric") != "last_epoch"
                or training_protocol.get("validation_enabled") is not False
            ):
                raise ValueError(
                    f"clean OOF fold {fold_id} training_protocol is invalid"
                )
            for artifact_name in ("resolved_config", "run_info", "sampling_index"):
                _validate_manifest_artifact(
                    training_protocol[artifact_name],
                    name=f"fold[{fold_id}].training_protocol.{artifact_name}",
                )
            if not isinstance(
                training_protocol.get("execution_provenance"), Mapping
            ):
                raise ValueError(
                    f"clean OOF fold {fold_id} execution_provenance is invalid"
                )
        if checkpoint["sha256"] in checkpoint_hashes:
            raise ValueError(
                "OOF provenance reuses an identical checkpoint across folds"
            )
        checkpoint_hashes.add(checkpoint["sha256"])
        predictions = fold["predictions"]
        if not isinstance(predictions, dict) or set(predictions) != heldout_set:
            raise ValueError(
                f"OOF provenance fold {fold_id} predictions must cover exactly "
                "its heldout scenes"
            )
        for scene_id, artifact in predictions.items():
            if schema_version == CLEAN_OOF_PROVENANCE_SCHEMA_VERSION:
                if not isinstance(artifact, dict) or set(artifact) != {
                    "path",
                    "sha256",
                    "receipt_path",
                    "receipt_sha256",
                }:
                    raise ValueError(
                        "clean OOF prediction artifact must bind its receipt"
                    )
                prediction = _validate_manifest_artifact(
                    {key: artifact[key] for key in ("path", "sha256")},
                    name=f"fold[{fold_id}].predictions[{scene_id}]",
                )
                receipt = _validate_manifest_artifact(
                    {
                        "path": artifact["receipt_path"],
                        "sha256": artifact["receipt_sha256"],
                    },
                    name=f"fold[{fold_id}].prediction_receipts[{scene_id}]",
                )
                if PurePosixPath(receipt["path"]).suffix.lower() != ".json":
                    raise ValueError("clean OOF prediction receipt must be JSON")
            else:
                prediction = _validate_manifest_artifact(
                    artifact,
                    name=f"fold[{fold_id}].predictions[{scene_id}]",
                )
            if PurePosixPath(prediction["path"]).suffix.lower() != ".png":
                raise ValueError("OOF prediction artifact must be a PNG file")
            if scene_id in scene_to_fold:
                raise ValueError(
                    f"OOF scene {scene_id} is held out by more than one fold"
                )
            scene_to_fold[scene_id] = fold_id

    if set(scene_to_fold) != train_set:
        missing = sorted(train_set - set(scene_to_fold))[:5]
        raise ValueError(
            "OOF provenance heldout folds must cover every official training "
            f"scene exactly once; missing={missing}"
        )
    return scene_to_fold


@dataclass(frozen=True)
class OfficialFBPProtocol:
    """The immutable official FBP 120-train/30-test scene protocol."""

    all_scene_ids: tuple[str, ...]
    train_scene_ids: tuple[str, ...]
    test_scene_ids: tuple[str, ...]

    @classmethod
    def from_root(cls, root: str | Path) -> "OfficialFBPProtocol":
        splits = validate_fbp_splits(root)
        protocol = cls(
            all_scene_ids=tuple(splits["all"]),
            train_scene_ids=tuple(splits["train"]),
            test_scene_ids=tuple(splits["test"]),
        )
        if len(protocol.train_scene_ids) != 120:
            raise ValueError("MS75 requires exactly 120 official training scenes")
        if len(protocol.test_scene_ids) != 30:
            raise ValueError("MS75 requires exactly 30 official test scenes")
        if set(protocol.train_scene_ids) & set(protocol.test_scene_ids):
            raise ValueError("official FBP train and test scenes overlap")
        return protocol


@dataclass(frozen=True)
class SceneDevelopmentSplit:
    """A scene-disjoint development split made only from official train scenes."""

    train_scene_ids: tuple[str, ...]
    dev_scene_ids: tuple[str, ...]


def make_scene_development_split(
    protocol: OfficialFBPProtocol,
    *,
    dev_scene_count: int = 12,
    seed: int = 3407,
) -> SceneDevelopmentSplit:
    """Create a deterministic scene-level split without touching official test.

    Hash ranking makes the result independent of filesystem enumeration and
    Python's randomized hash seed.  Final fitting should bypass this helper and
    use all ``protocol.train_scene_ids``.
    """

    if not 0 < int(dev_scene_count) < len(protocol.train_scene_ids):
        raise ValueError("dev_scene_count must be between 1 and 119")

    def rank(scene_id: str) -> bytes:
        return hashlib.blake2b(
            f"{int(seed)}:{scene_id}".encode("utf-8"), digest_size=16
        ).digest()

    ranked = sorted(protocol.train_scene_ids, key=rank)
    dev_set = set(ranked[: int(dev_scene_count)])
    train = tuple(
        scene_id
        for scene_id in protocol.train_scene_ids
        if scene_id not in dev_set
    )
    dev = tuple(
        scene_id for scene_id in protocol.train_scene_ids if scene_id in dev_set
    )
    if set(train) & set(dev) or set(train) | set(dev) != set(
        protocol.train_scene_ids
    ):
        raise AssertionError("scene development split is not a partition")
    if (set(train) | set(dev)) & set(protocol.test_scene_ids):
        raise AssertionError("official test scene leaked into development")
    return SceneDevelopmentSplit(train_scene_ids=train, dev_scene_ids=dev)


def _normalized_local_box(
    local: Window,
    support: Window,
) -> np.ndarray:
    support_height = float(support.height)
    support_width = float(support.width)
    if support_height <= 0 or support_width <= 0:
        raise ValueError("support window must be non-empty")
    return np.asarray(
        [
            (float(local.row_off) - float(support.row_off)) / support_height,
            (float(local.col_off) - float(support.col_off)) / support_width,
            (
                float(local.row_off)
                + float(local.height)
                - float(support.row_off)
            )
            / support_height,
            (
                float(local.col_off)
                + float(local.width)
                - float(support.col_off)
            )
            / support_width,
        ],
        dtype=np.float32,
    )


class MS75SceneReader(FBPSceneReader):
    """Read aligned local, short-range and whole-scene four-band MS views.

    FBP stores samples as NIR,R,G,B.  All returned image tensors follow the
    Line3 B,G,R,NIR normalization contract inherited from ``FBPSceneReader``.
    The whole-scene thumbnail is resized and normalized once per reader/size.
    """

    def __init__(self, root: str | Path, scene_id: str, config: dict):
        super().__init__(root, scene_id, config)
        self._global_thumbnail_cache: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] = {}
        self.shared_global_thumbnail_cache: OrderedDict[
            tuple[str, int], tuple[np.ndarray, np.ndarray]
        ] | None = None
        self.shared_global_thumbnail_cache_size = 0

    def close(self) -> None:
        self._global_thumbnail_cache = {}
        super().close()

    @staticmethod
    def _centered_support(local: Window, context_size: int) -> Window:
        context_size = int(context_size)
        if context_size < int(local.width) or context_size < int(local.height):
            raise ValueError("short_context_size cannot be smaller than local patch")
        center_x = float(local.col_off) + float(local.width) / 2.0
        center_y = float(local.row_off) + float(local.height) / 2.0
        left = math.floor(center_x - context_size / 2.0)
        top = math.floor(center_y - context_size / 2.0)
        return Window(left, top, context_size, context_size)

    def _read_resized_support(
        self,
        support: Window,
        output_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw, valid = _extract_boundless(
            self.image,
            left=int(support.col_off),
            top=int(support.row_off),
            width=int(support.width),
            height=int(support.height),
        )
        resized = _fbp_resize_multiband(raw, int(output_size))
        # Nearest-neighbour validity deliberately matches the spatial support
        # used by the resized pixels and never labels padded pixels as observed.
        from PIL import Image

        resized_valid = np.asarray(
            Image.fromarray(valid.astype(np.uint8) * 255, mode="L").resize(
                (int(output_size), int(output_size)),
                resample=Image.Resampling.NEAREST,
            )
        ) > 0
        normalized = self._normalize_ms(resized)
        return normalized * resized_valid[None], resized_valid

    def _global_thumbnail(
        self, output_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        output_size = int(output_size)
        if output_size <= 0:
            raise ValueError("global_input_size must be positive")
        shared_key = (self.record.scene_id, output_size)
        shared = self.shared_global_thumbnail_cache
        cached = shared.get(shared_key) if shared is not None else None
        if cached is not None:
            shared.move_to_end(shared_key)
            return cached
        cached = self._global_thumbnail_cache.get(output_size)
        if cached is None:
            resized = _fbp_resize_multiband(self.image, output_size)
            normalized = self._normalize_ms(resized)
            valid = np.ones((output_size, output_size), dtype=np.bool_)
            cached = (normalized, valid)
            self._global_thumbnail_cache[output_size] = cached
        if shared is not None and self.shared_global_thumbnail_cache_size > 0:
            shared[shared_key] = cached
            shared.move_to_end(shared_key)
            while len(shared) > self.shared_global_thumbnail_cache_size:
                shared.popitem(last=False)
        return cached

    def read_ms75_patch(
        self,
        local_window: Window,
        *,
        short_context_size: int,
        short_input_size: int,
        global_input_size: int,
    ) -> dict[str, np.ndarray]:
        if int(local_window.width) <= 0 or int(local_window.height) <= 0:
            raise ValueError("local window must be non-empty")
        local_raw, local_valid = _extract_boundless(
            self.image,
            left=int(local_window.col_off),
            top=int(local_window.row_off),
            width=int(local_window.width),
            height=int(local_window.height),
        )
        target, supervised = self.read_label_valid(local_window)
        short_window = self._centered_support(
            local_window, int(short_context_size)
        )
        short, short_valid = self._read_resized_support(
            short_window, int(short_input_size)
        )
        global_ms, global_valid = self._global_thumbnail(
            int(global_input_size)
        )
        global_window = Window(0, 0, self.width, self.height)
        local = self._normalize_ms(local_raw) * local_valid[None]
        return {
            # ``ms`` remains the local view for compatibility with Line3.
            "ms": local.astype(np.float32, copy=False),
            "ms_valid": local_valid,
            "ms_short": short.astype(np.float32, copy=False),
            "ms_short_valid": short_valid,
            "ms_global": global_ms.astype(np.float32, copy=False),
            "ms_global_valid": global_valid,
            "target": target.astype(np.int64, copy=False),
            "supervised": supervised,
            "local_box_short": _normalized_local_box(
                local_window, short_window
            ),
            "local_box_global": _normalized_local_box(
                local_window, global_window
            ),
            "short_support": np.asarray(
                [
                    int(short_window.row_off),
                    int(short_window.col_off),
                    int(short_window.height),
                    int(short_window.width),
                ],
                dtype=np.int64,
            ),
        }


def _flip_spatial(value: object, *, horizontal: bool, vertical: bool) -> object:
    if isinstance(value, torch.Tensor):
        dims: list[int] = []
        if vertical:
            dims.append(-2)
        if horizontal:
            dims.append(-1)
        return torch.flip(value, dims=dims) if dims else value
    if isinstance(value, np.ndarray):
        result = value
        if vertical:
            result = np.flip(result, axis=-2)
        if horizontal:
            result = np.flip(result, axis=-1)
        return result.copy() if (horizontal or vertical) else result
    raise TypeError(f"unsupported spatial value type: {type(value).__name__}")


def transform_normalized_box_for_flip(
    box: np.ndarray | torch.Tensor,
    *,
    horizontal: bool,
    vertical: bool,
) -> np.ndarray | torch.Tensor:
    """Transform a normalized ``[y0,x0,y1,x1]`` box after view flips."""

    if tuple(box.shape) != (4,):
        raise ValueError(f"expected normalized box shape (4,), got {box.shape}")
    result = box.clone() if isinstance(box, torch.Tensor) else box.copy()
    y0, x0, y1, x1 = (box[index].item() for index in range(4))
    if vertical:
        result[0], result[2] = 1.0 - y1, 1.0 - y0
    if horizontal:
        result[1], result[3] = 1.0 - x1, 1.0 - x0
    return result


def synchronized_ms75_flip(
    sample: Mapping[str, object],
    *,
    horizontal: bool,
    vertical: bool,
) -> dict[str, object]:
    """Flip every aligned MS75 view, mask, target and local-position box."""

    result = dict(sample)
    spatial_keys = (
        "ms",
        "ms_valid",
        "ms_short",
        "ms_short_valid",
        "ms_global",
        "ms_global_valid",
        "target",
        "supervised",
    )
    for key in spatial_keys:
        if key in result:
            result[key] = _flip_spatial(
                result[key], horizontal=horizontal, vertical=vertical
            )
    for key in ("local_box_short", "local_box_global"):
        if key in result:
            box = result[key]
            if not isinstance(box, (np.ndarray, torch.Tensor)):
                raise TypeError(f"{key} must be a numpy array or tensor")
            result[key] = transform_normalized_box_for_flip(
                box, horizontal=horizontal, vertical=vertical
            )
    return result


def synchronized_ms75_spectral_augmentation(
    sample: Mapping[str, object],
    *,
    data_config: Mapping[str, object],
    rng: random.Random,
) -> dict[str, object]:
    """Apply one radiometric perturbation to every aligned MS view.

    The inherited Line3 configuration already defines reflectance-space
    global and per-band gain/offset ranges.  Ignoring it here would make the
    new dataset silently weaker than both MS70 and the imported anchor's
    original training recipe.  One draw is shared by local, short and global
    evidence so the augmentation cannot create an artificial cross-scale
    disagreement.  Invalid padded pixels retain their original zero value.
    """

    result = dict(sample)
    spectral = data_config.get("spectral_augmentation", {})
    if not isinstance(spectral, Mapping) or not bool(
        spectral.get("enabled", False)
    ):
        return result
    probability = float(spectral.get("probability", 0.0))
    if rng.random() >= probability:
        return result
    normalization_root = data_config.get("normalization", {})
    if not isinstance(normalization_root, Mapping):
        raise ValueError("MS75 normalization must be a mapping")
    normalization = normalization_root.get("ms", {})
    if not isinstance(normalization, Mapping):
        raise ValueError("MS75 normalization.ms must be a mapping")
    mean = np.asarray(normalization.get("mean", ()), dtype=np.float32)
    std = np.asarray(normalization.get("std", ()), dtype=np.float32)
    if mean.shape != (4,) or std.shape != (4,) or bool((std <= 0).any()):
        raise ValueError("MS75 spectral augmentation requires four valid MS statistics")
    mean = mean[:, None, None]
    std = std[:, None, None]

    def interval(name: str, default: tuple[float, float]) -> tuple[float, float]:
        values = spectral.get(name, default)
        if not isinstance(values, Sequence) or len(values) != 2:
            raise ValueError(f"spectral_augmentation.{name} must have two values")
        lower, upper = (float(value) for value in values)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"invalid spectral_augmentation.{name}")
        return lower, upper

    global_gain = rng.uniform(*interval("global_gain_range", (1.0, 1.0)))
    global_offset = rng.uniform(
        *interval("global_offset_range", (0.0, 0.0))
    )
    band_gain_range = interval("band_gain_range", (1.0, 1.0))
    band_offset_range = interval("band_offset_range", (0.0, 0.0))
    band_gain = np.asarray(
        [rng.uniform(*band_gain_range) for _ in range(4)], dtype=np.float32
    )[:, None, None]
    band_offset = np.asarray(
        [rng.uniform(*band_offset_range) for _ in range(4)], dtype=np.float32
    )[:, None, None]
    for key, valid_key in (
        ("ms", "ms_valid"),
        ("ms_short", "ms_short_valid"),
        ("ms_global", "ms_global_valid"),
    ):
        value = result.get(key)
        valid = result.get(valid_key)
        if not isinstance(value, np.ndarray) or not isinstance(valid, np.ndarray):
            raise TypeError(f"MS75 radiometric augmentation requires numpy {key}")
        original = value.astype(np.float32, copy=False)
        reflectance = original * std + mean
        augmented = (
            reflectance * global_gain * band_gain
            + global_offset
            + band_offset
        )
        normalized = (np.clip(augmented, 0.0, 1.5) - mean) / std
        result[key] = np.where(valid[None], normalized, original).astype(
            np.float32, copy=False
        )
    return result


@dataclass(frozen=True)
class SamplingAnchor:
    scene_id: str
    row: int
    column: int
    true_class: int
    predicted_class: int = -1


@dataclass(frozen=True)
class _SamplingBlock:
    """One I/O-local training block with a fixed component/key/scene."""

    kind: str
    scene_id: str
    key: object | None = None


class BalancedPointIndex:
    """Class- or confusion-balanced points grouped by key and scene."""

    def __init__(
        self,
        groups: Mapping[object, Mapping[str, np.ndarray]],
        *,
        kind: str,
    ):
        if kind not in {"class", "confusion"}:
            raise ValueError("point index kind must be class or confusion")
        normalized: dict[object, dict[str, np.ndarray]] = {}
        for key, by_scene in groups.items():
            retained: dict[str, np.ndarray] = {}
            for scene_id, points in by_scene.items():
                values = np.asarray(points, dtype=np.int64)
                if values.ndim != 2 or values.shape[1] != 2:
                    raise ValueError("point coordinates must have shape Nx2")
                if len(values) and int(values.min()) < 0:
                    raise ValueError("point coordinates must be non-negative")
                if len(values):
                    retained[str(scene_id)] = values
            if retained:
                normalized[key] = retained
        if not normalized:
            raise ValueError(f"{kind} point index is empty for active scenes")
        self.kind = kind
        self.groups = normalized
        self.keys = tuple(sorted(normalized, key=str))
        self._keys_by_scene: dict[str, tuple[object, ...]] = {}
        scene_ids = {
            scene_id
            for by_scene in normalized.values()
            for scene_id in by_scene
        }
        for scene_id in sorted(scene_ids):
            self._keys_by_scene[scene_id] = tuple(
                key for key in self.keys if scene_id in self.groups[key]
            )

    def restricted_to_keys(self, keys: Sequence[object]) -> "BalancedPointIndex":
        """Return a new index containing exactly the requested available keys.

        The full content-addressed OOF archive remains the authenticated source;
        this in-memory view only prevents a targeted expert run from spending
        its confusion budget on the dozens of low-value sibling pairs.  The
        selected key contract is independently bound by the pair-report hash in
        the lightweight expert checkpoint.
        """

        requested = tuple(keys)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("restricted point-index keys must be non-empty and unique")
        retained = {key: self.groups[key] for key in requested if key in self.groups}
        missing = [key for key in requested if key not in self.groups]
        if missing:
            raise ValueError(
                f"requested {self.kind} keys are absent from active scenes: {missing}"
            )
        result = BalancedPointIndex(retained, kind=self.kind)
        if hasattr(self, "file_sha256"):
            result.file_sha256 = self.file_sha256
        return result

    @staticmethod
    def _scene_ids(archive: np.lib.npyio.NpzFile) -> list[str]:
        if "scene_ids" not in archive.files:
            raise ValueError("point index is missing scene_ids")
        scene_ids = [str(value) for value in archive["scene_ids"].tolist()]
        if not scene_ids or len(scene_ids) != len(set(scene_ids)):
            raise ValueError("point index has invalid scene_ids")
        for scene_id in scene_ids:
            if Path(scene_id).name != scene_id or "/" in scene_id or "\\" in scene_id:
                raise ValueError(f"unsafe point-index scene ID: {scene_id!r}")
        return scene_ids

    @staticmethod
    def _validate_scene_universe(
        scene_ids: Sequence[str],
        *,
        official_train_scene_ids: Sequence[str],
        official_test_scene_ids: Sequence[str],
    ) -> None:
        index_set = set(scene_ids)
        test_set = set(official_test_scene_ids)
        if index_set & test_set:
            raise ValueError("official test scene leaked into sampling index")
        outside = index_set - set(official_train_scene_ids)
        if outside:
            raise ValueError(
                "sampling index contains non-training scenes: "
                f"{sorted(outside)[:5]}"
            )

    @classmethod
    def from_class_npz(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None,
        active_scene_ids: Sequence[str],
        official_train_scene_ids: Sequence[str],
        official_test_scene_ids: Sequence[str],
        expected_source_split: str = "official_train",
        require_exact_active_scene_ids: bool = False,
    ) -> "BalancedPointIndex":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"missing class sampling index: {path}")
        file_sha256 = _verified_file_sha256(
            path,
            expected_sha256,
            artifact_name="class sampling index",
        )
        archive = np.load(path, allow_pickle=False)
        try:
            scene_ids = cls._scene_ids(archive)
            if "source_split" not in archive.files or str(
                np.asarray(archive["source_split"]).item()
            ) != str(expected_source_split):
                raise ValueError(
                    "class sampling index must come from "
                    f"{expected_source_split}"
                )
            cls._validate_scene_universe(
                scene_ids,
                official_train_scene_ids=official_train_scene_ids,
                official_test_scene_ids=official_test_scene_ids,
            )
            if require_exact_active_scene_ids and list(scene_ids) != list(
                active_scene_ids
            ):
                raise ValueError(
                    "class sampling index scene order must exactly match the "
                    "active clean-fold split"
                )
            active = set(active_scene_ids)
            groups: dict[int, dict[str, list[tuple[int, int]]]] = {}
            for class_id in range(24):
                name = f"class_{class_id}"
                if name not in archive.files:
                    raise ValueError(f"class sampling index is missing {name}")
                values = np.asarray(archive[name], dtype=np.int64)
                if values.ndim != 2 or values.shape[1] != 3:
                    raise ValueError(f"malformed {name}: expected Nx3")
                for scene_index, row, column in values:
                    if not 0 <= int(scene_index) < len(scene_ids):
                        raise ValueError(f"{name} contains invalid scene index")
                    if int(row) < 0 or int(column) < 0:
                        raise ValueError(f"{name} contains negative coordinate")
                    scene_id = scene_ids[int(scene_index)]
                    if scene_id in active:
                        groups.setdefault(class_id, {}).setdefault(
                            scene_id, []
                        ).append((int(row), int(column)))
        finally:
            archive.close()
        arrays = {
            key: {
                scene_id: np.asarray(points, dtype=np.int64)
                for scene_id, points in by_scene.items()
            }
            for key, by_scene in groups.items()
        }
        result = cls(arrays, kind="class")
        result.file_sha256 = file_sha256
        return result

    @classmethod
    def from_confusion_npz(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None,
        active_scene_ids: Sequence[str],
        official_train_scene_ids: Sequence[str],
        official_test_scene_ids: Sequence[str],
    ) -> "BalancedPointIndex":
        """Load a strictly training-side out-of-fold confusion index.

        Besides points/fold IDs, the archive must embed a canonical per-fold
        provenance manifest and its SHA256.  This makes the OOF assertion
        structural and machine-verifiable instead of trusting a boolean flag.
        """

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"missing OOF confusion index: {path}")
        file_sha256 = _verified_file_sha256(
            path,
            expected_sha256,
            artifact_name="OOF confusion index",
        )
        archive = np.load(path, allow_pickle=False)
        try:
            required = {
                "scene_ids",
                "source_split",
                "is_oof",
                "points",
                "fold_ids",
                "scene_fold_ids",
                "provenance_schema_version",
                "provenance_json",
                "provenance_sha256",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    f"OOF confusion index is missing keys: {sorted(missing)}"
                )
            source_split = str(np.asarray(archive["source_split"]).item())
            is_oof_value = np.asarray(archive["is_oof"]).item()
            if not isinstance(is_oof_value, (bool, np.bool_)):
                raise ValueError("OOF is_oof must be a boolean scalar")
            if source_split != "official_train_oof" or not bool(is_oof_value):
                raise ValueError(
                    "confusion index must be training-side official_train_oof"
                )
            scene_ids = cls._scene_ids(archive)
            cls._validate_scene_universe(
                scene_ids,
                official_train_scene_ids=official_train_scene_ids,
                official_test_scene_ids=official_test_scene_ids,
            )
            if set(scene_ids) != set(official_train_scene_ids):
                raise ValueError(
                    "OOF confusion index must cover exactly all official "
                    "training scenes"
                )
            schema_version = np.asarray(
                archive["provenance_schema_version"]
            ).item()
            if (
                isinstance(schema_version, bool)
                or int(schema_version)
                not in {
                    OOF_PROVENANCE_SCHEMA_VERSION,
                    CLEAN_OOF_PROVENANCE_SCHEMA_VERSION,
                }
            ):
                raise ValueError("OOF provenance schema version mismatch")
            provenance_json = str(
                np.asarray(archive["provenance_json"]).item()
            )
            provenance_sha256 = str(
                np.asarray(archive["provenance_sha256"]).item()
            )
            actual_provenance_sha256 = hashlib.sha256(
                provenance_json.encode("utf-8")
            ).hexdigest()
            if (
                not _SHA256_PATTERN.fullmatch(provenance_sha256)
                or provenance_sha256 != actual_provenance_sha256
            ):
                raise ValueError("OOF provenance JSON SHA256 mismatch")
            try:
                manifest = json.loads(provenance_json)
            except json.JSONDecodeError as error:
                raise ValueError("OOF provenance JSON is malformed") from error
            if canonical_oof_provenance_json(manifest) != provenance_json:
                raise ValueError("OOF provenance JSON is not canonical")
            if int(manifest.get("schema_version", -1)) != int(schema_version):
                raise ValueError(
                    "OOF archive provenance schema disagrees with its manifest"
                )
            scene_to_fold = validate_oof_provenance_manifest(
                manifest,
                official_train_scene_ids=official_train_scene_ids,
                official_test_scene_ids=official_test_scene_ids,
            )
            scene_folds = np.asarray(archive["scene_fold_ids"], dtype=np.int64)
            if scene_folds.shape != (len(scene_ids),):
                raise ValueError("OOF scene_fold_ids must match scene_ids")
            expected_scene_folds = np.asarray(
                [scene_to_fold[scene_id] for scene_id in scene_ids],
                dtype=np.int64,
            )
            if not np.array_equal(scene_folds, expected_scene_folds):
                raise ValueError(
                    "OOF scene_fold_ids disagree with provenance heldout folds"
                )
            points = np.asarray(archive["points"], dtype=np.int64)
            folds = np.asarray(archive["fold_ids"], dtype=np.int64)
            if points.ndim != 2 or points.shape[1] != 5:
                raise ValueError("OOF confusion points must have shape Nx5")
            if folds.shape != (len(points),) or (len(folds) and folds.min() < 0):
                raise ValueError("OOF fold_ids must be non-negative and match points")
            active = set(active_scene_ids)
            groups: dict[
                tuple[int, int], dict[str, list[tuple[int, int]]]
            ] = {}
            for point_index, (
                scene_index,
                row,
                column,
                true_class,
                predicted_class,
            ) in enumerate(points):
                if not 0 <= int(scene_index) < len(scene_ids):
                    raise ValueError("OOF confusion point has invalid scene index")
                if int(row) < 0 or int(column) < 0:
                    raise ValueError("OOF confusion point has negative coordinate")
                if not 0 <= int(true_class) < 24 or not 0 <= int(
                    predicted_class
                ) < 24:
                    raise ValueError("OOF confusion classes must be in [0,23]")
                if int(true_class) == int(predicted_class):
                    raise ValueError("OOF confusion point is not a confusion")
                if (
                    FBP_FINE_TO_FAMILY[int(true_class)]
                    != FBP_FINE_TO_FAMILY[int(predicted_class)]
                ):
                    raise ValueError(
                        "OOF confusion point must be a within-family confusion"
                    )
                if int(folds[point_index]) != int(scene_folds[int(scene_index)]):
                    raise ValueError(
                        "OOF point fold_id disagrees with its scene heldout fold"
                    )
                scene_id = scene_ids[int(scene_index)]
                if scene_id in active:
                    key = (int(true_class), int(predicted_class))
                    groups.setdefault(key, {}).setdefault(scene_id, []).append(
                        (int(row), int(column))
                    )
        finally:
            archive.close()
        arrays = {
            key: {
                scene_id: np.asarray(values, dtype=np.int64)
                for scene_id, values in by_scene.items()
            }
            for key, by_scene in groups.items()
        }
        result = cls(arrays, kind="confusion")
        result.file_sha256 = file_sha256
        return result

    def sample(
        self,
        rng: random.Random,
        *,
        preferred_scene_id: str | None = None,
    ) -> SamplingAnchor:
        # The scientific meaning of class/confusion-balanced is that the key
        # is uniform globally.  A preferred scene is only an I/O-locality hint;
        # it must never remove rare keys absent from that scene.
        key = rng.choice(self.keys)
        by_scene = self.groups[key]
        if preferred_scene_id in by_scene:
            scene_id = str(preferred_scene_id)
        else:
            scene_id = rng.choice(tuple(sorted(by_scene)))
        return self.sample_group(rng, key=key, scene_id=scene_id)

    def sample_group(
        self,
        rng: random.Random,
        *,
        key: object,
        scene_id: str,
    ) -> SamplingAnchor:
        """Draw a coordinate from a preselected block-level key and scene."""

        if key not in self.groups or scene_id not in self.groups[key]:
            raise ValueError("sampling key/scene is absent from point index")
        by_scene = self.groups[key]
        points = by_scene[scene_id]
        row, column = points[rng.randrange(len(points))]
        if self.kind == "class":
            return SamplingAnchor(
                scene_id, int(row), int(column), int(key), -1
            )
        true_class, predicted_class = key
        return SamplingAnchor(
            scene_id,
            int(row),
            int(column),
            int(true_class),
            int(predicted_class),
        )

    def scene_for_key(
        self,
        rng: random.Random,
        *,
        key: object,
        preferred_scene_id: str | None = None,
    ) -> str:
        """Choose a scene for an already globally balanced key."""

        if key not in self.groups:
            raise ValueError("unknown sampling key")
        by_scene = self.groups[key]
        if preferred_scene_id in by_scene:
            return str(preferred_scene_id)
        return str(rng.choice(tuple(sorted(by_scene))))


class MixedSamplingSchedule:
    """Deterministic worker-safe mixture with exact long-run proportions."""

    def __init__(
        self,
        *,
        scene_fraction: float = 0.60,
        class_fraction: float = 0.20,
        confusion_fraction: float = 0.20,
        seed: int = 3407,
        schedule_size: int = 1000,
    ):
        fractions = np.asarray(
            [scene_fraction, class_fraction, confusion_fraction],
            dtype=np.float64,
        )
        if not np.isfinite(fractions).all() or (fractions < 0).any():
            raise ValueError("dynamic sampling fractions must be finite and non-negative")
        if not math.isclose(float(fractions.sum()), 1.0, abs_tol=1e-8):
            raise ValueError("dynamic sampling fractions must sum to one")
        if int(schedule_size) <= 0:
            raise ValueError("schedule_size must be positive")
        raw = fractions * int(schedule_size)
        counts = np.floor(raw).astype(np.int64)
        remaining = int(schedule_size) - int(counts.sum())
        remainders = raw - counts
        order = sorted(
            range(len(SAMPLING_KINDS)),
            key=lambda index: (-float(remainders[index]), index),
        )
        for index in order[:remaining]:
            counts[index] += 1
        self.seed = int(seed)
        self.schedule_size = int(schedule_size)
        self.counts = {
            kind: int(count) for kind, count in zip(SAMPLING_KINDS, counts)
        }
        self._base = tuple(
            kind
            for kind in SAMPLING_KINDS
            for _ in range(self.counts[kind])
        )
        self._cache: dict[tuple[int, int], tuple[str, ...]] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def component(self, index: int, epoch: int = 0) -> str:
        if int(index) < 0:
            raise ValueError("sample index must be non-negative")
        cycle, position = divmod(int(index), self.schedule_size)
        key = (int(epoch), cycle)
        schedule = self._cache.get(key)
        if schedule is None:
            values = list(self._base)
            random.Random(
                self.seed + int(epoch) * 10_000_019 + cycle * 1_000_003
            ).shuffle(values)
            schedule = tuple(values)
            self._cache[key] = schedule
            # Dataset workers need only the current few cycles.
            if len(self._cache) > 8:
                oldest = next(iter(self._cache))
                self._cache.pop(oldest, None)
        return schedule[position]


class MS75MultiScaleDataset(Dataset):
    """Pure-MS training crops for the strict FBP 120/30 MS75 protocol.

    ``data.dataset_variant`` must be ``ms75_multiscale``.  Development mode
    uses a scene-disjoint subset of the official 120 scenes; final mode trains
    on all 120 and deliberately has no validation dataset.  The official 30
    scenes are available only in ``mode='test'``.
    """

    def __init__(
        self,
        config: dict,
        mode: str,
        *,
        reader_factory: Callable[[str | Path, str, dict], MS75SceneReader]
        | None = None,
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"unsupported MS75 mode: {mode}")
        self.config = config
        self.mode = mode
        data = config["data"]
        if str(data.get("dataset_variant", "")) != "ms75_multiscale":
            raise ValueError("MS75 dataset requires data.dataset_variant=ms75_multiscale")
        self.root = Path(data["root"])
        self.protocol = OfficialFBPProtocol.from_root(self.root)
        self.seed = int(config.get("experiment", {}).get("seed", 3407))
        stage = str(data.get("protocol_stage", "final"))
        if stage not in {
            "development",
            "final",
            "ms75_clean_fold0_train96",
            "dofa_full120_train",
        }:
            raise ValueError(
                "data.protocol_stage must be development, final, "
                "ms75_clean_fold0_train96, or dofa_full120_train"
            )
        if stage in {"ms75_clean_fold0_train96", "dofa_full120_train"}:
            split_names = {
                str(data.get(f"{split_mode}_split", ""))
                for split_mode in ("train", "val", "test")
            }
            expected_split = (
                "ms75_oof_seed1337_fold_0_train"
                if stage == "ms75_clean_fold0_train96"
                else "train"
            )
            if split_names != {expected_split}:
                raise ValueError(
                    f"{stage} requires the precommitted {expected_split} "
                    "loader alias"
                )
            clean_scene_ids = tuple(
                read_fbp_split(self.root, split_names.pop())
            )
            expected_count = 96 if stage == "ms75_clean_fold0_train96" else 120
            if (
                len(clean_scene_ids) != expected_count
                or len(set(clean_scene_ids)) != expected_count
                or not set(clean_scene_ids).issubset(
                    set(self.protocol.train_scene_ids)
                )
                or set(clean_scene_ids) & set(self.protocol.test_scene_ids)
                or (
                    stage == "dofa_full120_train"
                    and set(clean_scene_ids) != set(self.protocol.train_scene_ids)
                )
            ):
                raise ValueError(
                    f"{stage} split must contain exactly {expected_count} unique "
                    "official-training scenes"
                )
            # Every loader mode is intentionally confined to the authenticated
            # fitting list.  Formal fitting disables validation and automatic
            # testing; even an accidental generic loader cannot open held24 or
            # official-test imagery through these stages.
            self.scene_ids = list(clean_scene_ids)
        elif stage == "development":
            if data.get("train_split") and data.get("val_split"):
                development_train = tuple(
                    read_fbp_split(self.root, str(data["train_split"]))
                )
                development_val = tuple(
                    read_fbp_split(self.root, str(data["val_split"]))
                )
                if (
                    len(development_train) != 108
                    or len(development_val) != 12
                    or set(development_train) & set(development_val)
                    or set(development_train) | set(development_val)
                    != set(self.protocol.train_scene_ids)
                ):
                    raise ValueError(
                        "MS75 development train/val must be a disjoint "
                        "108/12 partition of the official 120 scenes"
                    )
                split = SceneDevelopmentSplit(
                    train_scene_ids=development_train,
                    dev_scene_ids=development_val,
                )
            else:
                split = make_scene_development_split(
                    self.protocol,
                    dev_scene_count=int(data.get("dev_scene_count", 12)),
                    seed=int(data.get("dev_split_seed", self.seed)),
                )
            self.scene_ids = list(
                split.train_scene_ids if mode == "train" else split.dev_scene_ids
            )
        elif mode == "train":
            self.scene_ids = list(self.protocol.train_scene_ids)
        elif mode == "val":
            raise ValueError(
                "final MS75 training uses all 120 scenes; select checkpoints in "
                "development mode before final fitting"
            )
        else:
            self.scene_ids = list(self.protocol.test_scene_ids)
        if mode == "test" and stage not in {
            "ms75_clean_fold0_train96",
            "dofa_full120_train",
        }:
            self.scene_ids = list(self.protocol.test_scene_ids)
        if set(self.scene_ids) & set(self.protocol.test_scene_ids) and mode != "test":
            raise ValueError("official test scene leaked into MS75 development")
        debug_scene_limit = data.get("debug_scene_limit")
        if debug_scene_limit is not None:
            self.scene_ids = self.scene_ids[: int(debug_scene_limit)]

        self.patch_size = int(data["patch_size"])
        self.short_context_size = int(
            data.get("short_context_size", self.patch_size * 3)
        )
        self.short_input_size = int(
            data.get("short_input_size", self.patch_size)
        )
        self.global_input_size = int(
            data.get("global_input_size", self.patch_size)
        )
        if min(
            self.patch_size,
            self.short_context_size,
            self.short_input_size,
            self.global_input_size,
        ) <= 0:
            raise ValueError("MS75 spatial sizes must be positive")
        if self.short_context_size < self.patch_size:
            raise ValueError("short_context_size cannot be smaller than patch_size")

        dynamic = data.get("dynamic_sampling", {})
        self.sampling_fractions = {
            "scene": float(dynamic.get("scene_fraction", 0.60)),
            "class": float(dynamic.get("class_fraction", 0.20)),
            "confusion": float(dynamic.get("confusion_fraction", 0.20)),
        }
        self.center_jitter = int(dynamic.get("center_jitter", 0))
        if self.center_jitter < 0:
            raise ValueError("dynamic_sampling.center_jitter must be non-negative")
        self.max_scene_repeat = int(dynamic.get("max_scene_repeat", 8))
        if self.max_scene_repeat <= 0:
            raise ValueError("dynamic_sampling.max_scene_repeat must be positive")
        self.samples_per_epoch = int(
            data.get(
                "debug_samples_per_epoch",
                data.get("samples_per_epoch", len(self.scene_ids) * 128),
            )
            if mode == "train"
            else data.get("eval_patches_per_scene", 1) * len(self.scene_ids)
        )
        if self.samples_per_epoch <= 0:
            raise ValueError("MS75 dataset length must be positive")
        block_count = math.ceil(self.samples_per_epoch / self.max_scene_repeat)
        self.schedule = MixedSamplingSchedule(
            scene_fraction=self.sampling_fractions["scene"],
            class_fraction=self.sampling_fractions["class"],
            confusion_fraction=self.sampling_fractions["confusion"],
            seed=self.seed,
            schedule_size=block_count,
        )
        self.min_labeled_fraction = float(data.get("min_labeled_fraction", 0.0))
        self.sampling_attempts = int(data.get("sampling_attempts", 16))
        if not 0 <= self.min_labeled_fraction <= 1:
            raise ValueError("min_labeled_fraction must be in [0,1]")
        if self.sampling_attempts <= 0:
            raise ValueError("sampling_attempts must be positive")

        self.class_index: BalancedPointIndex | None = None
        self.confusion_index: BalancedPointIndex | None = None
        self.class_probability_manifest: ClassProbabilityManifest | None = None
        self.class_probabilities: tuple[float, ...] | None = None
        if mode == "train":
            if self.sampling_fractions["class"] > 0:
                class_path = dynamic.get("class_index")
                if not class_path:
                    raise ValueError(
                        "class_fraction is positive but dynamic_sampling.class_index is missing"
                    )
                self.class_index = BalancedPointIndex.from_class_npz(
                    self._resolve_index(class_path),
                    expected_sha256=dynamic.get("class_index_sha256"),
                    active_scene_ids=self.scene_ids,
                    official_train_scene_ids=self.protocol.train_scene_ids,
                    official_test_scene_ids=self.protocol.test_scene_ids,
                    expected_source_split=(
                        "official_train_oof_fold"
                        if stage == "ms75_clean_fold0_train96"
                        else "official_train"
                    ),
                    require_exact_active_scene_ids=(
                        stage
                        in {"ms75_clean_fold0_train96", "dofa_full120_train"}
                    ),
                )
            if self.sampling_fractions["confusion"] > 0:
                confusion_path = dynamic.get("confusion_index")
                if not confusion_path:
                    raise ValueError(
                        "confusion_fraction is positive but "
                        "dynamic_sampling.confusion_index is missing"
                    )
                self.confusion_index = BalancedPointIndex.from_confusion_npz(
                    self._resolve_index(confusion_path),
                    expected_sha256=dynamic.get("confusion_index_sha256"),
                    active_scene_ids=self.scene_ids,
                    official_train_scene_ids=self.protocol.train_scene_ids,
                    official_test_scene_ids=self.protocol.test_scene_ids,
                )
            probability_path = dynamic.get("class_probability_manifest")
            probability_sha256 = dynamic.get("class_probability_manifest_sha256")
            if (probability_path is None) != (probability_sha256 is None):
                raise ValueError(
                    "dynamic_sampling.class_probability_manifest and its "
                    "SHA256 must be configured together"
                )
            if probability_path is not None:
                if self.class_index is None:
                    raise ValueError(
                        "class probability manifest requires a positive class arm"
                    )
                self.class_probability_manifest = load_class_probability_manifest(
                    Path(str(probability_path)),
                    expected_sha256=str(probability_sha256),
                )
                if set(self.class_index.keys) != set(range(CLASS_PROBABILITY_COUNT)):
                    raise ValueError(
                        "authenticated class probabilities require all 24 class keys"
                    )
                self.class_probabilities = tuple(
                    self.class_probability_manifest.probabilities[int(key)]
                    for key in self.class_index.keys
                )
            elif self.class_index is not None:
                # Backward-compatible legacy configuration: uniform over the
                # class keys that remain in the active scene subset.
                self.class_probabilities = tuple(
                    1.0 / len(self.class_index.keys)
                    for _ in self.class_index.keys
                )

        self.horizontal_flip_probability = float(
            data.get("horizontal_flip_probability", 0.0)
        )
        self.vertical_flip_probability = float(
            data.get("vertical_flip_probability", 0.0)
        )
        for probability in (
            self.horizontal_flip_probability,
            self.vertical_flip_probability,
        ):
            if not 0 <= probability <= 1:
                raise ValueError("MS75 flip probabilities must be in [0,1]")
        self.reader_factory = reader_factory or MS75SceneReader
        self.cache_scenes = int(data.get("cache_scenes", 2))
        if self.cache_scenes <= 0:
            raise ValueError("data.cache_scenes must be positive")
        self._readers: OrderedDict[str, MS75SceneReader] = OrderedDict()
        self.global_cache_scenes = int(
            data.get("global_cache_scenes", len(self.scene_ids))
        )
        if self.global_cache_scenes <= 0:
            raise ValueError("data.global_cache_scenes must be positive")
        self._global_thumbnail_cache: OrderedDict[
            tuple[str, int], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self.epoch = 0
        self.class_schedule_epoch = 0
        self._class_key_plan_audit: dict[str, object] | None = None
        self._eval_plan = self._make_eval_plan(data) if mode != "train" else []
        self._training_block_plan = (
            self._make_training_block_plan() if mode == "train" else []
        )

    def _resolve_index(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_readers"] = OrderedDict()
        state["_global_thumbnail_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        for reader in getattr(self, "_readers", {}).values():
            reader.close()
        self._readers = OrderedDict()
        cache = getattr(self, "_global_thumbnail_cache", None)
        if cache is not None:
            cache.clear()

    def set_epoch(
        self,
        epoch: int,
        *,
        class_schedule_epoch: int | None = None,
    ) -> None:
        """Set data RNG epoch and, optionally, the continuation-local class epoch."""

        self.epoch = int(epoch)
        self.class_schedule_epoch = int(
            epoch if class_schedule_epoch is None else class_schedule_epoch
        )
        if self.class_schedule_epoch < 0:
            raise ValueError("class_schedule_epoch must be non-negative")
        if self.mode == "train":
            self._training_block_plan = self._make_training_block_plan()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _reader(self, scene_id: str) -> MS75SceneReader:
        reader = self._readers.pop(scene_id, None)
        if reader is None:
            reader = self.reader_factory(self.root, scene_id, self.config["data"])
            if isinstance(reader, MS75SceneReader):
                reader.shared_global_thumbnail_cache = self._global_thumbnail_cache
                reader.shared_global_thumbnail_cache_size = (
                    self.global_cache_scenes
                )
        self._readers[scene_id] = reader
        while len(self._readers) > self.cache_scenes:
            _, expired = self._readers.popitem(last=False)
            expired.close()
        return reader

    def _planned_scene_for_block(self, block: int) -> str:
        cycle, position = divmod(int(block), len(self.scene_ids))
        order = list(self.scene_ids)
        random.Random(
            self.seed + self.epoch * 10_000_019 + cycle * 1_000_003
        ).shuffle(order)
        return order[position]

    def _balanced_block_keys(
        self,
        point_index: BalancedPointIndex,
        count: int,
        *,
        seed_offset: int,
    ) -> list[object]:
        """Cycle globally over all keys; counts can differ by at most one."""

        result: list[object] = []
        cycle = 0
        while len(result) < int(count):
            keys = list(point_index.keys)
            random.Random(
                self.seed
                + self.epoch * 10_000_019
                + int(seed_offset)
                + cycle * 1_000_003
            ).shuffle(keys)
            result.extend(keys)
            cycle += 1
        return result[: int(count)]

    def _weighted_fair_class_block_keys(
        self,
        point_index: BalancedPointIndex,
        count: int,
        *,
        seed_offset: int,
    ) -> tuple[list[object], dict[str, object]]:
        """Allocate class slots with deterministic, prefix-balanced deficits.

        The same scheduler handles legacy/A0 uniform and A1 weighted vectors.
        A fixed tie permutation is the only random choice; allocation itself is
        deterministic and carries its virtual counts across dataset epochs.
        """

        keys = tuple(point_index.keys)
        if not keys:
            raise ValueError("class point index has no keys")
        probabilities = self.class_probabilities
        if probabilities is None or len(probabilities) != len(keys):
            raise ValueError("class probability vector does not match class keys")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("class probability vector must sum to one")
        slots_per_epoch = self.schedule.counts["class"]
        start = self.class_schedule_epoch * slots_per_epoch
        stop = start + int(count)
        priority = list(keys)
        random.Random(self.seed + int(seed_offset)).shuffle(priority)
        tie_rank = {key: index for index, key in enumerate(priority)}
        counts = {key: 0 for key in keys}
        selected: list[object] = []
        max_abs_prefix_error = 0.0
        max_oversampling_error = 0.0
        max_undersampling_error = 0.0
        for slot in range(stop):
            prefix = slot + 1
            key = max(
                keys,
                key=lambda candidate: (
                    prefix * probabilities[keys.index(candidate)]
                    - counts[candidate],
                    -tie_rank[candidate],
                ),
            )
            counts[key] += 1
            if slot >= start:
                selected.append(key)
            for class_index, candidate in enumerate(keys):
                error = counts[candidate] - prefix * probabilities[class_index]
                max_abs_prefix_error = max(max_abs_prefix_error, abs(error))
                max_oversampling_error = max(max_oversampling_error, error)
                max_undersampling_error = max(max_undersampling_error, -error)
        current_counts = {key: selected.count(key) for key in keys}
        audit = {
            "algorithm": "largest_deficit_prefix_balanced_v1",
            "slot_offset": start,
            "slot_count": int(count),
            "global_slot_count": stop,
            "keys": list(keys),
            "probabilities": list(probabilities),
            "current_counts": [current_counts[key] for key in keys],
            "current_targets": [int(count) * value for value in probabilities],
            "cumulative_counts": [counts[key] for key in keys],
            "cumulative_targets": [stop * value for value in probabilities],
            "max_abs_prefix_error": max_abs_prefix_error,
            "max_oversampling_error": max_oversampling_error,
            "max_undersampling_error": max_undersampling_error,
        }
        return selected, audit

    def _make_training_block_plan(self) -> list[_SamplingBlock]:
        """Fix component/key/scene per block while drawing coordinates per item.

        The plan restores source-scene locality without returning to the old
        scene-first bias: class/confusion keys are globally balanced before a
        preferred scene is considered.  It is plain immutable state, so worker
        processes receive exactly the same epoch plan.
        """

        block_count = math.ceil(self.samples_per_epoch / self.max_scene_repeat)
        kinds = [
            self.schedule.component(block_index, self.epoch)
            for block_index in range(block_count)
        ]
        positions_by_kind = {
            kind: [index for index, value in enumerate(kinds) if value == kind]
            for kind in ("class", "confusion")
        }
        keys_by_position: dict[int, object] = {}
        self._class_key_plan_audit = None
        for kind, point_index, seed_offset in (
            ("class", self.class_index, 31_337),
            ("confusion", self.confusion_index, 79_919),
        ):
            positions = positions_by_kind[kind]
            if positions:
                if point_index is None:
                    raise AssertionError(f"{kind} blocks require a point index")
                if kind == "class":
                    keys, self._class_key_plan_audit = (
                        self._weighted_fair_class_block_keys(
                            point_index,
                            len(positions),
                            seed_offset=seed_offset,
                        )
                    )
                else:
                    keys = self._balanced_block_keys(
                        point_index,
                        len(positions),
                        seed_offset=seed_offset,
                    )
                keys_by_position.update(zip(positions, keys))

        plan: list[_SamplingBlock] = []
        for block_index, kind in enumerate(kinds):
            preferred_scene = self._planned_scene_for_block(block_index)
            if kind == "scene":
                plan.append(_SamplingBlock(kind, preferred_scene))
                continue
            point_index = (
                self.class_index if kind == "class" else self.confusion_index
            )
            assert point_index is not None
            key = keys_by_position[block_index]
            rng = random.Random(
                self.seed
                + self.epoch * 10_000_019
                + block_index * 65_537
                + (101 if kind == "class" else 211)
            )
            scene_id = point_index.scene_for_key(
                rng,
                key=key,
                preferred_scene_id=preferred_scene,
            )
            plan.append(_SamplingBlock(kind, scene_id, key))
        return plan

    @staticmethod
    def _plan_fingerprint(value: object) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def sampling_plan_audit(
        self,
        *,
        consumed_samples: int | None = None,
    ) -> dict[str, object]:
        """Return serializable fingerprints and realized class allocation.

        ``consumed_samples`` lets an exact successful-update stop report a
        partial terminal epoch rather than claiming every planned crop ran.
        """

        if self.mode != "train":
            raise ValueError("sampling plan audit is available only in train mode")
        consumed = len(self) if consumed_samples is None else int(consumed_samples)
        if not 0 <= consumed <= len(self):
            raise ValueError("consumed_samples must be within the dataset epoch")
        full_plan = [
            [index, block.kind, block.scene_id, block.key]
            for index, block in enumerate(self._training_block_plan)
        ]
        # A paired A0/A1 run is required to match this fingerprint.  Class
        # scene/key are deliberately omitted because they are downstream of
        # the sole treatment variable, the class probability vector.
        paired_invariant_plan = [
            [index, block.kind]
            if block.kind == "class"
            else [index, block.kind, block.scene_id, block.key]
            for index, block in enumerate(self._training_block_plan)
        ]
        manifest = self.class_probability_manifest
        realized_kind_samples = {kind: 0 for kind in SAMPLING_KINDS}
        class_keys = tuple(self.class_index.keys) if self.class_index else ()
        realized_class_crop_counts = {key: 0 for key in class_keys}
        realized_class_blocks_touched = {key: 0 for key in class_keys}
        for block_index, block in enumerate(self._training_block_plan):
            first_sample = block_index * self.max_scene_repeat
            sample_count = min(
                self.max_scene_repeat,
                max(0, consumed - first_sample),
            )
            if sample_count <= 0:
                break
            realized_kind_samples[block.kind] += sample_count
            if block.kind == "class":
                realized_class_crop_counts[block.key] += sample_count
                realized_class_blocks_touched[block.key] += 1
        return {
            "epoch": self.epoch,
            "class_schedule_epoch": self.class_schedule_epoch,
            "block_size": self.max_scene_repeat,
            "block_count": len(self._training_block_plan),
            "consumed_samples": consumed,
            "realized_kind_samples": realized_kind_samples,
            "realized_class_keys": list(class_keys),
            "realized_class_crop_counts": [
                realized_class_crop_counts[key] for key in class_keys
            ],
            "realized_class_blocks_touched": [
                realized_class_blocks_touched[key] for key in class_keys
            ],
            "class_probability_arm": manifest.arm if manifest else "legacy_uniform",
            "class_probability_file_sha256": (
                manifest.file_sha256 if manifest else None
            ),
            "class_probability_manifest_sha256": (
                manifest.manifest_sha256 if manifest else None
            ),
            "class_index_file_sha256": (
                self.class_index.file_sha256 if self.class_index else None
            ),
            "confusion_index_file_sha256": (
                self.confusion_index.file_sha256 if self.confusion_index else None
            ),
            "class_key_allocation": dict(self._class_key_plan_audit or {}),
            "full_plan_sha256": self._plan_fingerprint(full_plan),
            "paired_invariant_plan_sha256": self._plan_fingerprint(
                paired_invariant_plan
            ),
        }

    def _random_window(
        self, reader: MS75SceneReader, rng: random.Random
    ) -> Window:
        maximum_left = reader.width - self.patch_size
        maximum_top = reader.height - self.patch_size
        if maximum_left < 0 or maximum_top < 0:
            raise ValueError(
                f"scene {reader.record.scene_id} is smaller than patch_size"
            )
        return Window(
            rng.randint(0, maximum_left),
            rng.randint(0, maximum_top),
            self.patch_size,
            self.patch_size,
        )

    def _acceptable_random_window(
        self, reader: MS75SceneReader, rng: random.Random
    ) -> Window:
        best: Window | None = None
        best_fraction = -1.0
        for _ in range(self.sampling_attempts):
            candidate = self._random_window(reader, rng)
            _, supervised = reader.read_label_valid(candidate)
            fraction = float(supervised.mean())
            if fraction > best_fraction:
                best = candidate
                best_fraction = fraction
            if fraction >= self.min_labeled_fraction:
                return candidate
        assert best is not None
        return best

    def _window_at_anchor(
        self,
        reader: MS75SceneReader,
        anchor: SamplingAnchor,
        rng: random.Random,
    ) -> Window:
        row = int(anchor.row)
        column = int(anchor.column)
        if self.center_jitter:
            row += rng.randint(-self.center_jitter, self.center_jitter)
            column += rng.randint(-self.center_jitter, self.center_jitter)
        left = min(
            max(column - self.patch_size // 2, 0),
            reader.width - self.patch_size,
        )
        top = min(
            max(row - self.patch_size // 2, 0),
            reader.height - self.patch_size,
        )
        if left < 0 or top < 0:
            raise ValueError(
                f"scene {anchor.scene_id} is smaller than patch_size"
            )
        return Window(left, top, self.patch_size, self.patch_size)

    def _make_eval_plan(self, data: dict) -> list[tuple[str, Window]]:
        expected_height, expected_width = (
            int(value) for value in data.get("expected_hw", [6908, 7300])
        )
        if expected_height < self.patch_size or expected_width < self.patch_size:
            raise ValueError("expected_hw is smaller than patch_size")
        count = int(data.get("eval_patches_per_scene", 1))
        plan: list[tuple[str, Window]] = []
        for scene_index, scene_id in enumerate(self.scene_ids):
            rng = random.Random(self.seed + (scene_index + 1) * 100_003)
            for _ in range(count):
                plan.append(
                    (
                        scene_id,
                        Window(
                            rng.randint(0, expected_width - self.patch_size),
                            rng.randint(0, expected_height - self.patch_size),
                            self.patch_size,
                            self.patch_size,
                        ),
                    )
                )
        return plan

    def __getitem__(self, index: int) -> dict[str, object]:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        rng = random.Random(
            self.seed + self.epoch * 10_000_019 + int(index) * 65_537
        )
        true_class = predicted_class = -1
        if self.mode == "train":
            block = self._training_block_plan[
                int(index) // self.max_scene_repeat
            ]
            kind = block.kind
            if kind == "class":
                assert self.class_index is not None
                anchor = self.class_index.sample_group(
                    rng, key=block.key, scene_id=block.scene_id
                )
            elif kind == "confusion":
                assert self.confusion_index is not None
                anchor = self.confusion_index.sample_group(
                    rng, key=block.key, scene_id=block.scene_id
                )
            else:
                anchor = None
            scene_id = block.scene_id
            reader = self._reader(scene_id)
            window = (
                self._window_at_anchor(reader, anchor, rng)
                if anchor is not None
                else self._acceptable_random_window(reader, rng)
            )
            if anchor is not None:
                true_class = anchor.true_class
                predicted_class = anchor.predicted_class
        else:
            kind = "evaluation"
            scene_id, window = self._eval_plan[int(index)]
            reader = self._reader(scene_id)

        sample: dict[str, object] = reader.read_ms75_patch(
            window,
            short_context_size=self.short_context_size,
            short_input_size=self.short_input_size,
            global_input_size=self.global_input_size,
        )
        horizontal = (
            self.mode == "train"
            and rng.random() < self.horizontal_flip_probability
        )
        vertical = (
            self.mode == "train"
            and rng.random() < self.vertical_flip_probability
        )
        if horizontal or vertical:
            sample = synchronized_ms75_flip(
                sample, horizontal=horizontal, vertical=vertical
            )
        if self.mode == "train":
            sample = synchronized_ms75_spectral_augmentation(
                sample,
                data_config=self.config["data"],
                rng=rng,
            )

        tensor_keys = {
            "ms": torch.float32,
            "ms_short": torch.float32,
            "ms_global": torch.float32,
            "ms_valid": torch.bool,
            "ms_short_valid": torch.bool,
            "ms_global_valid": torch.bool,
            "target": torch.long,
            "supervised": torch.bool,
            "local_box_short": torch.float32,
            "local_box_global": torch.float32,
        }
        result: dict[str, object] = {}
        for key, dtype in tensor_keys.items():
            value = sample[key]
            if isinstance(value, torch.Tensor):
                result[key] = value.to(dtype=dtype)
            else:
                result[key] = torch.from_numpy(np.asarray(value)).to(dtype=dtype)
        result.update(
            {
                "scene_id": scene_id,
                "left": int(window.col_off),
                "top": int(window.row_off),
                "sampling_kind": kind,
                "anchor_true_class": int(true_class),
                "anchor_predicted_class": int(predicted_class),
                "flipped_horizontal": bool(horizontal),
                "flipped_vertical": bool(vertical),
            }
        )
        return result


__all__ = [
    "BalancedPointIndex",
    "ClassProbabilityManifest",
    "MS75MultiScaleDataset",
    "MS75SceneReader",
    "MixedSamplingSchedule",
    "OfficialFBPProtocol",
    "SamplingAnchor",
    "SceneDevelopmentSplit",
    "canonical_class_probability_json",
    "load_class_probability_manifest",
    "make_scene_development_split",
    "synchronized_ms75_flip",
    "synchronized_ms75_spectral_augmentation",
    "transform_normalized_box_for_flip",
]
