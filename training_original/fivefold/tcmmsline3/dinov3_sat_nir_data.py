"""Strict train96 sampling for the DINOv3-SAT/NIR development expert.

The three label-side arms share one crop budget: 50% ordinary scene blocks,
30% rare-class blocks and 20% known-hard-class blocks.  Both targeted arms
draw only from the authenticated fold0-train96 class-coordinate index.  No
teacher prediction or confusion archive is consulted.  Blocks contain 16
crops and are deterministically reordered so every 64-crop optimizer update
sees four distinct scenes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Sampler, default_collate

from .data import FBPSamplingIndex, _extract_boundless, read_fbp_split, validate_fbp_splits
from .dinov3_sat_nir_protocol import (
    DINOV3_SAT_NIR_CLASS_INDEX,
    DINOV3_SAT_NIR_CLASS_INDEX_SHA256,
    DINOV3_SAT_NIR_FOLD0_HELD_SPLIT,
    DINOV3_SAT_NIR_FOLD0_SPLIT,
    DINOV3_SAT_NIR_HARD_CLASS_IDS,
    DINOV3_SAT_NIR_RARE_CLASS_IDS,
)
from .ms75_data import (
    BalancedPointIndex,
    MS75MultiScaleDataset,
    synchronized_ms75_flip,
)
from .runtime import atomic_json_dump


DINOV3_SAT_NIR_SAMPLING_ARMS = ("scene", "rare", "hard")


@dataclass(frozen=True)
class _DINOv3SATNIRSamplingBlock:
    """One immutable 16-crop source-local sampling block."""

    kind: str
    scene_id: str
    key: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def four_scene_block_order(
    scene_ids: Sequence[str], *, block_size: int = 16
) -> tuple[int, ...]:
    """Reorder complete blocks into four-distinct-scene optimizer updates."""

    scenes = tuple(str(value) for value in scene_ids)
    if int(block_size) != 16:
        raise ValueError("DINOv3SATNIR requires exactly 16 crops per scene block")
    if not scenes or len(scenes) % 4:
        raise ValueError("scene-block count must be a positive multiple of four")
    queues: dict[str, deque[int]] = defaultdict(deque)
    for block_index, scene_id in enumerate(scenes):
        queues[scene_id].append(block_index)
    update_count = len(scenes) // 4
    if len(queues) < 4 or max(map(len, queues.values())) > update_count:
        raise RuntimeError("sampling plan cannot form four-distinct-scene updates")
    block_order: list[int] = []
    for _ in range(update_count):
        available = [scene_id for scene_id, queue in queues.items() if queue]
        available.sort(
            key=lambda scene_id: (-len(queues[scene_id]), queues[scene_id][0], scene_id)
        )
        if len(available) < 4:
            raise RuntimeError("sampling plan exhausted distinct scenes")
        selected = [queues[scene_id].popleft() for scene_id in available[:4]]
        if len({scenes[index] for index in selected}) != 4:
            raise AssertionError("optimizer update contains a repeated scene")
        block_order.extend(selected)
    sample_order = tuple(
        sample_index
        for block_index in block_order
        for sample_index in range(
            block_index * int(block_size), (block_index + 1) * int(block_size)
        )
    )
    expected = len(scenes) * int(block_size)
    if len(sample_order) != expected or set(sample_order) != set(range(expected)):
        raise AssertionError("scene-block reorder lost or duplicated crops")
    return sample_order


def exact_sampling_arm_schedule(
    block_count: int,
    *,
    scene_fraction: float,
    rare_fraction: float,
    hard_fraction: float,
    seed: int,
) -> tuple[str, ...]:
    """Create a deterministic complete-epoch 50/30/20 block schedule.

    Largest-remainder apportionment is used for short probe epochs.  Formal
    and ten-update smoke epochs divide exactly and therefore realize the
    configured fractions without rounding.
    """

    block_count = int(block_count)
    if block_count <= 0:
        raise ValueError("DINOv3SATNIR block_count must be positive")
    fractions = np.asarray(
        [scene_fraction, rare_fraction, hard_fraction], dtype=np.float64
    )
    if not np.isfinite(fractions).all() or (fractions < 0).any():
        raise ValueError("DINOv3SATNIR sampling fractions must be non-negative")
    if not math.isclose(float(fractions.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("DINOv3SATNIR sampling fractions must sum to one")
    raw = fractions * block_count
    counts = np.floor(raw).astype(np.int64)
    remaining = block_count - int(counts.sum())
    remainders = raw - counts
    order = sorted(
        range(len(DINOV3_SAT_NIR_SAMPLING_ARMS)),
        key=lambda index: (-float(remainders[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    schedule = [
        kind
        for kind, count in zip(DINOV3_SAT_NIR_SAMPLING_ARMS, counts)
        for _ in range(int(count))
    ]
    random.Random(int(seed)).shuffle(schedule)
    if len(schedule) != block_count:
        raise AssertionError("DINOv3SATNIR arm apportionment lost blocks")
    return tuple(schedule)


def sampling_order_audit(
    blocks: Sequence[object],
    sample_order: Sequence[int],
    *,
    epoch: int,
    physical_batch: int = 1,
) -> dict[str, Any]:
    """Prove arm counts and four-scene composition for one complete epoch."""

    physical_batch = int(physical_batch)
    if physical_batch not in {1, 2, 4, 8, 16} or 64 % physical_batch:
        raise ValueError("invalid DINOv3SATNIR physical batch")
    accumulation_steps = 64 // physical_batch
    if len(sample_order) != len(blocks) * 16 or len(sample_order) % 64:
        raise ValueError("sampling order does not contain complete 64-crop updates")
    block_order = [int(sample_order[offset]) // 16 for offset in range(0, len(sample_order), 16)]
    if len(block_order) != len(set(block_order)):
        raise ValueError("sampling order repeats a block")
    arm_blocks = {kind: 0 for kind in DINOV3_SAT_NIR_SAMPLING_ARMS}
    ordered_manifest: list[list[object]] = []
    for block_index in block_order:
        block = blocks[block_index]
        kind = str(getattr(block, "kind"))
        scene_id = str(getattr(block, "scene_id"))
        if kind not in arm_blocks:
            raise ValueError(f"unexpected sampling arm: {kind}")
        arm_blocks[kind] += 1
        ordered_manifest.append([block_index, kind, scene_id, getattr(block, "key")])
    update_scenes: list[list[str]] = []
    for offset in range(0, len(block_order), 4):
        scenes = [str(getattr(blocks[index], "scene_id")) for index in block_order[offset : offset + 4]]
        if len(set(scenes)) != 4:
            raise ValueError("sampling audit found a repeated scene within an update")
        update_scenes.append(scenes)
    # Expand the exact DataLoader physical batches, not merely abstract scene
    # blocks. Every batch must stay within one scene/arm and every accumulation
    # group must still realize four scenes x sixteen crops.
    physical_batches = [
        sample_order[offset : offset + physical_batch]
        for offset in range(0, len(sample_order), physical_batch)
    ]
    for batch in physical_batches:
        block_indices = {int(sample_index) // 16 for sample_index in batch}
        if len(block_indices) != 1:
            raise ValueError("a physical batch crossed a scene/sampling-kind block")
    for offset in range(0, len(physical_batches), accumulation_steps):
        expanded = [
            int(sample_index)
            for batch in physical_batches[offset : offset + accumulation_steps]
            for sample_index in batch
        ]
        scenes = [str(getattr(blocks[index // 16], "scene_id")) for index in expanded]
        kinds = [str(getattr(blocks[index // 16], "kind")) for index in expanded]
        if len(expanded) != 64 or sorted(scenes.count(scene) for scene in set(scenes)) != [16] * 4:
            raise ValueError("batch-expanded update is not four scenes x sixteen crops")
        for batch_offset in range(0, 64, physical_batch):
            if len(set(kinds[batch_offset : batch_offset + physical_batch])) != 1:
                raise ValueError("physical batch mixed sampling kinds")
    encoded = json.dumps(
        ordered_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "schema_version": 2,
        "role": "dinov3_sat_nir_train96_label_side_sampling_plan",
        "epoch": int(epoch),
        "sample_count": len(sample_order),
        "block_size": 16,
        "block_count": len(block_order),
        "effective_crops_per_update": 64,
        "physical_batch": physical_batch,
        "gradient_accumulation_steps": accumulation_steps,
        "optimizer_update_count": len(update_scenes),
        "all_updates_four_distinct_scenes": True,
        "batch_expanded_scene_and_kind_contract": True,
        "teacher_prediction_artifact_used": False,
        "sampling_coordinates_train96_labels_only": True,
        "arm_block_counts": arm_blocks,
        "arm_crop_counts": {kind: count * 16 for kind, count in arm_blocks.items()},
        "ordered_plan_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validate_lightweight_training_identity(dataset: MS75MultiScaleDataset) -> None:
    """Reject a wrong fold, class index, or legacy confusion arm."""

    if len(dataset.scene_ids) != 96 or len(set(dataset.scene_ids)) != 96:
        raise ValueError("DINOv3SATNIR fold0 training split must contain 96 scenes")
    if dataset.class_index is None:
        raise ValueError("DINOv3SATNIR requires the fold0-train96 class index")
    if dataset.class_index.file_sha256 != DINOV3_SAT_NIR_CLASS_INDEX_SHA256:
        raise ValueError("DINOv3SATNIR class index identity changed")
    if getattr(dataset, "confusion_index", None) is not None:
        raise ValueError("DINOv3SATNIR rejects every teacher/confusion point index")
    if int(dataset.max_scene_repeat) != 16:
        raise ValueError("DINOv3SATNIR sampling blocks must contain 16 crops")


def _repair_single_update_probe_scenes(
    blocks: Sequence[object],
    *,
    scene_ids: Sequence[str],
    class_index: BalancedPointIndex,
    seed: int,
) -> list[object]:
    """Make the four-block probe a real four-distinct-scene update."""

    if len(blocks) != 4:
        return list(blocks)
    choices: list[list[str]] = []
    for block_index, block in enumerate(blocks):
        kind = str(getattr(block, "kind"))
        key = getattr(block, "key")
        if kind == "scene":
            candidates = list(scene_ids)
        else:
            if kind not in {"rare", "hard"}:
                raise ValueError(f"unexpected targeted sampling arm: {kind}")
            candidates = list(class_index.groups[key])
        random.Random(int(seed) + block_index * 65_537).shuffle(candidates)
        original = str(getattr(block, "scene_id"))
        choices.append(
            [original, *[str(value) for value in candidates if str(value) != original]]
        )

    assignment: list[str] = []

    def solve(position: int, used: set[str]) -> bool:
        if position == len(choices):
            return True
        for candidate in choices[position]:
            if candidate in used:
                continue
            assignment.append(candidate)
            if solve(position + 1, {*used, candidate}):
                return True
            assignment.pop()
        return False

    if not solve(0, set()):
        raise RuntimeError("one-update probe cannot realize four distinct scenes")
    result = [replace(block, scene_id=scene) for block, scene in zip(blocks, assignment)]
    if len({str(getattr(block, "scene_id")) for block in result}) != 4:
        raise AssertionError("one-update probe scene repair failed")
    return result


def _validate_coordinate_labels(
    *,
    root: Path,
    train_ids: Sequence[str],
    class_index: FBPSamplingIndex,
) -> tuple[int, ...]:
    """Verify every label-only class anchor against train96 labels once."""

    per_class_points = [0] * 24
    for scene_id in train_ids:
        label_path = root / "labels" / f"{scene_id}.png"
        with Image.open(label_path) as source:
            if source.mode != "L":
                raise ValueError(f"FBP label must be indexed grayscale: {label_path}")
            label = np.asarray(source, dtype=np.uint8)
        height, width = label.shape
        for class_id, coordinates in class_index.by_scene[scene_id].items():
            if not 0 < len(coordinates) <= 64:
                raise ValueError("class index exceeds its train-only per-scene cap")
            rows, columns = coordinates[:, 0], coordinates[:, 1]
            if (
                np.any(rows >= height)
                or np.any(columns >= width)
                or np.any(label[rows, columns] != int(class_id) + 1)
            ):
                raise ValueError("class-index coordinate disagrees with train96 label")
            per_class_points[int(class_id)] += len(coordinates)
    if any(count <= 0 for count in per_class_points):
        raise ValueError("fold0-train96 class index must cover all 24 classes")
    return tuple(per_class_points)


def audit_dinov3_sat_nir_data_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the 96/24 split and label-only train96 sampling index."""

    data = config["data"]
    root = Path(str(data["root"])).expanduser().resolve()
    canonical = validate_fbp_splits(root)
    train_ids = read_fbp_split(root, DINOV3_SAT_NIR_FOLD0_SPLIT)
    held_ids = read_fbp_split(root, DINOV3_SAT_NIR_FOLD0_HELD_SPLIT)
    if len(train_ids) != 96 or len(held_ids) != 24:
        raise ValueError("DINOv3SATNIR fold0 must be an exact 96/24 scene split")
    if set(train_ids) & set(held_ids):
        raise ValueError("DINOv3SATNIR fold0 train96 overlaps held24")
    if set(train_ids) | set(held_ids) != set(canonical["train"]):
        raise ValueError("DINOv3SATNIR fold0 is not a partition of official train120")
    if (set(train_ids) | set(held_ids)) & set(canonical["test"]):
        raise ValueError("DINOv3SATNIR development split overlaps official30")

    class_path = (root / DINOV3_SAT_NIR_CLASS_INDEX).resolve()
    if _sha256(class_path) != DINOV3_SAT_NIR_CLASS_INDEX_SHA256:
        raise ValueError("DINOv3SATNIR train-only class index SHA256 changed")
    class_index = FBPSamplingIndex(class_path)
    if (
        class_index.source_split != "official_train_oof_fold"
        or class_index.scene_ids != list(train_ids)
        or class_index.seed != 1337
        or class_index.stride != 16
        or any(not class_index.scenes_by_class[class_id] for class_id in range(24))
    ):
        raise ValueError("DINOv3SATNIR class index is not exact fold0 train96 provenance")
    per_class_points = _validate_coordinate_labels(
        root=root,
        train_ids=train_ids,
        class_index=class_index,
    )
    split_paths = {
        "train96": root / "splits" / f"{DINOV3_SAT_NIR_FOLD0_SPLIT}.txt",
        "held24": root / "splits" / f"{DINOV3_SAT_NIR_FOLD0_HELD_SPLIT}.txt",
        "official_train120": root / "splits" / "train.txt",
        "official_test30": root / "splits" / "test.txt",
    }
    return {
        "data_root": str(root),
        "train_scene_count": len(train_ids),
        "heldout_scene_count": len(held_ids),
        "official_test_scene_count": len(canonical["test"]),
        "official30_split_metadata_read": True,
        "official30_rasters_or_labels_read": False,
        "train_scene_ids_sha256": hashlib.sha256(
            ("\n".join(train_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "heldout_scene_ids_sha256": hashlib.sha256(
            ("\n".join(held_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "split_file_sha256": {name: _sha256(path) for name, path in split_paths.items()},
        "class_index": str(class_path),
        "class_index_sha256": DINOV3_SAT_NIR_CLASS_INDEX_SHA256,
        "class_index_all_24_classes": True,
        "sampling_index_teacher_predictions_used": False,
        "sampling_index_scene_ids_exact_train96": True,
        "rare_class_ids": list(DINOV3_SAT_NIR_RARE_CLASS_IDS),
        "hard_class_ids": list(DINOV3_SAT_NIR_HARD_CLASS_IDS),
        "rare_anchor_point_count": sum(
            per_class_points[class_id] for class_id in DINOV3_SAT_NIR_RARE_CLASS_IDS
        ),
        "hard_anchor_point_count": sum(
            per_class_points[class_id] for class_id in DINOV3_SAT_NIR_HARD_CLASS_IDS
        ),
        "class_anchor_point_counts": list(per_class_points),
        "sampling_coordinate_labels_verified": True,
        "held24_labels_or_rasters_read_for_sampling": False,
    }


def _augment_local_spectral(
    sample: dict[str, np.ndarray], *, data: Mapping[str, Any], rng: random.Random
) -> dict[str, np.ndarray]:
    """Apply the shared reflectance augmentation to the local view only."""

    spectral = data.get("spectral_augmentation", {})
    if not isinstance(spectral, Mapping) or not bool(spectral.get("enabled", False)):
        return sample
    if rng.random() >= float(spectral.get("probability", 0.0)):
        return sample

    def interval(name: str, default: tuple[float, float]) -> tuple[float, float]:
        values = spectral.get(name, default)
        if not isinstance(values, Sequence) or len(values) != 2:
            raise ValueError(f"spectral_augmentation.{name} must contain two values")
        lower, upper = (float(value) for value in values)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"invalid spectral_augmentation.{name}")
        return lower, upper

    normalization = data["normalization"]["ms"]
    mean = np.asarray(normalization["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(normalization["std"], dtype=np.float32)[:, None, None]
    global_gain = rng.uniform(*interval("global_gain_range", (1.0, 1.0)))
    global_offset = rng.uniform(*interval("global_offset_range", (0.0, 0.0)))
    band_gain_range = interval("band_gain_range", (1.0, 1.0))
    band_offset_range = interval("band_offset_range", (0.0, 0.0))
    band_gain = np.asarray(
        [rng.uniform(*band_gain_range) for _ in range(4)], dtype=np.float32
    )[:, None, None]
    band_offset = np.asarray(
        [rng.uniform(*band_offset_range) for _ in range(4)], dtype=np.float32
    )[:, None, None]
    original = sample["ms"].astype(np.float32, copy=False)
    reflectance = original * std + mean
    augmented = reflectance * global_gain * band_gain + global_offset + band_offset
    normalized = (np.clip(augmented, 0.0, 1.5) - mean) / std
    sample["ms"] = np.where(sample["ms_valid"][None], normalized, original).astype(
        np.float32, copy=False
    )
    return sample


class DINOv3SATNIRDataset(MS75MultiScaleDataset):
    """Local-only reader with a DINO-specific label-side block plan."""

    def __init__(self, config: dict, mode: str) -> None:
        if mode != "train":
            raise ValueError("DINOv3SATNIR fitting loader is train96-only")
        planner = copy.deepcopy(config)
        data = planner["data"]
        configured_dynamic = copy.deepcopy(data["dynamic_sampling"])
        forbidden = {
            "class_fraction",
            "confusion_fraction",
            "confusion_index",
            "confusion_index_sha256",
        }
        if forbidden & set(configured_dynamic):
            raise ValueError("DINOv3SATNIR rejects legacy teacher/confusion sampling")
        data["dataset_variant"] = "ms75_multiscale"
        data["protocol_stage"] = "ms75_clean_fold0_train96"
        for split_name in ("train_split", "val_split", "test_split"):
            data[split_name] = DINOV3_SAT_NIR_FOLD0_SPLIT
        data["short_context_size"] = 512
        data["short_input_size"] = 512
        data["global_input_size"] = 32
        data["horizontal_flip_probability"] = 0.5
        data["vertical_flip_probability"] = 0.5
        # The parent is reused only for strict scene discovery/readers.  Its
        # generic class/confusion planner is disabled; the authenticated
        # train96 class index and the rare/hard plan are installed below.
        data["dynamic_sampling"] = {
            "scene_fraction": 1.0,
            "class_fraction": 0.0,
            "confusion_fraction": 0.0,
            "center_jitter": int(configured_dynamic["center_jitter"]),
            "max_scene_repeat": int(configured_dynamic["max_scene_repeat"]),
        }
        super().__init__(planner, mode)
        self.config["data"]["dynamic_sampling"] = configured_dynamic
        self.class_index = BalancedPointIndex.from_class_npz(
            self._resolve_index(configured_dynamic["class_index"]),
            expected_sha256=str(configured_dynamic["class_index_sha256"]),
            active_scene_ids=self.scene_ids,
            official_train_scene_ids=self.protocol.train_scene_ids,
            official_test_scene_ids=self.protocol.test_scene_ids,
            # This is the immutable legacy metadata value of the label-only
            # fold0 archive; it does not denote teacher predictions.
            expected_source_split="official_train_oof_fold",
            require_exact_active_scene_ids=True,
        )
        _validate_lightweight_training_identity(self)
        rare_ids = tuple(int(value) for value in data["rare_class_ids"])
        hard_ids = tuple(int(value) for value in data["hard_class_ids"])
        keys = tuple(int(value) for value in self.class_index.keys)
        if not set(rare_ids).issubset(keys) or not set(hard_ids).issubset(keys):
            raise ValueError("DINOv3SATNIR targeted classes are absent from the class index")
        if set(rare_ids) & set(hard_ids):
            raise ValueError("DINOv3SATNIR rare and known-hard classes must be disjoint")
        self.rare_class_ids = rare_ids
        self.hard_class_ids = hard_ids
        self._dinov3_sat_sampling_fractions = {
            "scene": float(configured_dynamic["scene_fraction"]),
            "rare": float(configured_dynamic["rare_fraction"]),
            "hard": float(configured_dynamic["hard_fraction"]),
        }
        self.sampling_fractions = dict(self._dinov3_sat_sampling_fractions)
        self.set_epoch(0)

    def _balanced_subset_keys(
        self, class_ids: Sequence[int], count: int, *, seed_offset: int
    ) -> list[int]:
        """Balance one targeted arm over its fixed class set within an epoch."""

        result: list[int] = []
        cycle = 0
        while len(result) < int(count):
            keys = list(int(value) for value in class_ids)
            random.Random(
                self.seed
                + self.class_schedule_epoch * 10_000_019
                + int(seed_offset)
                + cycle * 1_000_003
            ).shuffle(keys)
            result.extend(keys)
            cycle += 1
        return result[: int(count)]

    def _make_dinov3_sat_training_block_plan(self) -> list[_DINOv3SATNIRSamplingBlock]:
        if self.samples_per_epoch % self.max_scene_repeat:
            raise ValueError("DINOv3SATNIR epoch must contain complete 16-crop blocks")
        block_count = self.samples_per_epoch // self.max_scene_repeat
        kinds = exact_sampling_arm_schedule(
            block_count,
            scene_fraction=self._dinov3_sat_sampling_fractions["scene"],
            rare_fraction=self._dinov3_sat_sampling_fractions["rare"],
            hard_fraction=self._dinov3_sat_sampling_fractions["hard"],
            seed=self.seed + self.epoch * 10_000_019,
        )
        keys_by_position: dict[int, int] = {}
        for kind, class_ids, seed_offset in (
            ("rare", self.rare_class_ids, 31_337),
            ("hard", self.hard_class_ids, 79_919),
        ):
            positions = [index for index, value in enumerate(kinds) if value == kind]
            keys = self._balanced_subset_keys(
                class_ids, len(positions), seed_offset=seed_offset
            )
            keys_by_position.update(zip(positions, keys))

        plan: list[_DINOv3SATNIRSamplingBlock] = []
        for block_index, kind in enumerate(kinds):
            preferred_scene = self._planned_scene_for_block(block_index)
            if kind == "scene":
                plan.append(_DINOv3SATNIRSamplingBlock(kind, preferred_scene))
                continue
            key = keys_by_position[block_index]
            rng = random.Random(
                self.seed
                + self.epoch * 10_000_019
                + block_index * 65_537
                + (101 if kind == "rare" else 211)
            )
            scene_id = self.class_index.scene_for_key(
                rng, key=key, preferred_scene_id=preferred_scene
            )
            plan.append(_DINOv3SATNIRSamplingBlock(kind, scene_id, key))
        if len(plan) != block_count:
            raise AssertionError("DINOv3SATNIR block planner lost blocks")
        return plan

    def set_epoch(
        self, epoch: int, *, class_schedule_epoch: int | None = None
    ) -> None:
        self.epoch = int(epoch)
        self.class_schedule_epoch = int(
            epoch if class_schedule_epoch is None else class_schedule_epoch
        )
        if self.class_schedule_epoch < 0:
            raise ValueError("class_schedule_epoch must be non-negative")
        self._training_block_plan = self._make_dinov3_sat_training_block_plan()
        if len(self._training_block_plan) == 4:
            self._training_block_plan = _repair_single_update_probe_scenes(
                self._training_block_plan,
                scene_ids=self.scene_ids,
                class_index=self.class_index,
                seed=self.seed + int(epoch) * 10_000_019,
            )

    def __getitem__(self, index: int) -> dict[str, object]:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        rng = random.Random(self.seed + self.epoch * 10_000_019 + int(index) * 65_537)
        block = self._training_block_plan[int(index) // self.max_scene_repeat]
        if block.kind in {"rare", "hard"}:
            anchor = self.class_index.sample_group(
                rng, key=block.key, scene_id=block.scene_id
            )
        else:
            anchor = None
        reader = self._reader(block.scene_id)
        window = (
            self._window_at_anchor(reader, anchor, rng)
            if anchor is not None
            else self._acceptable_random_window(reader, rng)
        )
        raw, valid = _extract_boundless(
            reader.image,
            left=int(window.col_off),
            top=int(window.row_off),
            width=self.patch_size,
            height=self.patch_size,
        )
        target, supervised = reader.read_label_valid(window)
        sample: dict[str, np.ndarray] = {
            "ms": reader._normalize_ms(raw) * valid[None],
            "ms_valid": valid,
            "target": target.astype(np.int64, copy=False),
            "supervised": supervised,
        }
        horizontal = rng.random() < self.horizontal_flip_probability
        vertical = rng.random() < self.vertical_flip_probability
        if horizontal or vertical:
            sample = synchronized_ms75_flip(
                sample, horizontal=horizontal, vertical=vertical
            )  # type: ignore[assignment]
        sample = _augment_local_spectral(sample, data=self.config["data"], rng=rng)
        sample = {key: np.ascontiguousarray(value) for key, value in sample.items()}
        return {
            "ms": torch.from_numpy(sample["ms"]).float(),
            "ms_valid": torch.from_numpy(sample["ms_valid"]).bool(),
            "target": torch.from_numpy(sample["target"]).long(),
            "supervised": torch.from_numpy(sample["supervised"]).bool(),
            "scene_id": str(block.scene_id),
            "left": int(window.col_off),
            "top": int(window.row_off),
            "sampling_kind": str(block.kind),
            "sampling_true_class": int(anchor.true_class) if anchor is not None else -1,
            "sampling_predicted_class": (
                int(anchor.predicted_class) if anchor is not None else -1
            ),
        }


class _DynamicFourSceneSampler(Sampler[int]):
    def __init__(self, dataset: DINOv3SATNIRDataset, *, start_index: int) -> None:
        self.dataset = dataset
        self.start_index = int(start_index)

    def __len__(self) -> int:
        return len(self.dataset) - self.start_index

    def __iter__(self) -> Iterator[int]:
        blocks = tuple(self.dataset._training_block_plan)
        order = four_scene_block_order(
            [str(block.scene_id) for block in blocks], block_size=16
        )
        audit = sampling_order_audit(
            blocks,
            order,
            epoch=self.dataset.epoch,
            physical_batch=int(self.dataset.config["train"]["batch_size"]),
        )
        audit["train_scene_count"] = len(self.dataset.scene_ids)
        audit["train_scene_ids_sha256"] = hashlib.sha256(
            ("\n".join(self.dataset.scene_ids) + "\n").encode("utf-8")
        ).hexdigest()
        audit["class_index_sha256"] = self.dataset.class_index.file_sha256
        audit["rare_class_ids"] = list(self.dataset.rare_class_ids)
        audit["hard_class_ids"] = list(self.dataset.hard_class_ids)
        audit["targeted_arm_class_block_counts"] = {
            kind: {
                str(class_id): sum(
                    block.kind == kind and block.key == class_id for block in blocks
                )
                for class_id in (
                    self.dataset.rare_class_ids
                    if kind == "rare"
                    else self.dataset.hard_class_ids
                )
            }
            for kind in ("rare", "hard")
        }
        audit["resume_start_index"] = self.start_index
        output = Path(str(self.dataset.config["experiment"]["output_dir"]))
        receipt = output / "sampling_audits" / f"epoch_{self.dataset.epoch:04d}.json"
        atomic_json_dump(audit, receipt)
        return iter(order[self.start_index :])


def _validated_collate(samples: list[dict[str, object]]) -> dict[str, object]:
    """Assert the actual worker-returned scene/kind identity per batch."""

    if not samples:
        raise ValueError("DINOv3SATNIR cannot collate an empty physical batch")
    if len({str(sample["scene_id"]) for sample in samples}) != 1:
        raise RuntimeError("DINOv3SATNIR physical batch crossed source scenes")
    if len({str(sample["sampling_kind"]) for sample in samples}) != 1:
        raise RuntimeError("DINOv3SATNIR physical batch mixed sampling arms")
    return default_collate(samples)


def build_dinov3_sat_nir_dataloader(
    config: dict,
    mode: str,
    *,
    start_batch_index: int = 0,
    io_batch_size: int | None = None,
    num_workers_override: int | None = None,
) -> DataLoader:
    if str(config["data"].get("dataset_format")) != "fbp_ms":
        raise ValueError("DINOv3SATNIR loader requires FBP MS data")
    if mode != "train":
        raise ValueError("DINOv3SATNIR development runner cannot open held24/official30")
    if io_batch_size is not None:
        raise ValueError("DINOv3SATNIR does not use I/O super-batches")
    dataset = DINOv3SATNIRDataset(config, mode)
    batch_size = int(config["train"]["batch_size"])
    start_batch_index = int(start_batch_index)
    if start_batch_index < 0:
        raise ValueError("invalid DINOv3SATNIR loader resume offset")
    start_index = start_batch_index * batch_size
    if start_index > len(dataset) or start_index % 64:
        raise ValueError("DINOv3SATNIR resume must lie on a 64-crop optimizer boundary")
    workers = int(
        config["data"]["num_workers"]
        if num_workers_override is None
        else num_workers_override
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=_DynamicFourSceneSampler(dataset, start_index=start_index),
        drop_last=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=_validated_collate,
    )


__all__ = [
    "DINOv3SATNIRDataset",
    "audit_dinov3_sat_nir_data_identity",
    "build_dinov3_sat_nir_dataloader",
    "exact_sampling_arm_schedule",
    "four_scene_block_order",
    "sampling_order_audit",
]
