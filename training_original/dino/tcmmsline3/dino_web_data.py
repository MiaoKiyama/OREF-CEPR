"""Isolated train96/full120 identities over the proven label-only crop planner.

The inherited implementation provides deterministic local BGRNIR extraction,
spectral/flipping augmentation, and the original 50/30/20 block schedule. Only
the explicit split/index construction and receipt role are new here.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler

from .data import FBPSamplingIndex, read_fbp_split, validate_fbp_splits
from .dinov3_sat_nir_data import (
    DINOv3SATNIRDataset,
    _validated_collate,
    exact_sampling_arm_schedule,
    four_scene_block_order,
    sampling_order_audit as _base_sampling_order_audit,
)
from .dinov3_sat_nir_full120_data import _validate_coordinate_labels
from .dino_web_protocol import (
    HARD_CLASS_IDS,
    HELD24_SPLIT,
    RARE_CLASS_IDS,
    TRAIN96_SPLIT,
    sha256_file,
    stage_parameters,
    validate_dino_web_config,
)
from .ms75_data import BalancedPointIndex, MS75MultiScaleDataset
from .runtime import atomic_json_dump

__all__ = [
    "DinoWebDataset",
    "audit_dino_web_data_identity",
    "build_dino_web_dataloader",
    "exact_sampling_arm_schedule",
    "four_scene_block_order",
    "sampling_order_audit",
]


def _scene_hash(scene_ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(scene_ids) + "\n").encode()).hexdigest()


def audit_dino_web_data_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate metadata plus every train-only anchor before any model load.

    Official30 split text is read for disjointness; its raster and label files
    are never opened. The same holds for held24 labels during train96 fitting.
    """
    validate_dino_web_config(config, require_pretrained_file=False)
    stage = stage_parameters(config)
    root = Path(str(config["data"]["root"])).expanduser().resolve()
    canonical = validate_fbp_splits(root)
    train_ids = read_fbp_split(root, stage["train_split"])
    full = stage["stage"] == "full120"
    held_ids = [] if full else read_fbp_split(root, HELD24_SPLIT)
    if len(train_ids) != stage["scene_count"] or len(set(train_ids)) != len(train_ids):
        raise ValueError("DINO WEB training split has wrong or duplicated scene count")
    if full:
        if train_ids != canonical["train"]:
            raise ValueError(
                "DINO WEB full120 must contain exact ordered official train120"
            )
    elif (
        len(held_ids) != 24
        or len(set(held_ids)) != 24
        or set(train_ids) & set(held_ids)
        or set(train_ids) | set(held_ids) != set(canonical["train"])
    ):
        raise ValueError(
            "DINO WEB fold0 must partition official train120 into disjoint 96/24"
        )
    if (set(train_ids) | set(held_ids)) & set(canonical["test"]):
        raise ValueError("DINO WEB fitting split overlaps official30")
    class_path = (root / stage["class_index"]).resolve()
    if sha256_file(class_path) != stage["class_index_sha256"]:
        raise ValueError("DINO WEB train-only sampling index SHA256 changed")
    index = FBPSamplingIndex(class_path)
    if (
        index.source_split != stage["index_source_split"]
        or index.scene_ids != list(train_ids)
        or index.seed != 1337
        or index.stride != 16
        or any(not index.scenes_by_class[class_id] for class_id in range(24))
    ):
        raise ValueError(
            "DINO WEB class index has wrong split/order/seed/stride/class provenance"
        )
    # Reuse the stronger existing coordinate validator, including negative
    # bounds, cap64, grayscale label encoding, and complete 24-class coverage.
    counts = _validate_coordinate_labels(
        root=root, train_ids=train_ids, class_index=index
    )
    split_names = ["all", "train", "train_dev", "val", "test"]
    if not full:
        split_names.extend([TRAIN96_SPLIT, HELD24_SPLIT])
    return {
        "schema_version": 1,
        "role": f"dino_web_{stage['stage']}_train_only_data_identity",
        "data_root": str(root),
        "train_scene_count": len(train_ids),
        "heldout_scene_count": len(held_ids),
        "official_test_scene_count": len(canonical["test"]),
        "train_scene_ids_sha256": _scene_hash(train_ids),
        "heldout_scene_ids_sha256": _scene_hash(held_ids),
        "split_file_sha256": {
            name: sha256_file(root / "splits" / f"{name}.txt") for name in split_names
        },
        "class_index": str(class_path),
        "class_index_sha256": stage["class_index_sha256"],
        "sampling_index_source_split": stage["index_source_split"],
        "sampling_index_scene_ids_exact_ordered_training_split": True,
        "sampling_coordinate_labels_verified": True,
        "class_anchor_point_counts": list(counts),
        "rare_class_ids": list(RARE_CLASS_IDS),
        "hard_class_ids": list(HARD_CLASS_IDS),
        "sampling_index_teacher_predictions_used": False,
        "official30_split_metadata_read": True,
        "official30_rasters_or_labels_read": False,
        "sar_or_pan_rasters_read": False,
        "held24_labels_or_rasters_read_for_train96_sampling": False,
        "full120_training_includes_former_held24": full,
        "validation_loader_enabled": False,
    }


class DinoWebDataset(DINOv3SATNIRDataset):
    """Use the original deterministic crop methods with a new stage identity."""

    def __init__(self, config: dict, mode: str) -> None:
        if mode != "train":
            raise ValueError(
                "DINO WEB fitting loader refuses held24/official30 and every non-train mode"
            )
        validate_dino_web_config(config, require_pretrained_file=False)
        stage = stage_parameters(config)
        planner = copy.deepcopy(config)
        data = planner["data"]
        dynamic = copy.deepcopy(data["dynamic_sampling"])
        data.update(
            {
                "dataset_variant": "ms75_multiscale",
                "protocol_stage": "final"
                if stage["stage"] == "full120"
                else "ms75_clean_fold0_train96",
                "short_context_size": 512,
                "short_input_size": 512,
                "global_input_size": 32,
                "horizontal_flip_probability": 0.5,
                "vertical_flip_probability": 0.5,
            }
        )
        for split_name in ("train_split", "val_split", "test_split"):
            data[split_name] = stage["train_split"]
        # Generic parent handles discovery/readers only. The authenticated
        # label-side rare/hard planner below replaces its disabled curriculum.
        data["dynamic_sampling"] = {
            "scene_fraction": 1.0,
            "class_fraction": 0.0,
            "confusion_fraction": 0.0,
            "center_jitter": 128,
            "max_scene_repeat": 16,
        }
        if data.get("debug_samples_per_epoch") is None:
            data.pop("debug_samples_per_epoch", None)
        MS75MultiScaleDataset.__init__(self, planner, mode)
        self.config = copy.deepcopy(config)
        self.class_index = BalancedPointIndex.from_class_npz(
            self._resolve_index(dynamic["class_index"]),
            expected_sha256=stage["class_index_sha256"],
            active_scene_ids=self.scene_ids,
            official_train_scene_ids=self.protocol.train_scene_ids,
            official_test_scene_ids=self.protocol.test_scene_ids,
            expected_source_split=stage["index_source_split"],
            require_exact_active_scene_ids=True,
        )
        if (
            len(self.scene_ids) != stage["scene_count"]
            or len(set(self.scene_ids)) != len(self.scene_ids)
            or set(self.scene_ids) & set(self.protocol.test_scene_ids)
            or self.confusion_index is not None
            or self.max_scene_repeat != 16
            or set(self.class_index.keys) != set(range(24))
        ):
            raise ValueError(
                "DINO WEB dataset identity or label-only sampling contract changed"
            )
        self.rare_class_ids, self.hard_class_ids = RARE_CLASS_IDS, HARD_CLASS_IDS
        self._dinov3_sat_sampling_fractions = {
            kind: float(dynamic[f"{kind}_fraction"])
            for kind in ("scene", "rare", "hard")
        }
        self.sampling_fractions = dict(self._dinov3_sat_sampling_fractions)
        self.set_epoch(0)


def sampling_order_audit(
    blocks: Sequence[object],
    sample_order: Sequence[int],
    *,
    epoch: int,
    physical_batch: int = 1,
    stage: str = "train96",
) -> dict[str, Any]:
    if stage not in ("train96", "full120"):
        raise ValueError("DINO WEB sampling audit stage is invalid")
    audit = _base_sampling_order_audit(
        blocks, sample_order, epoch=epoch, physical_batch=physical_batch
    )
    audit["role"] = f"dino_web_{stage}_label_side_sampling_plan"
    audit.pop("sampling_coordinates_train96_labels_only")
    audit[
        f"sampling_coordinates_{'train120' if stage == 'full120' else 'train96'}_labels_only"
    ] = True
    return audit


class _DynamicFourSceneSampler(Sampler[int]):
    def __init__(self, dataset: DinoWebDataset, *, start_index: int) -> None:
        self.dataset, self.start_index = dataset, int(start_index)

    def __len__(self) -> int:
        return len(self.dataset) - self.start_index

    def __iter__(self) -> Iterator[int]:
        dataset = self.dataset
        blocks = tuple(dataset._training_block_plan)
        order = four_scene_block_order([str(block.scene_id) for block in blocks])
        audit = sampling_order_audit(
            blocks,
            order,
            epoch=dataset.epoch,
            physical_batch=dataset.config["train"]["batch_size"],
            stage=stage_parameters(dataset.config)["stage"],
        )
        audit.update(
            {
                "variant": dataset.config["model"]["variant"],
                "mode": dataset.config["experiment"].get("mode", "formal"),
                "train_scene_count": len(dataset.scene_ids),
                "train_scene_ids_sha256": _scene_hash(dataset.scene_ids),
                "class_index_sha256": dataset.class_index.file_sha256,
                "rare_class_ids": list(RARE_CLASS_IDS),
                "hard_class_ids": list(HARD_CLASS_IDS),
                "resume_start_index": self.start_index,
            }
        )
        output = Path(str(dataset.config["experiment"]["output_dir"]))
        atomic_json_dump(
            audit, output / "sampling_audits" / f"epoch_{dataset.epoch:04d}.json"
        )
        return iter(order[self.start_index :])


def _worker_init(worker_id: int) -> None:
    del worker_id
    torch.set_num_threads(1)


def build_dino_web_dataloader(
    config: dict,
    mode: str,
    *,
    start_batch_index: int = 0,
    io_batch_size: int | None = None,
    num_workers_override: int | None = None,
) -> DataLoader:
    if mode != "train":
        raise ValueError("DINO WEB fitting loader refuses every non-train mode")
    if io_batch_size is not None:
        raise ValueError("DINO WEB does not use I/O super-batches")
    dataset = DinoWebDataset(config, mode)
    batch = config["train"]["batch_size"]
    if type(start_batch_index) is not int or start_batch_index < 0:
        raise ValueError("invalid DINO WEB resume offset")
    start_index = start_batch_index * batch
    if start_index > len(dataset) or start_index % 64:
        raise ValueError("DINO WEB resume must lie on a 64-crop optimizer boundary")
    workers = (
        config["data"]["num_workers"]
        if num_workers_override is None
        else num_workers_override
    )
    if type(workers) is not int or workers < 0:
        raise ValueError("DINO WEB worker count must be a nonnegative integer")
    return DataLoader(
        dataset,
        batch_size=batch,
        sampler=_DynamicFourSceneSampler(dataset, start_index=start_index),
        drop_last=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=_validated_collate,
        worker_init_fn=_worker_init,
    )
