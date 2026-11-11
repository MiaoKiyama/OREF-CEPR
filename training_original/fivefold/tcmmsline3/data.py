from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    ms_path: Path
    sar_path: Path
    label_path: Path
    ms_valid_path: Path
    sar_valid_path: Path
    metadata_path: Path
    time_gap_days: float
    qc_status: str


def load_manifest(
    root: str | Path,
    *,
    selected_scene_ids: Iterable[str] | None = None,
    expected_scene_count: int | None = 128,
    max_time_gap_days: float = 20.0,
) -> list[SceneRecord]:
    root = Path(root)
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    selected = None if selected_scene_ids is None else set(map(str, selected_scene_ids))

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if expected_scene_count is not None and len(rows) != expected_scene_count:
        raise ValueError(
            f"expected {expected_scene_count} retained scenes, found {len(rows)}"
        )

    records: list[SceneRecord] = []
    for row in rows:
        scene_id = str(row["scene_id"])
        if selected is not None and scene_id not in selected:
            continue
        qc_status = str(row.get("qc_status", "")).upper()
        if qc_status != "PASS":
            raise ValueError(f"scene {scene_id} is not QC PASS: {qc_status!r}")
        time_gap = abs(float(row["time_gap_days"]))
        if time_gap > max_time_gap_days:
            raise ValueError(
                f"scene {scene_id} has rejected time gap {time_gap:.3f} days"
            )

        def required_path(field: str) -> Path:
            relative = row.get(field, "")
            if not relative:
                raise ValueError(f"scene {scene_id} has empty required field {field}")
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"scene {scene_id} missing {field}: {path}")
            return path

        records.append(
            SceneRecord(
                scene_id=scene_id,
                ms_path=required_path("ms_path"),
                sar_path=required_path("sar_path"),
                label_path=required_path("label_path"),
                ms_valid_path=required_path("ms_valid_path"),
                sar_valid_path=required_path("sar_valid_path"),
                metadata_path=required_path("metadata_path"),
                time_gap_days=time_gap,
                qc_status=qc_status,
            )
        )

    if selected is not None:
        found = {record.scene_id for record in records}
        missing = selected - found
        if missing:
            raise ValueError(f"scene IDs missing from retained manifest: {sorted(missing)}")
    return records


def configured_sar_size(config: dict) -> int:
    """Return the input support size for the configured model architecture."""
    data = config["data"]
    architecture = str(
        config.get("model", {}).get("architecture", "ms_sar_fusion")
    )
    if architecture == "sar_native":
        return int(data.get("sar_native_size", data["sar_size"]))
    return int(data["sar_size"])


class SceneRasterReader:
    """Read an MS-grid patch and geographically matched SAR support."""

    def __init__(self, record: SceneRecord, config: dict):
        self.record = record
        self.config = config
        opened: list[rasterio.io.DatasetReader] = []
        try:
            self.ms = rasterio.open(record.ms_path)
            opened.append(self.ms)
            self.sar = rasterio.open(record.sar_path)
            opened.append(self.sar)
            self.label = rasterio.open(record.label_path)
            opened.append(self.label)
            self.ms_valid = rasterio.open(record.ms_valid_path)
            opened.append(self.ms_valid)
            self.sar_valid = rasterio.open(record.sar_valid_path)
            opened.append(self.sar_valid)
            self._validate_grids()
        except Exception:
            for dataset in opened:
                dataset.close()
            raise

    @staticmethod
    def _same_grid(
        left: rasterio.io.DatasetReader,
        right: rasterio.io.DatasetReader,
    ) -> bool:
        return (
            left.crs == right.crs
            and left.width == right.width
            and left.height == right.height
            and np.allclose(
                tuple(left.transform),
                tuple(right.transform),
                atol=1e-8,
                rtol=0.0,
            )
        )

    @staticmethod
    def _north_up(dataset: rasterio.io.DatasetReader) -> bool:
        transform = dataset.transform
        return (
            transform.a > 0
            and transform.e < 0
            and abs(transform.b) <= 1e-10
            and abs(transform.d) <= 1e-10
        )

    def _validate_grids(self) -> None:
        scene_id = self.record.scene_id
        if self.ms.count != 4:
            raise ValueError(f"scene {scene_id}: MS must have four bands")
        if self.sar.count != 2:
            raise ValueError(f"scene {scene_id}: SAR must have VV/VH bands")
        if self.ms.crs is None or self.sar.crs is None:
            raise ValueError(f"scene {scene_id}: MS/SAR CRS is missing")
        if self.ms.crs != self.sar.crs:
            raise ValueError(f"scene {scene_id}: MS/SAR CRS mismatch")
        if not self._north_up(self.ms) or not self._north_up(self.sar):
            raise ValueError(
                f"scene {scene_id}: rotated or south-up rasters are unsupported"
            )
        if not self._same_grid(self.ms, self.label):
            raise ValueError(
                f"scene {scene_id}: label is not on the exact MS grid"
            )
        if not self._same_grid(self.ms, self.ms_valid):
            raise ValueError(
                f"scene {scene_id}: MS validity mask is not on the exact MS grid"
            )
        if not self._same_grid(self.sar, self.sar_valid):
            raise ValueError(
                f"scene {scene_id}: SAR validity mask is not on the exact SAR grid"
            )

        descriptions = tuple(
            str(value or "").upper() for value in self.sar.descriptions
        )
        if any(descriptions) and not (
            "VV" in descriptions[0] and "VH" in descriptions[1]
        ):
            raise ValueError(
                f"scene {scene_id}: expected SAR band order VV/VH, "
                f"found {descriptions}"
            )

        tolerance = 1e-6
        ms_bounds = self.ms.bounds
        sar_bounds = self.sar.bounds
        if (
            sar_bounds.left > ms_bounds.left + tolerance
            or sar_bounds.bottom > ms_bounds.bottom + tolerance
            or sar_bounds.right < ms_bounds.right - tolerance
            or sar_bounds.top < ms_bounds.top - tolerance
        ):
            raise ValueError(
                f"scene {scene_id}: SAR does not fully cover the MS extent"
            )

    @property
    def width(self) -> int:
        return self.ms.width

    @property
    def height(self) -> int:
        return self.ms.height

    def close(self) -> None:
        for dataset in (self.ms, self.sar, self.label, self.ms_valid, self.sar_valid):
            dataset.close()

    def _normalize_ms(self, array: np.ndarray) -> np.ndarray:
        norm = self.config["normalization"]["ms"]
        divisor = float(norm["divisor"])
        mean = np.asarray(norm["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(norm["std"], dtype=np.float32)[:, None, None]
        return (array.astype(np.float32) / divisor - mean) / np.maximum(std, 1e-6)

    def _normalize_sar(
        self, array: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        norm = self.config["normalization"]["sar"]
        low, high = float(norm["db_min"]), float(norm["db_max"])
        array = np.nan_to_num(array.astype(np.float32), nan=low, posinf=high, neginf=low)
        array = np.clip(array, low, high)
        array = (array - low) / (high - low)
        return array * valid[None].astype(np.float32)

    def read_label_valid(self, window: Window) -> tuple[np.ndarray, np.ndarray]:
        label = self.label.read(
            1, window=window, boundless=True, fill_value=0
        ).astype(np.int64)
        ms_valid = (
            self.ms_valid.read(1, window=window, boundless=True, fill_value=0) > 0
        )
        supervised = ms_valid & (label > 0) & (label <= 24)
        target = np.where(supervised, label - 1, -1).astype(np.int64)
        return target, supervised

    def read_sar_patch(
        self, window: Window, sar_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read SAR support for a window expressed on this scene's MS grid."""
        left, bottom, right, top = window_bounds(window, self.ms.transform)
        sar_window = from_bounds(
            left, bottom, right, top, transform=self.sar.transform
        )
        sar_raw = self.sar.read(
            window=sar_window,
            out_shape=(2, sar_size, sar_size),
            boundless=True,
            fill_value=np.nan,
            resampling=Resampling.bilinear,
        )
        sar_valid = (
            self.sar_valid.read(
                1,
                window=sar_window,
                out_shape=(sar_size, sar_size),
                boundless=True,
                fill_value=0,
                resampling=Resampling.nearest,
            )
            > 0
        )
        intrinsic_valid = (
            self.sar.read_masks(
                window=sar_window,
                out_shape=(2, sar_size, sar_size),
                boundless=True,
                resampling=Resampling.nearest,
            )
            > 0
        ).all(axis=0)
        sar_valid &= intrinsic_valid & np.isfinite(sar_raw).all(axis=0)
        return self._normalize_sar(sar_raw, sar_valid), sar_valid

    def read_patch(self, window: Window, sar_size: int) -> dict[str, np.ndarray]:
        ms_raw = self.ms.read(window=window, boundless=True, fill_value=0)
        target, supervised = self.read_label_valid(window)
        ms_valid = (
            self.ms_valid.read(1, window=window, boundless=True, fill_value=0) > 0
        )

        if bool(self.config.get("load_sar", True)):
            sar, sar_valid = self.read_sar_patch(window, sar_size)
        else:
            sar = np.zeros((2, sar_size, sar_size), dtype=np.float32)
            sar_valid = np.zeros((sar_size, sar_size), dtype=np.bool_)

        sample = {
            "ms": self._normalize_ms(ms_raw),
            "sar": sar,
            "target": target,
            "supervised": supervised.astype(np.bool_),
            "ms_valid": ms_valid.astype(np.bool_),
            "sar_valid": sar_valid.astype(np.bool_),
        }
        if "context_size" not in self.config:
            sample["ms_context"] = sample["ms"].copy()
            sample["ms_context_valid"] = sample["ms_valid"].copy()
            return sample
        context_size = int(self.config["context_size"])
        context_input_size = int(
            self.config.get("context_input_size", int(window.width))
        )
        if context_size < int(window.width):
            raise ValueError("context_size must be at least the local patch size")
        offset_x = (context_size - int(window.width)) // 2
        offset_y = (context_size - int(window.height)) // 2
        context_window = Window(
            int(window.col_off) - offset_x,
            int(window.row_off) - offset_y,
            context_size,
            context_size,
        )
        context_raw = self.ms.read(
            window=context_window,
            out_shape=(self.ms.count, context_input_size, context_input_size),
            boundless=True,
            fill_value=0,
            resampling=Resampling.bilinear,
        )
        context_valid = (
            self.ms_valid.read(
                1,
                window=context_window,
                out_shape=(context_input_size, context_input_size),
                boundless=True,
                fill_value=0,
                resampling=Resampling.nearest,
            )
            > 0
        )
        normalized_context = self._normalize_ms(context_raw)
        sample["ms_context"] = normalized_context * context_valid[None]
        sample["ms_context_valid"] = context_valid.astype(np.bool_)
        return sample


def iter_scene_windows(
    width: int, height: int, patch_size: int
) -> Iterator[Window]:
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            yield Window(
                left,
                top,
                min(patch_size, width - left),
                min(patch_size, height - top),
            )


class MSSARDataset(Dataset):
    def __init__(self, config: dict, mode: str):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"unsupported mode: {mode}")
        self.config = config
        self.mode = mode
        data = config["data"]
        self.patch_size = int(data["patch_size"])
        self.sar_size = configured_sar_size(config)
        self.seed = int(config["experiment"]["seed"])
        self.epoch = 0
        self.sampling_config = data.get("sampling", {})
        self.sampling_strategy = str(
            self.sampling_config.get("strategy", "uniform")
        )
        self.sar_pairing = str(data.get("sar_pairing", "aligned"))
        scene_ids = data[f"{mode}_scene_ids"]
        self.records = load_manifest(
            data["root"],
            selected_scene_ids=scene_ids,
            expected_scene_count=int(data["expected_scene_count"]),
            max_time_gap_days=float(data["max_time_gap_days"]),
        )
        self.record_index = {record.scene_id: index for index, record in enumerate(self.records)}
        self.samples_per_epoch = int(
            data["samples_per_epoch"]
            if mode == "train"
            else data["eval_patches_per_scene"] * len(self.records)
        )
        self._readers: dict[str, SceneRasterReader] = {}
        self._rare_candidates: dict[int, list[tuple[int, int, int]]] = {}
        self._hard_candidates: dict[int, list[tuple[int, int, int]]] = {}
        if mode == "train" and self.sampling_strategy == "directed":
            self._rare_candidates = self._build_class_candidates(
                [int(value) for value in self.sampling_config["rare_class_ids"]]
            )
            self._hard_candidates = self._build_class_candidates(
                [int(value) for value in self.sampling_config["hard_class_ids"]]
            )
            print(
                json.dumps(
                    {
                        "directed_sampling_index": {
                            "rare": {
                                str(key): len(value)
                                for key, value in self._rare_candidates.items()
                            },
                            "hard": {
                                str(key): len(value)
                                for key, value in self._hard_candidates.items()
                            },
                        }
                    }
                ),
                flush=True,
            )
        self._eval_windows = self._build_eval_windows() if mode != "train" else []
        if mode != "train":
            for reader in self._readers.values():
                reader.close()
            self._readers = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        for reader in getattr(self, "_readers", {}).values():
            reader.close()
        if hasattr(self, "_readers"):
            self._readers = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _reader(self, record: SceneRecord) -> SceneRasterReader:
        reader = self._readers.get(record.scene_id)
        if reader is None:
            reader = SceneRasterReader(record, self.config["data"])
            self._readers[record.scene_id] = reader
        return reader

    def _candidate_window(
        self, reader: SceneRasterReader, rng: random.Random
    ) -> Window:
        max_left = max(0, reader.width - self.patch_size)
        max_top = max(0, reader.height - self.patch_size)
        left = rng.randint(0, max_left) if max_left else 0
        top = rng.randint(0, max_top) if max_top else 0
        return Window(left, top, self.patch_size, self.patch_size)

    def _build_class_candidates(
        self, class_ids: list[int]
    ) -> dict[int, list[tuple[int, int, int]]]:
        """Index at most one center per label block and class.

        The compact index avoids loading label rasters in DataLoader workers
        and gives extremely rare classes a deterministic way to be sampled.
        """
        candidates = {class_id: [] for class_id in class_ids}
        block_size = int(self.sampling_config.get("index_block_size", 256))
        for record_index, record in enumerate(self.records):
            # Read each scene sequentially once. Repeated 256 px raster reads
            # made the one-time index scan IOPS-bound on the server archive.
            # Only one scene is resident, so peak host memory remains bounded.
            with rasterio.open(record.label_path) as label_dataset:
                labels = label_dataset.read(1)
            with rasterio.open(record.ms_valid_path) as valid_dataset:
                valid = valid_dataset.read(1) > 0
            height, width = labels.shape
            for top in range(0, height, block_size):
                for left in range(0, width, block_size):
                    label_block = labels[
                        top : top + block_size, left : left + block_size
                    ]
                    valid_block = valid[
                        top : top + block_size, left : left + block_size
                    ]
                    present = set(
                        int(value) - 1
                        for value in np.unique(label_block[valid_block])
                        if 1 <= int(value) <= 24
                    )
                    for class_id in present.intersection(candidates):
                        locations = np.argwhere(
                            (label_block == class_id + 1) & valid_block
                        )
                        row, column = locations[len(locations) // 2]
                        candidates[class_id].append(
                            (
                                record_index,
                                left + int(column),
                                top + int(row),
                            )
                        )
        return {
            class_id: values
            for class_id, values in candidates.items()
            if values
        }

    def _window_around_candidate(
        self,
        pool: dict[int, list[tuple[int, int, int]]],
        rng: random.Random,
    ) -> tuple[int, Window] | None:
        if not pool:
            return None
        class_id = rng.choice(sorted(pool))
        record_index, center_x, center_y = rng.choice(pool[class_id])
        record = self.records[record_index]
        reader = self._reader(record)
        jitter = int(self.sampling_config.get("center_jitter", self.patch_size // 4))
        offset_x = rng.randint(-jitter, jitter) if jitter else 0
        offset_y = rng.randint(-jitter, jitter) if jitter else 0
        max_left = max(0, reader.width - self.patch_size)
        max_top = max(0, reader.height - self.patch_size)
        left = min(
            max(center_x - self.patch_size // 2 + offset_x, 0),
            max_left,
        )
        top = min(
            max(center_y - self.patch_size // 2 + offset_y, 0),
            max_top,
        )
        return (
            record_index,
            Window(left, top, self.patch_size, self.patch_size),
        )

    def _acceptable_window(
        self, reader: SceneRasterReader, rng: random.Random
    ) -> Window:
        threshold = float(self.config["data"]["min_labeled_fraction"])
        attempts = int(self.config["data"]["sampling_attempts"])
        best_window: Window | None = None
        best_fraction = -1.0
        for _ in range(attempts):
            window = self._candidate_window(reader, rng)
            _, supervised = reader.read_label_valid(window)
            fraction = float(supervised.mean())
            if fraction > best_fraction:
                best_window, best_fraction = window, fraction
            if fraction >= threshold:
                return window
        assert best_window is not None
        return best_window

    def _same_scene_spatial_window(
        self,
        reader: SceneRasterReader,
        source: Window,
    ) -> Window:
        """Choose a deterministic distant support while keeping scene identity."""
        max_left = max(0, reader.width - self.patch_size)
        max_top = max(0, reader.height - self.patch_size)
        if max_left == 0 and max_top == 0:
            return source
        left_step = max(self.patch_size, max_left // 2)
        top_step = max(self.patch_size, max_top // 2)
        left = (
            (int(source.col_off) + left_step) % (max_left + 1)
            if max_left
            else 0
        )
        top = (
            (int(source.row_off) + top_step) % (max_top + 1)
            if max_top
            else 0
        )
        return Window(left, top, self.patch_size, self.patch_size)

    def _apply_sar_pairing(
        self,
        sample: dict[str, np.ndarray],
        *,
        record_index: int,
        source_window: Window,
    ) -> tuple[str, Window]:
        source_record = self.records[record_index]
        if self.sar_pairing == "aligned":
            return source_record.scene_id, source_window
        if self.sar_pairing == "zero":
            sample["sar"].fill(0)
            sample["sar_valid"].fill(False)
            return "none", source_window
        if self.sar_pairing == "same_scene_spatial":
            reader = self._reader(source_record)
            donor_window = self._same_scene_spatial_window(reader, source_window)
            sar, valid = reader.read_sar_patch(donor_window, self.sar_size)
            sample["sar"], sample["sar_valid"] = sar, valid
            return source_record.scene_id, donor_window
        if self.sar_pairing == "cross_scene":
            if len(self.records) < 2:
                raise ValueError(
                    f"{self.mode} cross_scene SAR pairing requires at least two scenes"
                )
            offset = 1 + self.seed % (len(self.records) - 1)
            donor_index = (record_index + offset) % len(self.records)
            donor_record = self.records[donor_index]
            donor_reader = self._reader(donor_record)
            # Keep the source-window sampling and augmentation RNG identical
            # across aligned/same-scene/cross-scene experiments.
            donor_rng = random.Random(
                self.seed
                + self.epoch * 1_000_003
                + record_index * 10_007
                + int(source_window.col_off) * 101
                + int(source_window.row_off) * 103
            )
            donor_window = self._candidate_window(donor_reader, donor_rng)
            sar, valid = donor_reader.read_sar_patch(
                donor_window, self.sar_size
            )
            sample["sar"], sample["sar_valid"] = sar, valid
            return donor_record.scene_id, donor_window
        raise ValueError(f"unsupported SAR pairing: {self.sar_pairing}")

    def _build_eval_windows(self) -> list[tuple[int, Window]]:
        windows: list[tuple[int, Window]] = []
        count = int(self.config["data"]["eval_patches_per_scene"])
        for record_index, record in enumerate(self.records):
            reader = self._reader(record)
            rng = random.Random(self.seed + 100_003 * (record_index + 1))
            for _ in range(count):
                windows.append(
                    (record_index, self._acceptable_window(reader, rng))
                )
        return windows

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _augment(
        self, sample: dict[str, np.ndarray], rng: random.Random
    ) -> dict[str, np.ndarray]:
        if rng.random() < 0.5:
            for key in ("ms", "ms_context", "sar"):
                sample[key] = np.flip(sample[key], axis=-1)
            for key in (
                "target",
                "supervised",
                "ms_valid",
                "ms_context_valid",
                "sar_valid",
            ):
                sample[key] = np.flip(sample[key], axis=-1)
            for key in (
                "ms75_truth_probability",
                "local512_truth_probability",
            ):
                if key in sample:
                    sample[key] = np.flip(sample[key], axis=-1)
        if rng.random() < 0.5:
            for key in ("ms", "ms_context", "sar"):
                sample[key] = np.flip(sample[key], axis=-2)
            for key in (
                "target",
                "supervised",
                "ms_valid",
                "ms_context_valid",
                "sar_valid",
            ):
                sample[key] = np.flip(sample[key], axis=-2)
            for key in (
                "ms75_truth_probability",
                "local512_truth_probability",
            ):
                if key in sample:
                    sample[key] = np.flip(sample[key], axis=-2)
        rotations = rng.randrange(4)
        if rotations:
            for key in ("ms", "ms_context", "sar"):
                sample[key] = np.rot90(sample[key], rotations, axes=(-2, -1))
            for key in (
                "target",
                "supervised",
                "ms_valid",
                "ms_context_valid",
                "sar_valid",
            ):
                sample[key] = np.rot90(sample[key], rotations, axes=(-2, -1))
            for key in (
                "ms75_truth_probability",
                "local512_truth_probability",
            ):
                if key in sample:
                    sample[key] = np.rot90(
                        sample[key], rotations, axes=(-2, -1)
                    )
        spectral = self.config["data"].get("spectral_augmentation", {})
        if bool(spectral.get("enabled", False)) and rng.random() < float(
            spectral.get("probability", 0.0)
        ):
            if "fusion_truth_probability_valid" in sample:
                sample["fusion_truth_probability_valid"] = np.asarray(
                    False, dtype=np.bool_
                )
            normalization = self.config["data"]["normalization"]["ms"]
            mean = np.asarray(
                normalization["mean"], dtype=np.float32
            )[:, None, None]
            std = np.asarray(
                normalization["std"], dtype=np.float32
            )[:, None, None]
            global_gain_range = spectral.get("global_gain_range", [1.0, 1.0])
            global_offset_range = spectral.get(
                "global_offset_range", [0.0, 0.0]
            )
            band_gain_range = spectral.get("band_gain_range", [1.0, 1.0])
            band_offset_range = spectral.get("band_offset_range", [0.0, 0.0])
            global_gain = rng.uniform(
                float(global_gain_range[0]), float(global_gain_range[1])
            )
            global_offset = rng.uniform(
                float(global_offset_range[0]), float(global_offset_range[1])
            )
            band_gain = np.asarray(
                [
                    rng.uniform(
                        float(band_gain_range[0]),
                        float(band_gain_range[1]),
                    )
                    for _ in range(4)
                ],
                dtype=np.float32,
            )[:, None, None]
            band_offset = np.asarray(
                [
                    rng.uniform(
                        float(band_offset_range[0]), float(band_offset_range[1])
                    )
                    for _ in range(4)
                ],
                dtype=np.float32,
            )[:, None, None]
            for key, valid_key in (
                ("ms", "ms_valid"),
                ("ms_context", "ms_context_valid"),
            ):
                original = sample[key]
                reflectance = original * std + mean
                augmented = (
                    reflectance * global_gain * band_gain
                    + global_offset
                    + band_offset
                )
                augmented = np.clip(augmented, 0.0, 1.5)
                normalized = (augmented - mean) / np.maximum(std, 1e-6)
                sample[key] = np.where(
                    sample[valid_key][None], normalized, original
                ).astype(np.float32, copy=False)
        return {
            key: (
                np.asarray(value, dtype=np.bool_)
                if key == "fusion_truth_probability_valid"
                else np.ascontiguousarray(value)
            )
            for key, value in sample.items()
        }

    @staticmethod
    def _shift_sar(
        sar: np.ndarray,
        valid: np.ndarray,
        *,
        shift_y: int,
        shift_x: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        shifted_sar = np.zeros_like(sar)
        shifted_valid = np.zeros_like(valid)
        height, width = valid.shape
        source_y0 = max(0, -shift_y)
        source_y1 = min(height, height - shift_y)
        source_x0 = max(0, -shift_x)
        source_x1 = min(width, width - shift_x)
        target_y0 = max(0, shift_y)
        target_y1 = min(height, height + shift_y)
        target_x0 = max(0, shift_x)
        target_x1 = min(width, width + shift_x)
        if source_y1 > source_y0 and source_x1 > source_x0:
            shifted_sar[
                :, target_y0:target_y1, target_x0:target_x1
            ] = sar[:, source_y0:source_y1, source_x0:source_x1]
            shifted_valid[
                target_y0:target_y1, target_x0:target_x1
            ] = valid[source_y0:source_y1, source_x0:source_x1]
        return shifted_sar, shifted_valid

    def _degrade_sar(
        self,
        sample: dict[str, np.ndarray],
        rng: random.Random,
    ) -> tuple[dict[str, np.ndarray], bool, bool]:
        train = self.config["train"]
        shifted = False
        noisy = False
        if rng.random() < float(train.get("sar_shift_probability", 0.0)):
            maximum = int(train.get("sar_shift_max_pixels", 0))
            if maximum > 0:
                shift_y = rng.randint(-maximum, maximum)
                shift_x = rng.randint(-maximum, maximum)
                if shift_y == 0 and shift_x == 0:
                    shift_x = 1
                sample["sar"], sample["sar_valid"] = self._shift_sar(
                    sample["sar"],
                    sample["sar_valid"],
                    shift_y=shift_y,
                    shift_x=shift_x,
                )
                shifted = True
        if rng.random() < float(train.get("sar_noise_probability", 0.0)):
            standard_deviation = float(train.get("sar_noise_std", 0.0))
            if standard_deviation > 0:
                generator = np.random.default_rng(rng.randrange(2**32))
                noise = generator.normal(
                    0.0,
                    standard_deviation,
                    size=sample["sar"].shape,
                ).astype(sample["sar"].dtype)
                sample["sar"] = np.where(
                    sample["sar_valid"][None],
                    sample["sar"] + noise,
                    0.0,
                )
                noisy = True
        contiguous = {
            key: np.ascontiguousarray(value)
            for key, value in sample.items()
        }
        return contiguous, shifted, noisy

    def __getitem__(
        self, index: int
    ) -> dict[str, torch.Tensor | str | int | float]:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + int(index))
        if self.mode == "train":
            directed: tuple[int, Window] | None = None
            if self.sampling_strategy == "directed":
                draw = rng.random()
                rare_fraction = float(
                    self.sampling_config.get("rare_fraction", 0.0)
                )
                hard_fraction = float(
                    self.sampling_config.get("hard_fraction", 0.0)
                )
                if draw < rare_fraction:
                    directed = self._window_around_candidate(
                        self._rare_candidates, rng
                    )
                elif draw < rare_fraction + hard_fraction:
                    directed = self._window_around_candidate(
                        self._hard_candidates, rng
                    )
            if directed is None:
                record_index = rng.randrange(len(self.records))
                record = self.records[record_index]
                reader = self._reader(record)
                window = self._acceptable_window(reader, rng)
            else:
                record_index, window = directed
                record = self.records[record_index]
                reader = self._reader(record)
        else:
            record_index, window = self._eval_windows[index]
            record = self.records[record_index]
            reader = self._reader(record)

        sample = reader.read_patch(window, self.sar_size)
        sar_source_scene_id, sar_source_window = self._apply_sar_pairing(
            sample,
            record_index=record_index,
            source_window=window,
        )
        sar_dropped = False
        sar_shifted = False
        sar_noisy = False
        if self.mode == "train":
            sample = self._augment(sample, rng)
            if rng.random() < float(self.config["train"]["sar_dropout_probability"]):
                sample["sar"].fill(0)
                sample["sar_valid"].fill(False)
                sar_dropped = True
            else:
                sample, sar_shifted, sar_noisy = self._degrade_sar(
                    sample, rng
                )

        # Keep acquisition-gap metadata fixed to the source MS scene in
        # counterfactual pairing controls. Otherwise a cross-scene SAR swap
        # changes both the image and the router covariate.
        sar_time_gap_days = record.time_gap_days
        sar_aux_eligible = (
            self.sar_pairing == "aligned"
            and not sar_dropped
            and not sar_shifted
        )

        return {
            "ms": torch.from_numpy(sample["ms"]).float(),
            "ms_context": torch.from_numpy(sample["ms_context"]).float(),
            "ms_context_valid": torch.from_numpy(
                sample["ms_context_valid"]
            ).bool(),
            "sar": torch.from_numpy(sample["sar"]).float(),
            "target": torch.from_numpy(sample["target"]).long(),
            "supervised": torch.from_numpy(sample["supervised"]).bool(),
            "ms_valid": torch.from_numpy(sample["ms_valid"]).bool(),
            "sar_valid": torch.from_numpy(sample["sar_valid"]).bool(),
            "scene_id": record.scene_id,
            "left": int(window.col_off),
            "top": int(window.row_off),
            "sar_pairing": self.sar_pairing,
            "sar_source_scene_id": sar_source_scene_id,
            "sar_source_left": int(sar_source_window.col_off),
            "sar_source_top": int(sar_source_window.row_off),
            "time_gap_days": float(sar_time_gap_days),
            "sar_dropped": sar_dropped,
            "sar_shifted": sar_shifted,
            "sar_noisy": sar_noisy,
            "sar_aux_eligible": sar_aux_eligible,
        }


@dataclass(frozen=True)
class FBPSceneRecord:
    scene_id: str
    time_gap_days: float = 0.0


def read_fbp_split(root: str | Path, split: str) -> list[str]:
    path = Path(root) / "splits" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing FBP split: {path}")
    scene_ids = [
        value.strip()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if not scene_ids:
        raise ValueError(f"empty FBP split: {path}")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError(f"duplicate scene IDs in FBP split: {path}")
    for scene_id in scene_ids:
        if (
            Path(scene_id).name != scene_id
            or "/" in scene_id
            or "\\" in scene_id
        ):
            raise ValueError(f"unsafe FBP scene ID: {scene_id!r}")
    return scene_ids


def validate_fbp_splits(root: str | Path) -> dict[str, list[str]]:
    root = Path(root)
    splits = {
        name: read_fbp_split(root, name)
        for name in ("all", "train", "train_dev", "val", "test")
    }
    all_ids = set(splits["all"])
    official_train = set(splits["train"])
    train_dev = set(splits["train_dev"])
    validation = set(splits["val"])
    test = set(splits["test"])
    if len(all_ids) != 150 or len(official_train) != 120 or len(test) != 30:
        raise ValueError(
            "FBP requires exactly 150 all, 120 official-train and 30 test scenes"
        )
    if len(train_dev) != 108 or len(validation) != 12:
        raise ValueError(
            "FBP development split must contain 108 train and 12 val scenes"
        )
    if official_train & test or official_train | test != all_ids:
        raise ValueError("FBP official train/test splits are not a disjoint partition")
    if train_dev & validation or train_dev | validation != official_train:
        raise ValueError("FBP train_dev/val splits are not a partition of official train")
    return splits


def _fbp_resize_multiband(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"expected HWC four-band input, got {image.shape}")
    bands = [
        np.asarray(
            Image.fromarray(image[..., band], mode="L").resize(
                (size, size), resample=Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
        for band in range(4)
    ]
    return np.stack(bands, axis=-1)


def _extract_boundless(
    array: np.ndarray,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    output_shape = (height, width, *array.shape[2:])
    output = np.zeros(output_shape, dtype=array.dtype)
    valid = np.zeros((height, width), dtype=np.bool_)
    source_left = max(left, 0)
    source_top = max(top, 0)
    source_right = min(left + width, array.shape[1])
    source_bottom = min(top + height, array.shape[0])
    if source_right <= source_left or source_bottom <= source_top:
        return output, valid
    target_left = source_left - left
    target_top = source_top - top
    target_right = target_left + source_right - source_left
    target_bottom = target_top + source_bottom - source_top
    output[target_top:target_bottom, target_left:target_right] = array[
        source_top:source_bottom, source_left:source_right
    ]
    valid[target_top:target_bottom, target_left:target_right] = True
    return output, valid


class FBPSceneReader:
    """Safely decode one official FBP scene without CMYK color conversion."""

    def __init__(self, root: str | Path, scene_id: str, config: dict):
        self.root = Path(root)
        self.record = FBPSceneRecord(scene_id)
        self.config = config
        image_path = self.root / "images" / f"{scene_id}.tif"
        label_path = self.root / "labels" / f"{scene_id}.png"
        with Image.open(image_path) as source:
            if source.mode != "CMYK":
                raise ValueError(
                    f"FBP image must expose four untouched samples: {image_path}"
                )
            self.image = np.asarray(source, dtype=np.uint8).copy()
        with Image.open(label_path) as source:
            if source.mode != "L":
                raise ValueError(f"FBP label must be indexed grayscale: {label_path}")
            self.label = np.asarray(source, dtype=np.uint8).copy()
        expected = tuple(
            int(value)
            for value in config.get("expected_hw", [6908, 7300])
        )
        if self.image.shape != (*expected, 4):
            raise ValueError(
                f"scene {scene_id}: expected FBP image {(*expected, 4)}, "
                f"found {self.image.shape}"
            )
        if self.label.shape != expected:
            raise ValueError(
                f"scene {scene_id}: image/label shape mismatch {self.label.shape}"
            )
        if self.label.size and int(self.label.max()) > 24:
            raise ValueError(f"scene {scene_id}: label value exceeds 24")

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def close(self) -> None:
        self.image = np.empty((0, 0, 4), dtype=np.uint8)
        self.label = np.empty((0, 0), dtype=np.uint8)

    def _normalize_ms(self, nir_rgb: np.ndarray) -> np.ndarray:
        normalization = self.config["normalization"]["ms"]
        # Stored FBP NIR,R,G,B -> the repository-wide B,G,R,NIR contract.
        bgrn = nir_rgb[..., [3, 2, 1, 0]].transpose(2, 0, 1).astype(np.float32)
        mean = np.asarray(normalization["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(normalization["std"], dtype=np.float32)[:, None, None]
        return (
            bgrn / float(normalization["divisor"]) - mean
        ) / np.maximum(std, 1e-6)

    def read_label_valid(self, window: Window) -> tuple[np.ndarray, np.ndarray]:
        label, image_valid = _extract_boundless(
            self.label,
            left=int(window.col_off),
            top=int(window.row_off),
            width=int(window.width),
            height=int(window.height),
        )
        supervised = image_valid & (label >= 1) & (label <= 24)
        target = np.where(supervised, label.astype(np.int64) - 1, -1)
        return target, supervised

    def read_patch(self, window: Window, sar_size: int) -> dict[str, np.ndarray]:
        left = int(window.col_off)
        top = int(window.row_off)
        width = int(window.width)
        height = int(window.height)
        image, ms_valid = _extract_boundless(
            self.image,
            left=left,
            top=top,
            width=width,
            height=height,
        )
        target, supervised = self.read_label_valid(window)
        context_size = int(self.config.get("context_size", width))
        context_input_size = int(self.config.get("context_input_size", width))
        context_left = left - (context_size - width) // 2
        context_top = top - (context_size - height) // 2
        context, context_valid = _extract_boundless(
            self.image,
            left=context_left,
            top=context_top,
            width=context_size,
            height=context_size,
        )
        context = _fbp_resize_multiband(context, context_input_size)
        context_valid = np.asarray(
            Image.fromarray(context_valid.astype(np.uint8) * 255, mode="L").resize(
                (context_input_size, context_input_size),
                resample=Image.Resampling.NEAREST,
            )
        ) > 0
        normalized_context = self._normalize_ms(context)
        return {
            "ms": self._normalize_ms(image),
            "ms_context": normalized_context * context_valid[None],
            "ms_context_valid": context_valid,
            "sar": np.zeros((2, sar_size, sar_size), dtype=np.float32),
            "target": target.astype(np.int64, copy=False),
            "supervised": supervised,
            "ms_valid": ms_valid,
            "sar_valid": np.zeros((sar_size, sar_size), dtype=np.bool_),
        }


class FBPSamplingIndex:
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"missing FBP sampling index: {path}")
        archive = np.load(path, allow_pickle=False)
        expected_keys = {
            "scene_ids",
            "source_split",
            "stride",
            "seed",
            *(f"class_{class_id}" for class_id in range(24)),
        }
        if set(archive.files) != expected_keys:
            raise ValueError("FBP sampling index has unexpected or missing fields")
        scene_ids = [str(value) for value in archive["scene_ids"].tolist()]
        if not scene_ids or len(scene_ids) != len(set(scene_ids)):
            raise ValueError("FBP sampling index has invalid scene IDs")
        self.scene_ids = scene_ids
        self.source_split = str(np.asarray(archive["source_split"]).item())
        self.stride = int(np.asarray(archive["stride"]).item())
        self.seed = int(np.asarray(archive["seed"]).item())
        if self.stride <= 0 or self.seed < 0:
            raise ValueError("FBP sampling index stride/seed is invalid")
        self.by_scene: dict[str, dict[int, np.ndarray]] = {
            scene_id: {} for scene_id in scene_ids
        }
        for class_id in range(24):
            values = np.asarray(archive[f"class_{class_id}"], dtype=np.int32)
            if values.ndim != 2 or values.shape[1] != 3:
                raise ValueError(f"malformed FBP class_{class_id} sampling index")
            if values.size and (
                np.any(values[:, 0] < 0)
                or np.any(values[:, 0] >= len(scene_ids))
                or np.any(values[:, 1:] < 0)
            ):
                raise ValueError(
                    f"out-of-range FBP class_{class_id} sampling coordinates"
                )
            if len(values) != len(np.unique(values, axis=0)):
                raise ValueError(
                    f"duplicate FBP class_{class_id} sampling coordinates"
                )
            for scene_index in np.unique(values[:, 0]):
                scene_id = scene_ids[int(scene_index)]
                self.by_scene[scene_id][class_id] = values[
                    values[:, 0] == scene_index, 1:3
                ]
        self.scenes_by_class = {
            class_id: tuple(
                scene_id
                for scene_id in self.scene_ids
                if class_id in self.by_scene[scene_id]
            )
            for class_id in range(24)
        }

    def sample(
        self,
        scene_id: str,
        class_ids: list[int],
        rng: random.Random,
    ) -> tuple[int, int] | None:
        available = [
            class_id
            for class_id in class_ids
            if class_id in self.by_scene.get(scene_id, {})
        ]
        if not available:
            return None
        class_id = rng.choice(available)
        coordinates = self.by_scene[scene_id][class_id]
        row, column = coordinates[rng.randrange(len(coordinates))]
        return int(row), int(column)

    def sample_class_global(
        self,
        class_id: int,
        rng: random.Random,
    ) -> tuple[str, int, int]:
        """Choose class first, then a supporting scene and point uniformly."""

        class_id = int(class_id)
        if class_id not in range(24):
            raise ValueError("neutral class ID must be in [0,23]")
        scenes = self.scenes_by_class[class_id]
        if not scenes:
            raise ValueError(f"fold-train index has no points for class {class_id}")
        scene_id = scenes[rng.randrange(len(scenes))]
        coordinates = self.by_scene[scene_id][class_id]
        row, column = coordinates[rng.randrange(len(coordinates))]
        return scene_id, int(row), int(column)


def validate_fbp_sampling_index_semantics(
    index: FBPSamplingIndex,
    *,
    data_root: str | Path,
    expected_scene_ids: list[str],
    expected_seed: int,
    expected_stride: int = 16,
    expected_max_points_per_class_scene: int = 64,
    expected_source_split: str = "official_train_oof_fold",
) -> None:
    """Authenticate train-only point coordinates against train-only labels."""

    if (
        index.source_split != str(expected_source_split)
        or index.scene_ids != list(expected_scene_ids)
        or index.seed != int(expected_seed)
        or index.stride != int(expected_stride)
    ):
        raise ValueError("FBP sampling index split/scene/seed/stride mismatch")
    if any(not index.scenes_by_class[class_id] for class_id in range(24)):
        raise ValueError("FBP sampling index must cover all 24 classes")
    root = Path(data_root)
    for scene_id in index.scene_ids:
        label_path = root / "labels" / f"{scene_id}.png"
        with Image.open(label_path) as source:
            if source.mode != "L":
                raise ValueError(f"FBP label must be indexed grayscale: {label_path}")
            label = np.asarray(source, dtype=np.uint8)
        height, width = label.shape
        for class_id, coordinates in index.by_scene[scene_id].items():
            if not 0 < len(coordinates) <= int(
                expected_max_points_per_class_scene
            ):
                raise ValueError("FBP sampling index exceeds its per-scene cap")
            rows = coordinates[:, 0]
            columns = coordinates[:, 1]
            if (
                np.any(rows >= height)
                or np.any(columns >= width)
                or np.any(label[rows, columns] != class_id + 1)
            ):
                raise ValueError(
                    "FBP sampling index coordinate/class semantics mismatch"
                )


class FBPFixedTileIndex:
    """The shared FBP 40k tile pool with a fixed 32k/8k split."""

    def __init__(self, path: str | Path, subset: str):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"missing FBP fixed tile index: {path}")
        archive = np.load(path, allow_pickle=False)
        try:
            required = {
                "scene_ids",
                "crop_size",
                "train_tiles",
                "train_class_presence",
                "val_tiles",
                "val_class_presence",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    f"FBP fixed tile index is missing keys: {sorted(missing)}"
                )
            self.path = path
            self.scene_ids = [
                str(value) for value in archive["scene_ids"].tolist()
            ]
            if not self.scene_ids or len(self.scene_ids) != len(
                set(self.scene_ids)
            ):
                raise ValueError("FBP fixed tile index has invalid scene IDs")
            self.crop_size = int(np.asarray(archive["crop_size"]).item())
            if self.crop_size <= 0:
                raise ValueError("FBP fixed tile crop size must be positive")
            self.tiles_by_subset: dict[str, np.ndarray] = {}
            self.presence_by_subset: dict[str, np.ndarray] = {}
            for name in ("train", "val"):
                tiles = np.asarray(
                    archive[f"{name}_tiles"], dtype=np.int32
                )
                presence = np.asarray(
                    archive[f"{name}_class_presence"], dtype=np.uint8
                )
                if tiles.ndim != 2 or tiles.shape[1] != 3:
                    raise ValueError(
                        f"malformed FBP fixed {name} tiles: {tiles.shape}"
                    )
                if presence.shape != (len(tiles), 24):
                    raise ValueError(
                        f"malformed FBP fixed {name} presence: {presence.shape}"
                    )
                if np.any(presence > 1):
                    raise ValueError(
                        f"FBP fixed {name} presence must be binary"
                    )
                if len(tiles) and (
                    int(tiles[:, 0].min()) < 0
                    or int(tiles[:, 0].max()) >= len(self.scene_ids)
                    or int(tiles[:, 1:].min()) < 0
                ):
                    raise ValueError(
                        f"FBP fixed {name} tiles contain invalid coordinates"
                    )
                order = np.lexsort((tiles[:, 2], tiles[:, 1], tiles[:, 0]))
                self.tiles_by_subset[name] = tiles[order]
                self.presence_by_subset[name] = presence[order]
        finally:
            archive.close()

        training_keys = {
            tuple(int(value) for value in row)
            for row in self.tiles_by_subset["train"]
        }
        validation_keys = {
            tuple(int(value) for value in row)
            for row in self.tiles_by_subset["val"]
        }
        if len(training_keys) != len(self.tiles_by_subset["train"]):
            raise ValueError("FBP fixed training tiles contain duplicates")
        if len(validation_keys) != len(self.tiles_by_subset["val"]):
            raise ValueError("FBP fixed validation tiles contain duplicates")
        if training_keys & validation_keys:
            raise ValueError("FBP fixed train/validation tiles overlap")
        if subset not in self.tiles_by_subset:
            raise ValueError("FBP fixed tile subset must be train or val")
        self.subset = subset
        self.tiles = self.tiles_by_subset[subset]
        self.class_presence = self.presence_by_subset[subset]
        if not len(self.tiles):
            raise ValueError(f"FBP fixed {subset} tile subset is empty")
        if not self.class_presence.any(axis=0).all():
            raise ValueError(
                f"FBP fixed {subset} tile subset does not cover all 24 classes"
            )
        self.class_counts = self.class_presence.sum(axis=0, dtype=np.int64)
        self.rows_by_scene: dict[str, np.ndarray] = {}
        for scene_index, scene_id in enumerate(self.scene_ids):
            rows = np.flatnonzero(self.tiles[:, 0] == scene_index)
            if len(rows):
                self.rows_by_scene[scene_id] = rows

    def __len__(self) -> int:
        return len(self.tiles)

    def tile_at(self, index: int) -> tuple[str, int, int]:
        scene_index, top, left = self.tiles[int(index)]
        return self.scene_ids[int(scene_index)], int(top), int(left)

    def sample_for_scene(
        self,
        scene_id: str,
        rng: random.Random,
        *,
        rare_probability: float,
    ) -> tuple[int, int]:
        rows = self.rows_by_scene.get(scene_id)
        if rows is None or not len(rows):
            raise ValueError(
                f"FBP fixed {self.subset} subset has no tile for {scene_id}"
            )
        selected_row: int
        if rare_probability > 0 and rng.random() < rare_probability:
            scene_presence = self.class_presence[rows]
            available = np.flatnonzero(scene_presence.any(axis=0))
            weights = [
                1.0 / math.sqrt(max(int(self.class_counts[class_id]), 1))
                for class_id in available
            ]
            class_id = rng.choices(available.tolist(), weights=weights, k=1)[0]
            class_rows = rows[np.flatnonzero(scene_presence[:, class_id])]
            selected_row = int(class_rows[rng.randrange(len(class_rows))])
        else:
            selected_row = int(rows[rng.randrange(len(rows))])
        _, top, left = self.tiles[selected_row]
        return int(top), int(left)


class FBPTruthProbabilityStore:
    """Authenticated, crop-addressable truth-class probability maps.

    Each scene is stored as one ``.npy`` array with shape ``[height, width]``.
    Only the probability assigned by a frozen MS75 producer to the ground-truth
    class is retained, so the training artifact is small enough to stream while
    preserving the exact equal-probability fusion objective.  The manifest is
    authenticated before any map is opened and binds the producer provenance,
    map dtype, shape, and file digest.
    """

    ROLE = "ms75_truth_class_probability_maps"
    SEMANTICS = "softmax_of_overlap_normalized_logits_at_ground_truth_class"
    SUPPORTED_DTYPES = {"float16", "float32"}

    @staticmethod
    def _validated_sha256(value: object, *, field: str) -> str:
        digest = str(value)
        if digest != digest.lower() or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
        return digest

    @staticmethod
    def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_manifest_sha256: str,
        expected_scene_ids: Iterable[str],
        expected_hw: tuple[int, int],
        expected_provenance: str,
        expected_producer_checkpoint_sha256: str,
        expected_producer_config_sha256: str,
        expected_scene_count: int = 120,
        verify_file_sha256: bool = True,
        cache_scenes: int = 2,
        expected_artifact_role: str | None = None,
        expected_probability_semantics: str | None = None,
        expected_producer_architecture: str = "ms75_multiscale",
        expected_producer_fields: Mapping[str, object] | None = None,
        expected_inference_fields: Mapping[str, object] | None = None,
        require_embedded_manifest_sha256: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"missing MS75 truth-probability manifest: {self.manifest_path}"
            )
        expected_digest = self._validated_sha256(
            expected_manifest_sha256,
            field="MS75 truth-probability manifest SHA256",
        )
        digest = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError(
                "MS75 truth-probability manifest SHA256 mismatch: "
                f"expected={expected_digest}, observed={digest}"
            )
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid MS75 truth-probability manifest: {self.manifest_path}"
            ) from error
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
            raise ValueError(
                "MS75 truth-probability manifest requires schema_version=1"
            )
        artifact_role = (
            self.ROLE
            if expected_artifact_role is None
            else str(expected_artifact_role)
        )
        probability_semantics = (
            self.SEMANTICS
            if expected_probability_semantics is None
            else str(expected_probability_semantics)
        )
        if payload.get("artifact_role") != artifact_role:
            raise ValueError("unexpected MS75 truth-probability artifact role")
        if payload.get("probability_semantics") != probability_semantics:
            raise ValueError("unexpected MS75 truth-probability semantics")
        if require_embedded_manifest_sha256:
            supplied_inner_digest = self._validated_sha256(
                payload.get("manifest_sha256", ""),
                field="truth-probability embedded manifest SHA256",
            )
            manifest_body = {
                key: value
                for key, value in payload.items()
                if key != "manifest_sha256"
            }
            canonical = json.dumps(
                manifest_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            observed_inner_digest = hashlib.sha256(canonical).hexdigest()
            if observed_inner_digest != supplied_inner_digest:
                raise ValueError(
                    "truth-probability embedded manifest SHA256 mismatch"
                )
        if str(payload.get("source_provenance", "")) != str(expected_provenance):
            raise ValueError(
                "MS75 truth-probability producer provenance mismatch: "
                f"expected={expected_provenance!r}, "
                f"observed={payload.get('source_provenance')!r}"
            )

        expected = [str(value) for value in expected_scene_ids]
        if not expected:
            raise ValueError("expected truth-probability scene IDs must not be empty")
        if len(set(expected)) != len(expected):
            raise ValueError("expected truth-probability scene IDs contain duplicates")
        if isinstance(expected_scene_count, bool) or int(expected_scene_count) <= 0:
            raise ValueError("expected truth-probability scene count must be positive")
        if len(expected) != int(expected_scene_count):
            raise ValueError(
                "truth-probability store scene-count contract mismatch: "
                f"expected={int(expected_scene_count)}, observed={len(expected)}"
            )
        if payload.get("official30_read") is not False:
            raise ValueError("truth-probability manifest must attest official30_read=false")
        manifest_scene_count = payload.get("scene_count")
        if isinstance(manifest_scene_count, bool) or manifest_scene_count != len(
            expected
        ):
            raise ValueError("truth-probability manifest scene_count mismatch")
        manifest_scene_ids = payload.get("scene_ids")
        if not isinstance(manifest_scene_ids, list) or [
            str(value) for value in manifest_scene_ids
        ] != expected:
            raise ValueError(
                "truth-probability manifest scene_ids must exactly match the "
                "configured ordered training split"
            )

        expected_checkpoint_digest = self._validated_sha256(
            expected_producer_checkpoint_sha256,
            field="expected MS75 producer checkpoint SHA256",
        )
        expected_config_digest = self._validated_sha256(
            expected_producer_config_sha256,
            field="expected MS75 producer config SHA256",
        )
        producer = payload.get("producer")
        if not isinstance(producer, dict):
            raise ValueError("truth-probability manifest producer must be a mapping")
        producer_contract = {
            "architecture": str(expected_producer_architecture),
            "checkpoint_sha256": expected_checkpoint_digest,
            "config_sha256": expected_config_digest,
            "state_container": "model",
            "strict_load": True,
        }
        if expected_producer_fields is not None:
            producer_contract.update(dict(expected_producer_fields))
        for key, expected_value in producer_contract.items():
            observed_value = producer.get(key)
            bool_type_mismatch = isinstance(expected_value, bool) and not isinstance(
                observed_value, bool
            )
            if bool_type_mismatch or observed_value != expected_value:
                raise ValueError(
                    f"truth-probability producer {key} mismatch: "
                    f"expected={expected_value!r}, observed={observed_value!r}"
                )

        inference = payload.get("inference")
        if not isinstance(inference, dict):
            raise ValueError("truth-probability manifest inference must be a mapping")
        inference_contract = dict(expected_inference_fields) if (
            expected_inference_fields is not None
        ) else {
            "tta_transforms": ["none"],
            "tile_stride": 256,
            "blend_window": "hann",
            "hann_floor": 0.05,
            "logit_accumulation": "FP32",
            "normalization_before_softmax": True,
            "truth_probability_storage": "float16",
        }
        for key, expected_value in inference_contract.items():
            observed_value = inference.get(key)
            bool_type_mismatch = isinstance(expected_value, bool) and not isinstance(
                observed_value, bool
            )
            if bool_type_mismatch or observed_value != expected_value:
                raise ValueError(
                    f"truth-probability inference {key} mismatch: "
                    f"expected={expected_value!r}, observed={observed_value!r}"
                )
        if not isinstance(verify_file_sha256, bool) or not verify_file_sha256:
            raise ValueError(
                "truth-probability files require full SHA256 verification"
            )

        expected_height, expected_width = (int(value) for value in expected_hw)
        if expected_height <= 0 or expected_width <= 0:
            raise ValueError("expected truth-probability map dimensions must be positive")
        self.expected_hw = (expected_height, expected_width)
        if int(cache_scenes) <= 0:
            raise ValueError("truth-probability cache_scenes must be positive")
        self.cache_scenes = int(cache_scenes)

        rows = payload.get("scenes")
        if not isinstance(rows, list) or not rows:
            raise ValueError("truth-probability manifest scenes must be a non-empty list")
        row_scene_ids: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("truth-probability scene record must be a mapping")
            row_scene_ids.append(str(row.get("scene_id", "")))
        if row_scene_ids != expected:
            raise ValueError(
                "truth-probability scene records must exactly match the configured "
                "ordered training split"
            )
        records: dict[str, dict[str, object]] = {}
        manifest_root = self.manifest_path.parent
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("truth-probability scene record must be a mapping")
            scene_id = str(row.get("scene_id", ""))
            if not scene_id or scene_id in records:
                raise ValueError(
                    f"missing or duplicate truth-probability scene ID: {scene_id!r}"
                )
            relative_path = Path(str(row.get("path", "")))
            if relative_path.is_absolute() or not str(relative_path):
                raise ValueError(
                    f"truth-probability path must be relative for scene {scene_id}"
                )
            map_path = (manifest_root / relative_path).resolve()
            try:
                map_path.relative_to(manifest_root.resolve())
            except ValueError as error:
                raise ValueError(
                    f"truth-probability path escapes manifest root for {scene_id}"
                ) from error
            if not map_path.is_file():
                raise FileNotFoundError(
                    f"missing truth-probability map for {scene_id}: {map_path}"
                )
            shape = row.get("shape")
            if not isinstance(shape, list) or [int(value) for value in shape] != [
                expected_height,
                expected_width,
            ]:
                raise ValueError(
                    f"truth-probability manifest shape mismatch for {scene_id}"
                )
            dtype = str(row.get("dtype", ""))
            if dtype not in self.SUPPORTED_DTYPES:
                raise ValueError(
                    f"unsupported truth-probability dtype for {scene_id}: {dtype!r}"
                )
            file_digest = self._validated_sha256(
                row.get("sha256", ""),
                field=f"truth-probability file SHA256 for {scene_id}",
            )
            observed_file_digest = self._sha256_file(map_path)
            if observed_file_digest != file_digest:
                raise ValueError(
                    f"truth-probability file SHA256 mismatch for {scene_id}: "
                    f"expected={file_digest}, observed={observed_file_digest}"
                )
            array = np.load(map_path, mmap_mode="r", allow_pickle=False)
            if tuple(int(value) for value in array.shape) != self.expected_hw:
                raise ValueError(
                    f"truth-probability array shape mismatch for {scene_id}"
                )
            if np.dtype(array.dtype).name != dtype:
                raise ValueError(
                    f"truth-probability array dtype mismatch for {scene_id}"
                )
            records[scene_id] = {
                "path": map_path,
                "dtype": dtype,
                "sha256": file_digest,
            }
            del array

        self.records = {scene_id: records[scene_id] for scene_id in expected}
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arrays"] = OrderedDict()
        return state

    def close(self) -> None:
        for array in self._arrays.values():
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        self._arrays = OrderedDict()

    def _array(self, scene_id: str) -> np.ndarray:
        array = self._arrays.pop(scene_id, None)
        if array is None:
            record = self.records.get(scene_id)
            if record is None:
                raise KeyError(f"no truth-probability map for scene {scene_id}")
            array = np.load(
                Path(record["path"]), mmap_mode="r", allow_pickle=False
            )
        self._arrays[scene_id] = array
        while len(self._arrays) > self.cache_scenes:
            _, evicted = self._arrays.popitem(last=False)
            memory_map = getattr(evicted, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        return array

    def crop(self, scene_id: str, window: Window) -> np.ndarray:
        top = int(window.row_off)
        left = int(window.col_off)
        height = int(window.height)
        width = int(window.width)
        if top < 0 or left < 0 or height <= 0 or width <= 0:
            raise ValueError("invalid truth-probability crop window")
        array = self._array(str(scene_id))
        if top + height > array.shape[0] or left + width > array.shape[1]:
            raise ValueError(
                f"truth-probability crop exceeds scene {scene_id}: "
                f"top={top}, left={left}, height={height}, width={width}"
            )
        crop = np.asarray(
            array[top : top + height, left : left + width], dtype=np.float32
        )
        if crop.shape != (height, width):
            raise RuntimeError("truth-probability crop has an unexpected shape")
        if not np.isfinite(crop).all() or np.any(crop < 0.0) or np.any(crop > 1.0):
            raise ValueError(
                f"truth-probability crop for {scene_id} is not finite in [0, 1]"
            )
        return np.ascontiguousarray(crop)


class FBPMSDataset(Dataset):
    """Cache-friendly dynamic crops from an official FBP scene protocol."""

    def __init__(self, config: dict, mode: str):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"unsupported mode: {mode}")
        self.config = config
        self.mode = mode
        data = config["data"]
        self.root = Path(data["root"])
        self.patch_size = int(data["patch_size"])
        self.sar_size = configured_sar_size(config)
        self.seed = int(config["experiment"]["seed"])
        self.epoch = 0
        self.scene_ids = read_fbp_split(self.root, str(data[f"{mode}_split"]))
        debug_scene_limit = data.get("debug_scene_limit")
        if debug_scene_limit is not None:
            self.scene_ids = self.scene_ids[: int(debug_scene_limit)]
        self.samples_per_epoch = int(
            data.get("debug_samples_per_epoch", data["samples_per_epoch"])
            if mode == "train"
            else data["eval_patches_per_scene"] * len(self.scene_ids)
        )
        self.scene_block_size = int(data.get("scene_block_size", 32))
        if self.scene_block_size <= 0:
            raise ValueError("data.scene_block_size must be positive")
        self.sampling_config = data.get("sampling", {})
        self.sampling_strategy = str(
            self.sampling_config.get("strategy", "uniform")
        )
        sampling_index = data.get("sampling_index")
        self.sampling_index = (
            FBPSamplingIndex(self.root / str(sampling_index))
            if mode == "train" and sampling_index
            else None
        )
        if mode == "train" and self.sampling_strategy == (
            "neutral_scene_class_uniform"
        ):
            from .ms75_neutral_oof import validate_neutral_sampling_config

            validate_neutral_sampling_config(self.sampling_config)
            if self.sampling_index is None:
                raise ValueError("neutral OOF sampling requires its fold index")
            if self.sampling_index.scene_ids != self.scene_ids:
                raise ValueError(
                    "neutral OOF sampling index scene order must equal fold train"
                )
            if self.sampling_index.source_split != "official_train_oof_fold":
                raise ValueError(
                    "neutral OOF sampling index must be fold-train-only"
                )
        fixed_tile_path = data.get("fixed_tile_index")
        fixed_tile_subsets = data.get(
            "fixed_tile_subsets", {"train": "train", "val": "val"}
        )
        fixed_subset = (
            str(fixed_tile_subsets[mode])
            if fixed_tile_path and mode in {"train", "val"}
            else None
        )
        if fixed_subset is not None:
            resolved_fixed_tile_path = Path(str(fixed_tile_path))
            if not resolved_fixed_tile_path.is_absolute():
                resolved_fixed_tile_path = self.root / resolved_fixed_tile_path
            self.fixed_tiles = FBPFixedTileIndex(
                resolved_fixed_tile_path, fixed_subset
            )
            if self.fixed_tiles.scene_ids != self.scene_ids:
                raise ValueError(
                    "FBP fixed tile scene order does not match the configured split"
                )
            if self.fixed_tiles.crop_size != self.patch_size:
                raise ValueError(
                    "FBP fixed tile crop size does not match data.patch_size"
                )
            expected_height, expected_width = (
                int(value)
                for value in data.get("expected_hw", [6908, 7300])
            )
            fixed_rows = self.fixed_tiles.tiles
            if (
                int(fixed_rows[:, 1].max()) + self.patch_size > expected_height
                or int(fixed_rows[:, 2].max()) + self.patch_size > expected_width
            ):
                raise ValueError(
                    "FBP fixed tile coordinates exceed data.expected_hw"
                )
            missing_scenes = set(self.scene_ids) - set(
                self.fixed_tiles.rows_by_scene
            )
            if missing_scenes:
                raise ValueError(
                    "FBP fixed tile subset misses configured scenes: "
                    f"{sorted(missing_scenes)[:5]}"
                )
        else:
            self.fixed_tiles = None
        self.exact_fixed_tiles = bool(
            self.fixed_tiles is not None
            and mode == "val"
            and data.get("validation_exact_fixed_tiles", True)
        )
        self.fixed_tile_rare_probability = float(
            data.get("fixed_tile_rare_probability", 0.0)
        )
        self.debug_fixed_tile_limit = data.get("debug_fixed_tile_limit")
        if self.exact_fixed_tiles:
            exact_count = len(self.fixed_tiles)
            if self.debug_fixed_tile_limit is not None:
                exact_count = min(exact_count, int(self.debug_fixed_tile_limit))
            self.samples_per_epoch = exact_count
        self.validation_holdout_fraction = float(
            data.get("validation_holdout_fraction", 0.0)
        )
        self.validation_grid_size = int(
            data.get("validation_grid_size", self.patch_size)
        )
        if self.validation_grid_size <= 0:
            raise ValueError("data.validation_grid_size must be positive")
        truth_manifest = data.get("ms75_truth_probability_manifest")
        if truth_manifest and mode == "train":
            expected_height, expected_width = (
                int(value) for value in data.get("expected_hw", [6908, 7300])
            )
            self.ms75_truth_probability_store = FBPTruthProbabilityStore(
                truth_manifest,
                expected_manifest_sha256=str(
                    data.get("ms75_truth_probability_manifest_sha256", "")
                ),
                expected_scene_ids=self.scene_ids,
                expected_hw=(expected_height, expected_width),
                expected_provenance=str(
                    data.get("ms75_truth_probability_expected_provenance", "")
                ),
                expected_producer_checkpoint_sha256=str(
                    data.get(
                        "ms75_truth_probability_expected_checkpoint_sha256",
                        "",
                    )
                ),
                expected_producer_config_sha256=str(
                    data.get(
                        "ms75_truth_probability_expected_config_sha256",
                        "",
                    )
                ),
                expected_scene_count=120,
                verify_file_sha256=data.get(
                    "ms75_truth_probability_verify_file_sha256", False
                ),
                cache_scenes=int(
                    data.get("ms75_truth_probability_cache_scenes", 2)
                ),
            )
        else:
            self.ms75_truth_probability_store = None
        local512_truth_manifest = data.get(
            "local512_truth_probability_manifest"
        )
        if local512_truth_manifest and mode == "train":
            expected_height, expected_width = (
                int(value) for value in data.get("expected_hw", [6908, 7300])
            )
            self.local512_truth_probability_store = FBPTruthProbabilityStore(
                local512_truth_manifest,
                expected_manifest_sha256=str(
                    data.get(
                        "local512_truth_probability_manifest_sha256", ""
                    )
                ),
                expected_scene_ids=self.scene_ids,
                expected_hw=(expected_height, expected_width),
                expected_provenance=str(
                    data.get(
                        "local512_truth_probability_expected_provenance", ""
                    )
                ),
                expected_producer_checkpoint_sha256=str(
                    data.get(
                        "local512_truth_probability_expected_checkpoint_sha256",
                        "",
                    )
                ),
                expected_producer_config_sha256=str(
                    data.get(
                        "local512_truth_probability_expected_config_sha256",
                        "",
                    )
                ),
                expected_scene_count=120,
                verify_file_sha256=data.get(
                    "local512_truth_probability_verify_file_sha256", False
                ),
                cache_scenes=int(
                    data.get("local512_truth_probability_cache_scenes", 2)
                ),
                expected_artifact_role=(
                    "local512_truth_class_probability_maps"
                ),
                expected_probability_semantics=(
                    "softmax_T1_of_overlap_normalized_logits_at_ground_truth_class"
                ),
                expected_producer_architecture="local_detail_cnxb",
                expected_producer_fields={
                    "profile_name": "full120_epoch30_selected_17999",
                },
                expected_inference_fields={
                    "tta_transforms": ["none"],
                    "patch_size": 512,
                    "tile_stride": 256,
                    "blend_window": "hann",
                    "hann_floor": 0.05,
                    "logit_accumulation": "FP32",
                    "normalization_before_softmax": True,
                    "softmax_temperature": 1.0,
                    "runtime_batch_size": 4,
                    "truth_probability_storage": "float16",
                },
                require_embedded_manifest_sha256=True,
            )
        else:
            self.local512_truth_probability_store = None
        self._readers: OrderedDict[str, FBPSceneReader] = OrderedDict()
        self._eval_windows = self._build_eval_windows() if mode != "train" else []

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_readers"] = OrderedDict()
        return state

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        for reader in getattr(self, "_readers", {}).values():
            reader.close()
        self._readers = OrderedDict()
        truth_store = getattr(self, "ms75_truth_probability_store", None)
        if truth_store is not None:
            truth_store.close()
        local512_truth_store = getattr(
            self, "local512_truth_probability_store", None
        )
        if local512_truth_store is not None:
            local512_truth_store.close()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _reader(self, scene_id: str) -> FBPSceneReader:
        reader = self._readers.pop(scene_id, None)
        if reader is None:
            reader = FBPSceneReader(self.root, scene_id, self.config["data"])
        self._readers[scene_id] = reader
        maximum = int(self.config["data"].get("cache_scenes", 1))
        while len(self._readers) > maximum:
            _, expired = self._readers.popitem(last=False)
            expired.close()
        return reader

    def _scene_for_index(self, index: int) -> str:
        block = index // self.scene_block_size
        cycle, position = divmod(block, len(self.scene_ids))
        order = list(self.scene_ids)
        random.Random(
            self.seed + self.epoch * 1_000_003 + cycle * 65_537
        ).shuffle(order)
        return order[position]

    def _random_window(
        self, reader: FBPSceneReader, rng: random.Random
    ) -> Window:
        maximum_left = reader.width - self.patch_size
        maximum_top = reader.height - self.patch_size
        return Window(
            rng.randint(0, maximum_left),
            rng.randint(0, maximum_top),
            self.patch_size,
            self.patch_size,
        )

    def _acceptable_window(
        self,
        reader: FBPSceneReader,
        rng: random.Random,
        *,
        scene_id: str,
        validation_partition: bool | None = None,
    ) -> Window:
        threshold = float(self.config["data"]["min_labeled_fraction"])
        attempts = int(self.config["data"]["sampling_attempts"])
        best: Window | None = None
        best_fraction = -1.0
        # Partition filtering and labeled-fraction filtering are independent;
        # allow enough draws for a 20% validation partition without silently
        # falling back to the wrong side of the split.
        draws = attempts * (8 if validation_partition is not None else 1)
        for _ in range(draws):
            window = self._random_window(reader, rng)
            if (
                validation_partition is not None
                and self._is_validation_window(scene_id, window)
                != validation_partition
            ):
                continue
            _, supervised = reader.read_label_valid(window)
            fraction = float(supervised.mean())
            if fraction > best_fraction:
                best = window
                best_fraction = fraction
            if fraction >= threshold:
                return window
        if best is None:
            raise RuntimeError(
                f"could not sample the requested validation partition for {scene_id}"
            )
        return best

    def _is_validation_window(self, scene_id: str, window: Window) -> bool:
        if self.validation_holdout_fraction <= 0:
            return False
        center_x = int(window.col_off) + int(window.width) // 2
        center_y = int(window.row_off) + int(window.height) // 2
        cell_x = center_x // self.validation_grid_size
        cell_y = center_y // self.validation_grid_size
        payload = (
            f"{self.seed}:{scene_id}:{cell_y}:{cell_x}"
        ).encode("utf-8")
        bucket = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "little"
        ) / float(2**64)
        return bucket < self.validation_holdout_fraction

    def _directed_window(
        self,
        reader: FBPSceneReader,
        scene_id: str,
        rng: random.Random,
        *,
        sample_index: int | None = None,
    ) -> Window | None:
        del sample_index
        if self.sampling_strategy != "directed" or self.sampling_index is None:
            return None
        draw = rng.random()
        rare_fraction = float(self.sampling_config.get("rare_fraction", 0.0))
        hard_fraction = float(self.sampling_config.get("hard_fraction", 0.0))
        if draw < rare_fraction:
            class_ids = [
                int(value) for value in self.sampling_config["rare_class_ids"]
            ]
        elif draw < rare_fraction + hard_fraction:
            class_ids = [
                int(value) for value in self.sampling_config["hard_class_ids"]
            ]
        else:
            return None
        center = self.sampling_index.sample(scene_id, class_ids, rng)
        if center is None:
            return None
        center_y, center_x = center
        jitter = int(
            self.sampling_config.get(
                "center_jitter", self.patch_size // 4
            )
        )
        center_y += rng.randint(-jitter, jitter) if jitter else 0
        center_x += rng.randint(-jitter, jitter) if jitter else 0
        left = min(
            max(center_x - self.patch_size // 2, 0),
            reader.width - self.patch_size,
        )
        top = min(
            max(center_y - self.patch_size // 2, 0),
            reader.height - self.patch_size,
        )
        return Window(left, top, self.patch_size, self.patch_size)

    def _neutral_class_window(
        self,
        class_id: int,
        rng: random.Random,
    ) -> tuple[str, Window]:
        if self.sampling_index is None:
            raise RuntimeError("neutral OOF sampling index is unavailable")
        scene_id, center_y, center_x = self.sampling_index.sample_class_global(
            class_id, rng
        )
        reader = self._reader(scene_id)
        jitter = int(self.sampling_config.get("center_jitter", 0))
        if jitter:
            center_y += rng.randint(-jitter, jitter)
            center_x += rng.randint(-jitter, jitter)
        left = min(
            max(center_x - self.patch_size // 2, 0),
            reader.width - self.patch_size,
        )
        top = min(
            max(center_y - self.patch_size // 2, 0),
            reader.height - self.patch_size,
        )
        return scene_id, Window(left, top, self.patch_size, self.patch_size)

    def _build_eval_windows(self) -> list[tuple[str, Window]]:
        if self.exact_fixed_tiles:
            assert self.fixed_tiles is not None
            return [
                (
                    scene_id,
                    Window(left, top, self.patch_size, self.patch_size),
                )
                for scene_id, top, left in (
                    self.fixed_tiles.tile_at(index)
                    for index in range(self.samples_per_epoch)
                )
            ]
        windows: list[tuple[str, Window]] = []
        count = int(self.config["data"]["eval_patches_per_scene"])
        for scene_index, scene_id in enumerate(self.scene_ids):
            reader = self._reader(scene_id)
            rng = random.Random(self.seed + 100_003 * (scene_index + 1))
            for _ in range(count):
                windows.append(
                    (
                        scene_id,
                        self._acceptable_window(
                            reader,
                            rng,
                            scene_id=scene_id,
                            validation_partition=(
                                True
                                if self.mode == "val"
                                and self.validation_holdout_fraction > 0
                                else None
                            ),
                        ),
                    )
                )
        self.close()
        return windows

    def __getitem__(
        self, index: int
    ) -> dict[str, torch.Tensor | str | int | float]:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + int(index))
        if self.mode == "train":
            neutral_class_id: int | None = None
            if self.sampling_strategy == "neutral_scene_class_uniform":
                from .ms75_neutral_oof import neutral_assignment

                neutral_arm, neutral_class_id = neutral_assignment(
                    int(index),
                    epoch=self.epoch,
                    samples_per_epoch=self.samples_per_epoch,
                    seed=self.seed,
                )
            else:
                neutral_arm = "legacy"
            if neutral_arm == "class":
                assert neutral_class_id is not None
                scene_id, window = self._neutral_class_window(
                    neutral_class_id, rng
                )
                reader = self._reader(scene_id)
            else:
                scene_id = self._scene_for_index(index)
                reader = self._reader(scene_id)
                window = None
            if self.fixed_tiles is not None:
                top, left = self.fixed_tiles.sample_for_scene(
                    scene_id,
                    rng,
                    rare_probability=self.fixed_tile_rare_probability,
                )
                if (
                    top + self.patch_size > reader.height
                    or left + self.patch_size > reader.width
                ):
                    raise RuntimeError(
                        f"FBP fixed tile is outside scene {scene_id}: "
                        f"top={top}, left={left}"
                    )
                window = Window(
                    left, top, self.patch_size, self.patch_size
                )
            elif neutral_arm != "class":
                window = self._directed_window(
                    reader,
                    scene_id,
                    rng,
                    sample_index=int(index),
                )
            if (
                window is not None
                and self.validation_holdout_fraction > 0
                and self._is_validation_window(scene_id, window)
            ):
                window = None
            if window is None:
                window = self._acceptable_window(
                    reader,
                    rng,
                    scene_id=scene_id,
                    validation_partition=(
                        False
                        if self.validation_holdout_fraction > 0
                        else None
                    ),
                )
        else:
            scene_id, window = self._eval_windows[index]
            reader = self._reader(scene_id)
            if self.fixed_tiles is not None and (
                int(window.row_off) + self.patch_size > reader.height
                or int(window.col_off) + self.patch_size > reader.width
            ):
                raise RuntimeError(
                    f"FBP fixed tile is outside scene {scene_id}: "
                    f"top={int(window.row_off)}, left={int(window.col_off)}"
                )
        sample = reader.read_patch(window, self.sar_size)
        if self.ms75_truth_probability_store is not None:
            sample["ms75_truth_probability"] = (
                self.ms75_truth_probability_store.crop(scene_id, window)
            )
        if self.local512_truth_probability_store is not None:
            sample["local512_truth_probability"] = (
                self.local512_truth_probability_store.crop(scene_id, window)
            )
        if (
            self.ms75_truth_probability_store is not None
            or self.local512_truth_probability_store is not None
        ):
            sample["fusion_truth_probability_valid"] = np.asarray(
                True, dtype=np.bool_
            )
        if self.mode == "train":
            sample = MSSARDataset._augment(self, sample, rng)
        result = {
            "ms": torch.from_numpy(sample["ms"]).float(),
            "ms_context": torch.from_numpy(sample["ms_context"]).float(),
            "ms_context_valid": torch.from_numpy(sample["ms_context_valid"]).bool(),
            "sar": torch.from_numpy(sample["sar"]).float(),
            "target": torch.from_numpy(sample["target"]).long(),
            "supervised": torch.from_numpy(sample["supervised"]).bool(),
            "ms_valid": torch.from_numpy(sample["ms_valid"]).bool(),
            "sar_valid": torch.from_numpy(sample["sar_valid"]).bool(),
            "scene_id": scene_id,
            "left": int(window.col_off),
            "top": int(window.row_off),
            "sar_pairing": "zero",
            "sar_source_scene_id": "none",
            "sar_source_left": 0,
            "sar_source_top": 0,
            "time_gap_days": 0.0,
            "sar_dropped": False,
            "sar_shifted": False,
            "sar_noisy": False,
            "sar_aux_eligible": False,
        }
        if "ms75_truth_probability" in sample:
            result["ms75_truth_probability"] = torch.from_numpy(
                sample["ms75_truth_probability"]
            ).float()
        if "local512_truth_probability" in sample:
            result["local512_truth_probability"] = torch.from_numpy(
                sample["local512_truth_probability"]
            ).float()
        if "fusion_truth_probability_valid" in sample:
            result["fusion_truth_probability_valid"] = torch.as_tensor(
                sample["fusion_truth_probability_valid"], dtype=torch.bool
            )
        return result


class _OffsetSequentialSampler(Sampler[int]):
    """Yield deterministic dataset indices after a completed-batch offset.

    Mid-epoch recovery must not decode and discard thousands of already
    optimized crops before reaching the saved optimizer-step boundary.  The
    training datasets in this repository are index-deterministic and never use
    DataLoader shuffling, so starting the sequential sampler at an aligned
    sample offset preserves the exact remaining crop sequence.
    """

    def __init__(self, data_source: Dataset, start_index: int = 0) -> None:
        self.data_source = data_source
        self.start_index = int(start_index)
        if not 0 <= self.start_index <= len(self.data_source):
            raise ValueError("sampler start_index is outside the dataset")

    def __iter__(self):
        return iter(range(self.start_index, len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index


class _AbsoluteBlockBatchSampler(Sampler[list[int]]):
    """Group sequential samples without crossing absolute block boundaries.

    This is an execution-only I/O optimization.  The flattened index stream is
    exactly the same as :class:`_OffsetSequentialSampler`; grouping merely lets
    one worker retain a large decoded scene while it prepares all samples in a
    locality block.  A mid-epoch start first emits the remainder of its
    absolute block and then resumes full blocks, so recovery cannot skip,
    repeat, or reorder a crop.
    """

    def __init__(
        self,
        data_source: Dataset,
        *,
        start_index: int,
        block_size: int,
        microbatch_size: int,
        drop_last: bool,
    ) -> None:
        self.data_source = data_source
        self.start_index = int(start_index)
        self.block_size = int(block_size)
        self.microbatch_size = int(microbatch_size)
        self.drop_last = bool(drop_last)
        if self.block_size <= 0 or self.microbatch_size <= 0:
            raise ValueError("block and microbatch sizes must be positive")
        if self.block_size % self.microbatch_size:
            raise ValueError("block_size must be a multiple of microbatch_size")
        if not 0 <= self.start_index <= len(self.data_source):
            raise ValueError("sampler start_index is outside the dataset")
        if self.start_index % self.microbatch_size:
            raise ValueError("sampler start_index must lie on a microbatch boundary")
        self.stop_index = len(self.data_source)
        if self.drop_last:
            self.stop_index -= self.stop_index % self.microbatch_size
        if self.start_index > self.stop_index:
            raise ValueError("sampler start_index lies beyond the usable dataset")

    def _ranges(self) -> Iterator[tuple[int, int]]:
        cursor = self.start_index
        while cursor < self.stop_index:
            absolute_boundary = (
                (cursor // self.block_size) + 1
            ) * self.block_size
            end = min(absolute_boundary, self.stop_index)
            if (end - cursor) % self.microbatch_size:
                raise AssertionError("I/O block split a training microbatch")
            yield cursor, end
            cursor = end

    def __iter__(self) -> Iterator[list[int]]:
        for start, end in self._ranges():
            yield list(range(start, end))

    def __len__(self) -> int:
        return sum(1 for _ in self._ranges())


def build_dataloader(
    config: dict,
    mode: str,
    *,
    start_batch_index: int = 0,
    io_batch_size: int | None = None,
    num_workers_override: int | None = None,
) -> DataLoader:
    dataset_format = str(config["data"].get("dataset_format", "ms_sar_manifest"))
    dataset_variant = str(config["data"].get("dataset_variant", "legacy"))
    if dataset_variant == "ms75_multiscale":
        if dataset_format != "fbp_ms":
            raise ValueError("MS75 multiscale data requires dataset_format=fbp_ms")
        # Local import avoids a module cycle: ms75_data deliberately reuses
        # the audited FBPSceneReader above.
        from .ms75_data import MS75MultiScaleDataset

        dataset = MS75MultiScaleDataset(config, mode)
    else:
        dataset = (
            FBPMSDataset(config, mode)
            if dataset_format == "fbp_ms"
            else MSSARDataset(config, mode)
        )
    train = mode == "train"
    start_batch_index = int(start_batch_index)
    if start_batch_index < 0:
        raise ValueError("start_batch_index must be non-negative")
    if not train and start_batch_index:
        raise ValueError("only a training loader may start mid-epoch")
    batch_size = int(
        config["train"]["batch_size"]
        if train
        else config["inference"]["batch_size"]
    )
    start_index = start_batch_index * batch_size
    if start_index > len(dataset):
        raise ValueError(
            "training resume batch lies beyond the configured dataset length"
        )
    num_workers = int(
        config["data"]["num_workers"]
        if num_workers_override is None
        else num_workers_override
    )
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    common_loader_arguments = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        # Workers are recreated each epoch so the deterministic epoch-dependent
        # window plan is propagated to spawned Windows and Linux workers.
        "persistent_workers": False,
    }
    if io_batch_size is not None:
        if not train:
            raise ValueError("I/O super-batches are supported only for training")
        io_batch_size = int(io_batch_size)
        if io_batch_size < batch_size or io_batch_size % batch_size:
            raise ValueError(
                "io_batch_size must be a positive multiple of the training batch size"
            )
        batch_sampler = _AbsoluteBlockBatchSampler(
            dataset,
            start_index=start_index,
            block_size=io_batch_size,
            microbatch_size=batch_size,
            drop_last=True,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **common_loader_arguments,
        )
    sampler = _OffsetSequentialSampler(dataset, start_index=start_index)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        **common_loader_arguments,
        drop_last=train,
    )
