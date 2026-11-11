"""Windowed, label-free SAR/PAN sampling onto the original FBP pixel frame.

The registration maps *pixel centres* in old FBP images to centres in the
orthorectified new MS image.  PAN's 4x relationship is to that new MS image,
not to the old FBP raster.  No images or labels are globally resized here.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from scipy.ndimage import map_coordinates


SCHEMA = "fbp_old_to_new_ms_projective_grid_v2"
CONTRACT = "dual_aux_pixel_center_pull_v1"
DEFAULT_OUT_SIZE = {"sar": 84, "pan": 1024}
MAX_WINDOW_PIXELS = 16 * 1024 * 1024


def _aux_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    return root if (root / "availability.csv").is_file() else root / "auxiliary"


def _asset_path(root: Path, relative: str, *, require_exists: bool = True) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"auxiliary asset must have a nonempty relative path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"auxiliary asset escapes its root: {relative!r}")
    if require_exists and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _flag(value: str) -> bool:
    value = str(value).strip().lower()
    if value not in {"yes", "no", "true", "false", "1", "0"}:
        raise ValueError(f"invalid auxiliary availability flag: {value!r}")
    return value in {"yes", "true", "1"}


def _availability(root: Path) -> list[dict[str, str]]:
    with (root / "availability.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scene_stem", "numeric_scene_id", "has_pan", "has_sar"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("auxiliary availability.csv lacks scene/availability fields")
        rows = list(reader)
    stems = [r["scene_stem"] for r in rows]
    numbers = [str(int(r["numeric_scene_id"])) for r in rows]
    if len(set(stems)) != len(rows) or len(set(numbers)) != len(rows):
        raise ValueError("auxiliary availability contains duplicate scene identities")
    for row in rows:
        _flag(row["has_sar"])
        _flag(row["has_pan"])
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _centres(top: int, left: int, size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = (np.arange(out_size, dtype=np.float64) + 0.5) * size / out_size - 0.5
    x, y = np.meshgrid(left + offsets, top + offsets)
    return x.ravel(), y.ravel()


def _snap_integer(array: np.ndarray) -> np.ndarray:
    rounded = np.rint(array)
    return np.where(np.abs(array - rounded) < 1e-7, rounded, array)


def _bilinear_support(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Every source pixel with nonzero interpolation weight must be valid."""
    x, y = _snap_integer(x), _snap_integer(y)
    good = np.isfinite(x) & np.isfinite(y)
    good &= (x >= 0) & (y >= 0) & (x <= mask.shape[1] - 1) & (y <= mask.shape[0] - 1)
    safe_x = np.clip(np.where(good, x, 0), 0, mask.shape[1] - 1)
    safe_y = np.clip(np.where(good, y, 0), 0, mask.shape[0] - 1)
    x0, x1 = np.floor(safe_x).astype(np.int64), np.ceil(safe_x).astype(np.int64)
    y0, y1 = np.floor(safe_y).astype(np.int64), np.ceil(safe_y).astype(np.int64)
    return good & mask[y0, x0] & mask[y0, x1] & mask[y1, x0] & mask[y1, x1]


def _bilinear_support_grid(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Equivalent support for a Cartesian grid, without dense float/index grids.

    ``x`` and ``y`` are its one-dimensional coordinate axes. Integer snapping
    and all four neighbours deliberately match ``_bilinear_support`` exactly.
    The general projective-coordinate case still uses the original function.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Cartesian support axes must be one-dimensional")
    x, y = _snap_integer(x), _snap_integer(y)
    good_x = np.isfinite(x) & (x >= 0) & (x <= mask.shape[1] - 1)
    good_y = np.isfinite(y) & (y >= 0) & (y <= mask.shape[0] - 1)
    safe_x = np.clip(np.where(good_x, x, 0), 0, mask.shape[1] - 1)
    safe_y = np.clip(np.where(good_y, y, 0), 0, mask.shape[0] - 1)
    x0, x1 = np.floor(safe_x).astype(np.int64), np.ceil(safe_x).astype(np.int64)
    y0, y1 = np.floor(safe_y).astype(np.int64), np.ceil(safe_y).astype(np.int64)
    return (good_y[:, None] & good_x[None, :]
            & mask[y0[:, None], x0[None, :]] & mask[y0[:, None], x1[None, :]]
            & mask[y1[:, None], x0[None, :]] & mask[y1[:, None], x1[None, :]])


class AuxiliaryScene:
    """Lazy, bounded native-raster reader for one availability.csv scene.

    ``read_patch`` returns numpy arrays. ``aux`` and ``native_valid`` describe
    the regular crop-support lattice at ``out_size``; ``valid`` is a
    conservative mask on the requested old-FBP ``size`` lattice.  A missing
    modality returns exact zeros; a declared available but missing or corrupt
    asset fails closed when that modality is requested. Unused modality rasters
    need not be present locally. Shared metadata/registration remains mandatory
    and is validated eagerly. ``quality`` is diagnostic metadata, not a model
    input or label.
    """

    def __init__(self, root: str | Path, scene_id: str | int, *, fast_sampling: bool = False) -> None:
        # Explicit opt-in retains the audited implementation as a parity
        # baseline. This affects work performed, never geometry or validity.
        if type(fast_sampling) is not bool:
            raise ValueError("fast_sampling must be a boolean")
        self.fast_sampling = fast_sampling
        self.root = _aux_root(root)
        query = str(scene_id)
        if query.isdecimal():
            query = str(int(query))
        matches = [r for r in _availability(self.root)
                   if r["scene_stem"] == query or str(int(r["numeric_scene_id"])) == query]
        if len(matches) != 1:
            raise KeyError(f"scene is not uniquely identified by auxiliary availability: {scene_id}")
        self.row = matches[0]
        self.scene_id = self.row["scene_stem"]
        self.numeric_scene_id = int(self.row["numeric_scene_id"])
        self.available = {mod: _flag(self.row[f"has_{mod}"]) for mod in ("sar", "pan")}
        self._datasets: dict[str, tuple[Any, Any]] = {}
        self._closed = False
        self._paths: dict[str, tuple[Path, Path]] = {}
        self.registration: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.registration_path: Path | None = None
        self.registration_sha256: str | None = None
        if any(self.available.values()):
            for mod, present in self.available.items():
                if present:
                    self._paths[mod] = (
                        _asset_path(self.root, self.row.get(f"{mod}_path", ""), require_exists=False),
                        _asset_path(self.root, self.row.get(f"{mod}_valid_path", ""), require_exists=False),
                    )
            metadata_path = self.row.get("metadata_path", "")
            if metadata_path:
                self.metadata = json.loads(_asset_path(self.root, metadata_path).read_text(encoding="utf-8-sig"))
            self._load_registration()

    def _load_registration(self) -> None:
        path = _asset_path(self.root, self.row.get("registration_path", ""))
        if path.suffix != ".npz":
            raise ValueError("registration_path must refer to the paired NPZ")
        record_path = path.with_suffix(".json")
        record = json.loads(record_path.read_text(encoding="utf-8-sig"))
        if (record.get("schema_version") != SCHEMA or record.get("status") != "PASS"
                or record.get("label_usage") is not False):
            raise ValueError("registration must be PASS, v2 and explicitly label-free")
        if (record.get("scene_stem") != self.scene_id
                or int(record.get("numeric_scene_id", -1)) != self.numeric_scene_id):
            raise ValueError("registration scene identity differs from availability")
        for mod in ("sar", "pan"):
            if record.get(f"has_{mod}") is not self.available[mod]:
                raise ValueError(f"registration {mod} availability mismatch")
        expected_sha = record.get("npz_sha256")
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                or _sha256(path) != expected_sha):
            raise ValueError("registration NPZ SHA256 mismatch or missing pin")
        reg = record["registration"]
        if reg.get("status") != "PASS" or reg.get("failure_reasons"):
            raise ValueError("registration contains failed geometric quality checks")
        with np.load(path, allow_pickle=False) as archive:
            def scalar(key: str) -> Any:
                return archive[key].item()
            if (scalar("schema_version") != SCHEMA or scalar("status") != "PASS"
                    or bool(scalar("label_usage")) or scalar("scene_stem") != self.scene_id
                    or int(scalar("numeric_scene_id")) != self.numeric_scene_id
                    or scalar("record_sha256") != record.get("record_sha256")):
                raise ValueError("registration JSON and NPZ identity/label contract disagree")
            convention = str(scalar("residual_grid_coordinate_convention"))
            contract = str(scalar("residual_grid_contract"))
            if "zero-based full-resolution pixel centers" not in convention or "then add bilinearly sampled residual" not in contract:
                raise ValueError("unknown registration centre or residual convention")
            self.old_shape = tuple(int(v) for v in archive["old_shape"])
            self.new_ms_shape = tuple(int(v) for v in archive["new_ms_shape"])
            self._homography = np.array(archive["old_to_new_ms"], dtype=np.float64)
            self._residual = np.array(archive["old_to_new_ms_residual_grid"], dtype=np.float32)
            self._extent = np.array(archive["residual_grid_old_extent"], dtype=np.float64)
            transform = np.array(archive["new_ms_transform"], dtype=np.float64)
            self._new_crs = CRS.from_string(str(scalar("new_ms_crs_wkt")))
        if (len(self.old_shape) != 2 or len(self.new_ms_shape) != 2
                or min(*self.old_shape, *self.new_ms_shape) < 2
                or self._homography.shape != (3, 3) or self._residual.ndim != 3
                or self._residual.shape[-1] != 2 or min(self._residual.shape[:2]) < 2
                or self._extent.shape != (4,) or transform.shape != (6,)):
            raise ValueError("invalid registration shapes")
        for array in (self._homography, self._residual, self._extent, transform):
            if not np.isfinite(array).all():
                raise ValueError("registration contains non-finite geometry")
        expected_extent = [0, 0, self.old_shape[1] - 1, self.old_shape[0] - 1]
        if not np.allclose(self._extent, expected_extent, atol=1e-8, rtol=0):
            raise ValueError("forward residual grid must span old image centre endpoints")
        for key, observed in (("old_shape", self.old_shape), ("new_ms_shape", self.new_ms_shape),
                              ("old_to_new_ms", self._homography), ("new_ms_transform", transform)):
            if not np.array_equal(np.asarray(reg[key]), np.asarray(observed)):
                raise ValueError(f"registration JSON/NPZ geometry mismatch: {key}")
        if CRS.from_string(reg["new_ms_crs_wkt"]) != self._new_crs:
            raise ValueError("registration JSON/NPZ CRS mismatch")
        self._new_transform = Affine(*transform)
        if abs(self._new_transform.determinant) < 1e-12 or abs(np.linalg.det(self._homography)) < 1e-12:
            raise ValueError("registration has a singular transform")
        self.registration = record
        self.registration_path, self.registration_sha256 = path, expected_sha

    def _open(self, modality: str) -> tuple[Any, Any]:
        if modality in self._datasets:
            return self._datasets[modality]
        # Keep published availability and registration hashes unchanged. A
        # removed, unused PAN asset must not block a SAR-only reader, but it
        # must never be silently treated as a valid zero-valued PAN input.
        # Resolve again here to retain the root/symlink safety check at use.
        image_path = _asset_path(self.root, self.row.get(f"{modality}_path", ""))
        valid_path = _asset_path(self.root, self.row.get(f"{modality}_valid_path", ""))
        image = rasterio.open(image_path)
        valid = None
        try:
            valid = rasterio.open(valid_path)
            if (image.crs != self._new_crs or image.crs != valid.crs
                    or (image.height, image.width) != (valid.height, valid.width)
                    or not np.allclose(tuple(image.transform), tuple(valid.transform), rtol=0, atol=1e-8)
                    or valid.count != 1 or abs(image.transform.determinant) < 1e-12):
                raise ValueError(f"{modality} raster/validity/registration grids disagree")
            if modality == "sar":
                descriptions = tuple(str(s or "").upper() for s in image.descriptions)
                if (image.count != 2 or not all(np.dtype(t).kind == "f" for t in image.dtypes)
                        or not all(np.isclose(v, 10.0, rtol=0, atol=1e-6) for v in image.res)
                        or "VV" not in descriptions[0] or "VH" not in descriptions[1]
                        or image.tags().get("VALUE_SCALE") != "Sigma0_dB"):
                    raise ValueError("SAR must be native 10m float VV,VH Sigma0_dB")
            else:
                if (image.count != 1 or image.dtypes != ("uint16",)
                        or not all(np.isclose(v, 0.81, rtol=0, atol=1e-6) for v in image.res)
                        or image.tags().get("NATIVE_GSD_M") != "0.81"):
                    raise ValueError("PAN must be native 0.81m single-band uint16")
            self._datasets[modality] = (image, valid)
            return image, valid
        except Exception:
            image.close()
            if valid is not None:
                valid.close()
            raise

    def _native_coordinates(self, x: np.ndarray, y: np.ndarray, transform: Affine) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        old_valid = ((x >= 0) & (y >= 0) & (x <= self.old_shape[1] - 1)
                     & (y <= self.old_shape[0] - 1))
        gx = np.clip(x, 0, self.old_shape[1] - 1) * (self._residual.shape[1] - 1) / (self.old_shape[1] - 1)
        gy = np.clip(y, 0, self.old_shape[0] - 1) * (self._residual.shape[0] - 1) / (self.old_shape[0] - 1)
        h = self._homography
        denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        safe = np.abs(denominator) > 1e-12
        divisor = np.where(safe, denominator, 1)
        new_x = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / divisor
        new_y = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / divisor
        new_x += map_coordinates(self._residual[..., 0], [gy, gx], order=1, mode="nearest", prefilter=False)
        new_y += map_coordinates(self._residual[..., 1], [gy, gx], order=1, mode="nearest", prefilter=False)
        good = old_valid & safe & np.isfinite(new_x) & np.isfinite(new_y)
        good &= (new_x >= 0) & (new_y >= 0) & (new_x <= self.new_ms_shape[1] - 1) & (new_y <= self.new_ms_shape[0] - 1)
        # GDAL affines map *corners*. Both half-pixel shifts are essential.
        world_x, world_y = self._new_transform * (new_x + 0.5, new_y + 0.5)
        column, row = (~transform) * (world_x, world_y)
        return _snap_integer(column - 0.5), _snap_integer(row - 0.5), good

    def _sample(self, modality: str, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
        image, mask_image = self._open(modality)
        col, row, good = self._native_coordinates(x, y, image.transform)
        good &= np.isfinite(col) & np.isfinite(row)
        good &= (col >= 0) & (row >= 0) & (col <= image.width - 1) & (row <= image.height - 1)
        # read_patch consumes values only on its first (native) lattice, but
        # needs validity and the bounded read window from BOTH lattices. The
        # scoped count is absent for standalone _sample callers (including
        # wide SAR reads), which continue to receive values at every point.
        value_count = getattr(self, "_sample_value_count", len(x)) if self.fast_sampling else len(x)
        values = np.zeros((image.count, value_count), dtype=np.float32)
        if not good.any():
            return values, good, None
        x0, y0 = int(np.floor(col[good].min())), int(np.floor(row[good].min()))
        x1 = min(image.width, int(np.ceil(col[good].max())) + 1)
        y1 = min(image.height, int(np.ceil(row[good].max())) + 1)
        if (x1 - x0) * (y1 - y0) > MAX_WINDOW_PIXELS:
            raise ValueError("mapped auxiliary window exceeds bounded-read safety limit")
        window = Window(x0, y0, x1 - x0, y1 - y0)
        raw = image.read(window=window).astype(np.float32)
        raster_valid = (image.read_masks(window=window) > 0).all(axis=0)
        raster_valid &= np.isfinite(raw).all(axis=0)
        explicit_mask = mask_image.read(1, window=window)
        raster_valid &= np.isfinite(explicit_mask) & (explicit_mask > 0)
        raster_valid &= mask_image.read_masks(1, window=window) > 0
        for band, nodata in enumerate(image.nodatavals):
            if nodata is not None:
                raster_valid &= raw[band] != nodata
        local_x = np.where(good, col - x0, -1)
        local_y = np.where(good, row - y0, -1)
        good &= _bilinear_support(raster_valid, local_x, local_y)
        # Invalid values must never contaminate neighbouring interpolated data.
        raw[:, ~raster_valid] = 0
        for band in range(image.count):
            values[band] = map_coordinates(raw[band], [local_y[:value_count], local_x[:value_count]], order=1,
                                           mode="constant", cval=0, prefilter=False)
        values[:, ~good[:value_count]] = 0
        return values, good, (x0, y0, x1 - x0, y1 - y0)

    def read_patch(self, top: int, left: int, size: int = 256, modality: str = "sar", out_size: int | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("AuxiliaryScene is closed")
        if modality not in DEFAULT_OUT_SIZE:
            raise ValueError("modality must be 'sar' or 'pan'")
        if out_size is None:
            out_size = DEFAULT_OUT_SIZE[modality]
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (top, left, size, out_size)) or size <= 0 or out_size <= 0:
            raise ValueError("crop positions and positive sizes must be integers")
        if size * size > MAX_WINDOW_PIXELS or out_size * out_size > MAX_WINDOW_PIXELS:
            raise ValueError("auxiliary output exceeds bounded-read safety limit")
        channels = 2 if modality == "sar" else 1
        result: dict[str, Any] = {
            "aux": np.zeros((channels, out_size, out_size), dtype=np.float32),
            "native_valid": np.zeros((1, out_size, out_size), dtype=np.bool_),
            "valid": np.zeros((1, size, size), dtype=np.bool_),
            "available": self.available[modality],
            "quality": {"contract": CONTRACT, "scene_id": self.scene_id, "modality": modality,
                        "label_usage": False, "native_valid_fraction": 0.0, "valid_fraction": 0.0,
                        "read_window": None, "normalization": "clip_db[-30,10];(db+10)/20" if modality == "sar" else "uint16/65535"},
        }
        if not self.available[modality]:
            return result
        nx, ny = _centres(top, left, size, out_size)
        fx, fy = _centres(top, left, size, size)
        if self.fast_sampling:
            self._sample_value_count = len(nx)
            try:
                # Keep dynamic dispatch: SemanticAuxScene captures these same
                # raw native values before read_patch normalizes its result.
                values, support, bbox = self._sample(modality, np.concatenate([nx, fx]), np.concatenate([ny, fy]))
            finally:
                del self._sample_value_count
        else:
            values, support, bbox = self._sample(modality, np.concatenate([nx, fx]), np.concatenate([ny, fy]))
        native_valid = support[:len(nx)].reshape(out_size, out_size)
        fbp_valid = support[len(nx):].reshape(size, size)
        # Preserve validity through the feature lattice resize as well as the
        # direct physical mapping. Edge extension matches align_corners=False.
        offsets = np.clip((np.arange(size, dtype=np.float64) + 0.5) * out_size / size - 0.5, 0, out_size - 1)
        if self.fast_sampling:
            fbp_valid &= _bilinear_support_grid(native_valid, offsets, offsets)
        else:
            support_x, support_y = np.meshgrid(offsets, offsets)
            fbp_valid &= _bilinear_support(native_valid, support_x, support_y)
        if out_size >= size and out_size % size == 0:
            # A PAN feature reduced from 4x must not hide an invalid subpixel
            # simply because the central two samples happen to be valid.
            factor = out_size // size
            fbp_valid &= native_valid.reshape(size, factor, size, factor).all(axis=(1, 3))
        aux = values[:, :len(nx)].reshape(channels, out_size, out_size)
        if modality == "sar":
            aux = (np.clip(aux, -30.0, 10.0) + 10.0) / 20.0
        else:
            aux = aux / 65535.0
        aux[:, ~native_valid] = 0
        result["aux"] = np.ascontiguousarray(aux, dtype=np.float32)
        result["native_valid"][0] = native_valid
        result["valid"][0] = fbp_valid
        metrics = self.registration["registration"].get("metrics", {})
        result["quality"].update({
            "registration_status": "PASS", "registration_npz_sha256": self.registration_sha256,
            "registration_p95_new_ms_px": metrics.get("grid_holdout_forward_p95_new_ms_px"),
            "gradient_ncc": metrics.get("gradient_ncc"),
            "time_gap_days": self.metadata.get("sentinel1", {}).get("time_gap_days"),
            "native_valid_fraction": float(native_valid.mean()), "valid_fraction": float(fbp_valid.mean()),
            "read_window": bbox,
        })
        return result

    def close(self) -> None:
        for image, mask in self._datasets.values():
            image.close()
            mask.close()
        self._datasets.clear()
        self._closed = True

    def __enter__(self) -> "AuxiliaryScene":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def audit_assets(root: str | Path) -> dict[str, Any]:
    """Audit declared coverage, pinned registration and headers, never labels/pixels."""
    auxiliary = _aux_root(root)
    rows = _availability(auxiliary)
    result: dict[str, Any] = {
        "contract": CONTRACT, "auxiliary_root": str(auxiliary), "scene_count": len(rows),
        "sar_count": sum(_flag(r["has_sar"]) for r in rows),
        "pan_count": sum(_flag(r["has_pan"]) for r in rows),
        "registered_count": 0, "label_usage": False, "raster_pixels_read": False,
        "errors": [], "scenes": [], "splits": {},
    }
    for row in rows:
        summary: dict[str, Any] = {"scene_id": row["scene_stem"],
                                  "has_sar": _flag(row["has_sar"]), "has_pan": _flag(row["has_pan"])}
        try:
            with AuxiliaryScene(auxiliary, row["scene_stem"]) as scene:
                if any(scene.available.values()):
                    result["registered_count"] += 1
                    summary["registration_npz_sha256"] = scene.registration_sha256
                for modality, present in scene.available.items():
                    if present:
                        image, _ = scene._open(modality)
                        summary[modality] = {"path": str(scene._paths[modality][0]),
                                             "shape": [image.count, image.height, image.width],
                                             "resolution_m": list(image.res)}
                summary["ok"] = True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summary["ok"] = False
            result["errors"].append({"scene_id": row["scene_stem"], "error": str(exc)})
        result["scenes"].append(summary)
    by_id = {r["scene_stem"]: r for r in rows}
    for split in ("train", "test"):
        path = auxiliary.parent / "splits" / f"{split}.txt"
        if path.is_file():
            ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            result["splits"][split] = {
                "scene_count": len(ids), "missing_availability": [name for name in ids if name not in by_id],
                "sar_count": sum(_flag(by_id[name]["has_sar"]) for name in ids if name in by_id),
                "pan_count": sum(_flag(by_id[name]["has_pan"]) for name in ids if name in by_id),
            }
    result["ok"] = not result["errors"] and all(not x["missing_availability"] for x in result["splits"].values())
    return result
