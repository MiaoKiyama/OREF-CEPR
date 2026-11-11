"""Authenticated OOF train120 inputs; no target-scene training bank."""
import gc
import time
from pathlib import Path
import numpy as np
from native24_original82_data import Bank, canonical, read, sha, write
from training_data_common import miou_percent


class TrainBank120:
    """Ordered train120 view; no official feature or target bank is created."""

    def __init__(self, c):
        self.c = c
        self.source = Path(c["source"])
        self.source_config = read(self.source / "config.json")
        self.train_ids = list(self.source_config["train_ids"])
        assert len(self.train_ids) == 120 and len(set(self.train_ids)) == 120
        assert not set(self.train_ids) & set(self.source_config["official_ids"])
        self.ids = list(self.train_ids)
        self.targets = True
        self.domain = "train120"
        self._bank = Bank(self.source_config, self.ids, targets=True)
        folds = {
            sid: fold for fold in range(5)
            for sid in self.source_config["splits"][str(fold)]["validation"]
        }
        assert set(folds) == set(self.ids)
        assert all(sum(v == f for v in folds.values()) == 24 for f in range(5))
        self.rows = [
            dict(r, domain="train120", original_fold=folds[r["scene"]],
                 fold=folds[r["scene"]], source_row=dict(r))
            for r in self._bank.rows
        ]
        self.n, self.pixels = self._bank.n, self._bank.pixels
        self.ends = np.asarray([r["local_end"] for r in self.rows], np.int64)
        self.store_offsets = np.asarray(
            [r["start"] - r["local_start"] for r in self.rows], np.int64
        )
        self.pool_rows = {"train120": self.rows}
        self.pool_slices = {"train120": slice(0, self.n)}
        self.scene_statistics = []
        self.pool_statistics = {}

    def scene(self, row, targets=True):
        return self._bank.scene(row["source_row"], targets=targets)

    def raw(self, indices):
        ix = np.asarray(indices, np.int64)
        if ix.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if len(ix) and (ix.min() < 0 or ix.max() >= self.n):
            raise IndexError("train120 bank index out of bounds")
        owner = np.searchsorted(self.ends, ix, side="right")
        source_indices = ix + self.store_offsets[owner]
        return canonical(
            self._bank.numeric[source_indices], self._bank.labels[source_indices]
        )

    def target_pack(self, progress=None):
        counts = np.empty((self.n, 25), np.uint16)
        labels = np.empty((self.n, 7), np.uint8)
        mass = np.empty(self.n, np.uint32)
        valid_mass = np.empty(self.n, np.float32)
        fixed = np.zeros((25, 25), np.int64)
        fixed_obs = np.zeros(25, np.int64)
        baselines = np.zeros((7, 25, 25), np.int64)
        scene_baselines, scene_fixed, statistics = [], [], []
        for number, row in enumerate(self.rows):
            d = self.scene(row, True)
            sl = slice(row["local_start"], row["local_end"])
            counts[sl], labels[sl], mass[sl] = d["counts"], d["labels"], d["mass"]
            valid = d["counts"][:, 1:].sum(1, dtype=np.int64)
            valid_mass[sl] = valid
            fixed += d["fixed"]
            fixed_obs += d["fixed_prediction_counts"]
            baselines += d["baselines"]
            scene_baselines.append(d["baselines"].copy())
            scene_fixed.append(d["fixed"].copy())
            group_truth = d["counts"].sum(0, dtype=np.int64)
            truth = group_truth + d["fixed"].sum(1, dtype=np.int64)
            for baseline in d["baselines"]:
                np.testing.assert_array_equal(baseline.sum(1), truth)
            group_cm = d["baselines"][2] - d["fixed"]
            assert np.all(group_cm >= 0)
            np.testing.assert_array_equal(group_cm.sum(1), group_truth)
            assert int(d["mass"].sum(dtype=np.uint64)) + int(
                d["fixed_prediction_counts"].sum(dtype=np.int64)
            ) == row["pixels"]
            statistics.append(dict(
                scene=row["scene"], domain="train120",
                original_fold=row["original_fold"],
                groups=row["local_end"] - row["local_start"],
                valid_groups=int(np.count_nonzero(valid)),
                group_observed_pixels=int(d["mass"].sum(dtype=np.uint64)),
                valid_group_pixels=int(valid.sum(dtype=np.int64)),
                pixels=int(row["pixels"]), valid_gt_pixels=int(truth[1:].sum()),
                gt_pixels_by_class=truth.tolist(),
                group_gt_pixels_by_class=group_truth.tolist(),
            ))
            if progress is not None:
                progress(number + 1, len(self.rows), row)
            del d, valid
        self.valid_mass = valid_mass
        self.scene_statistics = statistics
        self.scene_baselines = np.stack(scene_baselines)
        self.scene_fixed = np.stack(scene_fixed)
        self.scene_baseline_group_cm = self.scene_baselines[:, 2] - self.scene_fixed
        self.pool_statistics = {}
        for name in ("train120", "all120") + tuple(
            "original_fold%d" % f for f in range(5)
        ):
            selected = (
                np.arange(len(self.rows))
                if name in ("train120", "all120")
                else np.asarray([i for i, r in enumerate(self.rows)
                                 if r["original_fold"] == int(name[-1])])
            )
            ss = [statistics[i] for i in selected]
            base = self.scene_baselines[selected, 2].sum(0)
            self.pool_statistics[name] = dict(
                scenes=len(selected),
                **{key: sum(s[key] for s in ss) for key in (
                    "groups", "valid_groups", "group_observed_pixels",
                    "valid_group_pixels", "pixels", "valid_gt_pixels"
                )},
                gt_pixels_by_class=np.sum(
                    [s["gt_pixels_by_class"] for s in ss], axis=0, dtype=np.int64
                ).tolist(),
                baseline_miou_percent=miou_percent(base),
            )
        p = self.source / "arms" / "crossfit" / "fivefold_completed.npz"
        with np.load(p) as z:
            expected = z["baseline"].sum(0)
        np.testing.assert_array_equal(baselines[2], expected)
        self.baseline_verification = {"train120": dict(
            exact_confusion_match=True, comparator_path=str(p),
            comparator_sha256=sha(p), miou_percent=miou_percent(expected),
        )}
        return dict(counts=counts, labels=labels, mass=mass, fixed=fixed,
                    fixed_obs=fixed_obs, baselines=baselines, pixels=self.pixels)


def inventory(c, chunk_size=200000):
    """Read only train120 counts in chunks to determine exposure before launch."""
    bank = TrainBank120(c)
    scenes = []
    for row in bank.rows:
        valid_groups = 0
        valid_pixels = 0
        group_truth = np.zeros(25, np.int64)
        for lo in range(row["start"], row["end"], chunk_size):
            ct = bank._bank.counts[lo:min(lo + chunk_size, row["end"])]
            vm = ct[:, 1:].sum(1, dtype=np.int64)
            valid_groups += int(np.count_nonzero(vm))
            valid_pixels += int(vm.sum(dtype=np.int64))
            group_truth += ct.sum(0, dtype=np.int64)
        scenes.append(dict(
            scene=row["scene"], domain="train120",
            original_fold=row["original_fold"],
            groups=row["local_end"] - row["local_start"],
            valid_groups=valid_groups, valid_group_pixels=valid_pixels,
            pixels=row["pixels"], group_gt_pixels_by_class=group_truth.tolist(),
        ))
    pools = {}
    for name in ("train120", "all120") + tuple(
        "original_fold%d" % f for f in range(5)
    ):
        rows = scenes if name in ("train120", "all120") else [
            r for r in scenes if r["original_fold"] == int(name[-1])
        ]
        pools[name] = dict(
            scenes=len(rows),
            **{key: sum(r[key] for r in rows) for key in (
                "groups", "valid_groups", "valid_group_pixels", "pixels"
            )},
            group_gt_pixels_by_class=np.sum(
                [r["group_gt_pixels_by_class"] for r in rows], 0, dtype=np.int64
            ).tolist(),
        )
    return dict(
        pools=pools, scenes=scenes, official_ground_truth_read=False,
        official_feature_bank_constructed=False,
        ground_truth_scene_ids=bank.ids, time_unix=time.time(),
    )
