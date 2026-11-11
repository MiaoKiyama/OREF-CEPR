"""Lossless replay120 bank and its original cache preparation."""
import argparse
import gc
import shutil
import time
from pathlib import Path
import numpy as np
from native24_original82_data import ORDER, canonical, read, sha, write
from training_data_common import miou_percent
SCHEMA = "native24_replay120_qc_lossless119_v1"


def numeric_from_groups(groups, q_table, has_sar=False, include_s=False):
    """Convert replay compact fields without altering probability precision."""
    labels = np.asarray(groups["labels"])
    n = len(labels)
    confidence = np.asarray(groups["confidence"], np.float32).reshape(n, 3, 7)
    confidence = confidence[:, :, ORDER].reshape(n, 21)
    probability = np.asarray(groups["probability"], np.float32).reshape(n, 49)
    known = np.asarray(groups["known_fraction"], np.float32).reshape(n, 49)
    roi = ((np.asarray(groups["keys"], np.int64) // 25**7) % 33).astype(np.int64)
    q_table = np.asarray(q_table, np.float32)
    if q_table.shape != (33, 18) or np.any(roi >= 32):
        raise ValueError("Replay Q requires the existing 33x18 table and 32 image ROIs")
    q = q_table[roi].copy()
    q[:, :9] *= q[:, 9:]
    numeric = np.zeros((n, 144), np.float32)
    numeric[:, :21] = confidence
    numeric[:, 21:70] = probability
    numeric[:, 70:119] = known
    if include_s:
        numeric[:, 119:125] = groups["sar"]
        numeric[:, 125] = float(has_sar)
    numeric[:, 126:144] = q
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite replay evidence")
    return numeric


def canonical_replay(groups, q_table, has_sar=False):
    return canonical(numeric_from_groups(groups, q_table, has_sar), groups["labels"])


def allowed_training_target_paths(c):
    index = read(Path(c["replay_store"]) / "index.json")
    return {str(Path(p).resolve()) for p in index["target_files"]}


def _pool_statistics(scenes, scene_baselines):
    pools = {}
    for name in ("replay120", "all120") + tuple("original_fold%d" % f for f in range(5)):
        indices = list(range(len(scenes))) if name in ("replay120", "all120") else [
            i for i, r in enumerate(scenes) if r["original_fold"] == int(name[-1])
        ]
        selected = [scenes[i] for i in indices]
        pools[name] = dict(
            scenes=len(indices),
            **{k: sum(r[k] for r in selected) for k in (
                "groups", "valid_groups", "group_observed_pixels",
                "valid_group_pixels", "pixels", "valid_gt_pixels"
            )},
            gt_pixels_by_class=np.sum(
                [r["gt_pixels_by_class"] for r in selected], axis=0, dtype=np.int64
            ).tolist(),
            baseline_miou_percent=miou_percent(scene_baselines[indices, 2].sum(0)),
        )
    return pools


class TrainBank120:
    """Prepared replay inputs only; no fallback to OOF counts or features."""

    def __init__(self, c):
        self.c = c
        self.source = Path(c["source"])
        self.source_config = read(self.source / "config.json")
        self.store = Path(c["replay_store"])
        self.index = read(self.store / "index.json")
        self.receipt = read(self.store / "completed.json")
        if self.receipt.get("status") != "completed":
            raise RuntimeError("Replay preparation has not completed")
        self.ids = list(self.source_config["train_ids"])
        self.train_ids = list(self.ids)
        assert len(self.ids) == 120 and len(set(self.ids)) == 120
        assert not set(self.ids) & set(self.source_config["official_ids"])
        mapping = {r["scene"]: r for r in self.index["rows"]}
        assert set(mapping) == set(self.ids)
        folds = {
            sid: f for f in range(5)
            for sid in self.source_config["splits"][str(f)]["validation"]
        }
        assert set(folds) == set(self.ids)
        self.rows = []
        self.arrays = []
        cursor = 0
        for sid in self.ids:
            row = dict(mapping[sid])
            assert int(row["original_fold"]) == folds[sid]
            n = int(row["end"]) - int(row["start"])
            row.update(domain="replay120", fold=folds[sid],
                       local_start=cursor, local_end=cursor + n)
            meta = read(row["files"]["meta"])
            row["pixels"] = int(meta["pixels"])
            a = {k: np.load(row["files"][k], mmap_mode="r") for k in
                 ("numeric119", "labels", "counts", "mass", "roi", "q_table")}
            assert a["numeric119"].shape == (n, 119) and a["numeric119"].dtype == np.float32
            assert a["labels"].shape == (n, 7) and a["labels"].dtype == np.uint8
            assert a["counts"].shape == (n, 25) and a["counts"].dtype == np.uint16
            assert a["mass"].shape == (n,) and a["mass"].dtype == np.uint32
            assert a["roi"].shape == (n,) and a["roi"].dtype == np.uint8
            assert a["q_table"].shape == (33, 18)
            a["meta"] = meta
            self.rows.append(row)
            self.arrays.append(a)
            cursor += n
        self.n = cursor
        self.pixels = sum(r["pixels"] for r in self.rows)
        self.ends = np.asarray([r["local_end"] for r in self.rows], np.int64)
        self.starts = np.asarray([r["local_start"] for r in self.rows], np.int64)
        self.row_number = {r["scene"]: i for i, r in enumerate(self.rows)}
        self.targets, self.domain = True, "replay120"
        self.pool_rows = {"replay120": self.rows}
        self.pool_slices = {"replay120": slice(0, self.n)}

    @staticmethod
    def _numeric(a, ix=slice(None)):
        compact = a["numeric119"][ix]
        numeric = np.zeros((len(compact), 144), np.float32)
        numeric[:, :119] = compact
        q = a["q_table"][a["roi"][ix]].copy()
        q[:, :9] *= q[:, 9:]
        numeric[:, 126:144] = q
        return numeric

    def scene(self, row, targets=True):
        a = self.arrays[self.row_number[row["scene"]]]
        meta = a["meta"]
        d = dict(numeric=self._numeric(a), labels=np.asarray(a["labels"]),
                 mass=np.asarray(a["mass"]),
                 fixed_prediction_counts=np.asarray(meta["fixed_obs"], np.int64))
        if targets:
            d.update(counts=np.asarray(a["counts"]),
                     fixed=np.asarray(meta["fixed"], np.int64),
                     baselines=np.asarray(meta["baselines"], np.int64))
        return d

    def raw(self, indices):
        ix = np.asarray(indices, np.int64)
        if ix.ndim != 1 or (len(ix) and (ix.min() < 0 or ix.max() >= self.n)):
            raise IndexError("Replay bank index out of bounds")
        owners = np.searchsorted(self.ends, ix, side="right")
        numeric, labels = np.empty((len(ix), 144), np.float32), np.empty((len(ix), 7), np.uint8)
        for i in np.unique(owners):
            take = np.flatnonzero(owners == i)
            local = ix[take] - self.starts[i]
            numeric[take] = self._numeric(self.arrays[i], local)
            labels[take] = self.arrays[i]["labels"][local]
        return canonical(numeric, labels)

    def target_pack(self, progress=None):
        counts = np.empty((self.n, 25), np.uint16)
        labels = np.empty((self.n, 7), np.uint8)
        mass = np.empty(self.n, np.uint32)
        self.valid_mass = np.empty(self.n, np.float32)
        fixed, fixed_obs = np.zeros((25, 25), np.int64), np.zeros(25, np.int64)
        baselines = np.zeros((7, 25, 25), np.int64)
        statistics, scene_baselines, scene_fixed = [], [], []
        for number, (row, a) in enumerate(zip(self.rows, self.arrays)):
            meta = a["meta"]
            sl = slice(row["local_start"], row["local_end"])
            counts[sl], labels[sl], mass[sl] = a["counts"], a["labels"], a["mass"]
            np.testing.assert_array_equal(a["counts"].sum(1, dtype=np.int64), a["mass"])
            valid = a["counts"][:, 1:].sum(1, dtype=np.int64)
            self.valid_mass[sl] = valid
            sf = np.asarray(meta["fixed"], np.int64)
            sb = np.asarray(meta["baselines"], np.int64)
            fo = np.asarray(meta["fixed_obs"], np.int64)
            fixed += sf
            fixed_obs += fo
            baselines += sb
            scene_baselines.append(sb)
            scene_fixed.append(sf)
            group_truth = a["counts"].sum(0, dtype=np.int64)
            truth = group_truth + sf.sum(1)
            np.testing.assert_array_equal(sf.sum(0), fo)
            for b in sb:
                np.testing.assert_array_equal(b.sum(1), truth)
            assert int(a["mass"].sum(dtype=np.uint64)) + int(fo.sum()) == row["pixels"]
            statistics.append(dict(
                scene=row["scene"], domain="replay120", original_fold=row["original_fold"],
                groups=row["local_end"] - row["local_start"],
                valid_groups=int(np.count_nonzero(valid)),
                group_observed_pixels=int(a["mass"].sum(dtype=np.uint64)),
                valid_group_pixels=int(valid.sum()), pixels=row["pixels"],
                valid_gt_pixels=int(truth[1:].sum()),
                gt_pixels_by_class=truth.tolist(),
                group_gt_pixels_by_class=group_truth.tolist(),
            ))
            if progress is not None:
                progress(number + 1, len(self.rows), row)
        expected = np.asarray(self.receipt["baseline_confusions"], np.int64)
        np.testing.assert_array_equal(baselines, expected)
        self.scene_baselines, self.scene_fixed = np.stack(scene_baselines), np.stack(scene_fixed)
        self.scene_baseline_group_cm = self.scene_baselines[:, 2] - self.scene_fixed
        self.scene_statistics = statistics
        self.pool_statistics = _pool_statistics(statistics, self.scene_baselines)
        self.baseline_verification = {"replay120": dict(
            exact_confusion_match=True, comparator_path=str(self.store / "completed.json"),
            comparator_sha256=sha(self.store / "completed.json"),
            miou_percent=miou_percent(expected[2]),
        )}
        return dict(counts=counts, labels=labels, mass=mass, fixed=fixed,
                    fixed_obs=fixed_obs, baselines=baselines, pixels=self.pixels)


def inventory(c):
    bank = TrainBank120(c)
    bank.target_pack()
    return dict(pools=bank.pool_statistics, scenes=bank.scene_statistics,
                official_ground_truth_read=False, official_feature_bank_constructed=False,
                ground_truth_scene_ids=bank.ids, time_unix=time.time())


def _write_array(path, value):
    np.save(path, value, allow_pickle=False)
    actual = np.load(path, mmap_mode="r")
    np.testing.assert_array_equal(actual, value)
    return dict(sha256=sha(path), bytes=Path(path).stat().st_size,
                shape=list(actual.shape), dtype=str(actual.dtype))


def prepare(store, source, replay_root, storage_roots):
    """Build new sharded inputs; existing expert/Q/output folders stay read-only."""
    store, source, replay_root = Path(store), Path(source), Path(replay_root)
    store.mkdir(parents=True, exist_ok=True)
    if (store / "completed.json").exists():
        receipt = read(store / "completed.json")
        assert receipt["status"] == "completed"
        return receipt
    started = time.time()
    c = read(source / "config.json")
    ids = list(c["train_ids"])
    assert len(ids) == 120 and not set(ids) & set(c["official_ids"])
    replay_receipt = read(replay_root / "completed.json")
    assert replay_receipt["status"] == "completed"
    assert set(replay_receipt["destinations"]) == set(ids)
    folds = {sid: f for f in range(5) for sid in c["splits"][str(f)]["validation"]}
    source_index = read(Path(c["store"]) / "index.json")
    source_rows = {r["scene"]: r for r in source_index["rows"]}
    source_inventory = {r["scene"]: r for r in c["scene_inventory"]["oof"]}
    source_numeric = np.load(Path(c["store"]) / "numeric.npy", mmap_mode="r")
    roots = [Path(p) for p in storage_roots]
    for root in roots:
        if not root.is_dir():
            raise ValueError("Storage root does not exist: " + str(root))
    prefix = store.parent.name + "_store"
    rows, target_files = [], []
    totals = np.zeros((7, 25, 25), np.int64)
    total_bytes, cursor = 0, 0
    for number, sid in enumerate(ids):
        folder = Path(replay_receipt["destinations"][sid])
        manifest_path = folder / "compact_manifest.json"
        m = read(manifest_path)
        assert m["action_indices"] == ORDER.tolist()
        assert m["grouping_uses_gt"] is False and m["counts_are_training_targets_only"] is True
        groups_path = folder / m["groups"]["file"]
        assert sha(groups_path) == m["groups"]["sha256"]
        with np.load(groups_path) as z:
            g = {k: z[k] for k in ("keys", "labels", "confidence", "probability",
                                   "known_fraction", "counts", "mass")}
        n = len(g["labels"])
        assert n == m["group_count"]
        assert np.all(g["labels"] >= 1) and np.all(g["labels"] <= 24)
        assert np.all(np.diff(g["keys"]) > 0)
        assert int(g["counts"].max(initial=0)) <= 65535
        np.testing.assert_array_equal(g["counts"].sum(1, dtype=np.uint64), g["mass"])
        q_path = Path(c["oof_q"]) / "sidecars" / (sid + ".npz")
        with np.load(q_path) as z:
            q_table = z["q"].astype(np.float32)
        numeric = numeric_from_groups(g, q_table)
        np.testing.assert_allclose(g["probability"].diagonal(axis1=1, axis2=2),
                                   numeric[:, :7], atol=4e-6, rtol=0)
        np.testing.assert_allclose(g["known_fraction"].diagonal(axis1=1, axis2=2),
                                   1, atol=4e-6, rtol=0)
        sr = source_rows[sid]
        original_folder = Path(sr.get("source", source_inventory[sid]["source"]))
        with np.load(original_folder / "groups.npz") as z:
            original_roi = ((z["keys"] // 25**7) % 33).astype(np.int64)
        expected_q = q_table[original_roi].copy()
        expected_q[:, :9] *= expected_q[:, 9:]
        np.testing.assert_array_equal(
            source_numeric[sr["start"]:sr["end"], 126:144], expected_q)
        del expected_q, original_roi
        with np.load(folder / "confusions.npz") as z:
            baselines = z["confusions"][ORDER].astype(np.int64)
            fixed = z["fixed_agreement"].astype(np.int64)
        fixed_obs = np.zeros(25, np.int64)
        hard_pixels = 0
        for chunk in m["hard_prediction_chunks"]:
            with np.load(folder / "predictions" / chunk["file"]) as z:
                hard = z["labels"]
            active = np.any(hard != hard[6], axis=0)
            fixed_obs += np.bincount(hard[6, ~active], minlength=25)
            hard_pixels += hard.shape[1]
        pixels = int(np.prod(m["shape"]))
        assert hard_pixels == pixels
        np.testing.assert_array_equal(fixed.sum(0), fixed_obs)
        assert int(g["mass"].sum(dtype=np.uint64)) + int(fixed_obs.sum()) == pixels
        # Rebuild every baseline from target counts and candidate labels.
        for action in range(7):
            cm = fixed.copy()
            for truth in range(25):
                np.add.at(cm[truth], g["labels"][:, action], g["counts"][:, truth])
            np.testing.assert_array_equal(cm, baselines[action])
        roi = ((g["keys"] // 25**7) % 33).astype(np.uint8)
        arrays = dict(numeric119=numeric[:, :119].copy(), labels=g["labels"].astype(np.uint8),
                      counts=g["counts"].astype(np.uint16), mass=g["mass"].astype(np.uint32),
                      roi=roi, q_table=q_table)
        needed = sum(a.nbytes for a in arrays.values()) + 2**20
        available = {
            root: shutil.disk_usage(root).free - (2 if "hot0" in str(root) else 1) * 2**30
            for root in roots
        }
        chosen = max(roots, key=lambda r: available[r])
        if available[chosen] < needed:
            raise RuntimeError("Insufficient reserved storage for next replay scene")
        shard = chosen / prefix / "scenes" / sid
        shard.mkdir(parents=True, exist_ok=True)
        files, file_receipts = {}, {}
        for name, array in arrays.items():
            path = shard / (name + ".npy")
            files[name] = str(path.resolve())
            file_receipts[name] = _write_array(path, array)
        valid = g["counts"][:, 1:].sum(1, dtype=np.int64)
        group_truth = g["counts"].sum(0, dtype=np.int64)
        truth = group_truth + fixed.sum(1)
        meta = dict(scene=sid, pixels=pixels, shape=m["shape"],
                    fixed=fixed.tolist(), fixed_obs=fixed_obs.tolist(),
                    baselines=baselines.tolist(), valid_groups=int(np.count_nonzero(valid)),
                    valid_group_pixels=int(valid.sum()), group_observed_pixels=int(g["mass"].sum(dtype=np.uint64)),
                    valid_gt_pixels=int(truth[1:].sum()), gt_pixels_by_class=truth.tolist(),
                    group_gt_pixels_by_class=group_truth.tolist())
        files["meta"] = str((shard / "meta.json").resolve())
        write(files["meta"], meta)
        file_receipts["meta"] = dict(sha256=sha(files["meta"]),
                                     bytes=Path(files["meta"]).stat().st_size)
        row = dict(scene=sid, start=cursor, end=cursor + n, original_fold=folds[sid],
                   shard_dir=str(shard.resolve()), pixels=pixels, files=files,
                   file_receipts=file_receipts,
                   source_manifest_sha256=sha(manifest_path), source_groups_sha256=m["groups"]["sha256"],
                   q_table_source=str(q_path), q_table_sha256=sha(q_path),
                   q_exactly_matches_oof=True, baselines_exactly_reconstructed=True)
        rows.append(row)
        target_files.extend([files["counts"], files["meta"]])
        totals += baselines
        cursor += n
        total_bytes += sum(r["bytes"] for r in file_receipts.values())
        write(store / "preparation_progress.json", dict(
            status="running", completed_scenes=number + 1, total_scenes=len(ids),
            groups=cursor, bytes=total_bytes, elapsed_seconds=time.time() - started,
            scene=sid, time_unix=time.time()))
        print("prepared %d/120 groups=%d GiB=%.3f seconds=%.1f" %
              (number + 1, cursor, total_bytes / 2**30, time.time() - started), flush=True)
        del g, numeric, arrays, valid, hard
        gc.collect()
    index = dict(schema=SCHEMA, source=str(source), replay_root=str(replay_root),
                 train_ids=ids, groups=cursor, rows=rows, target_files=target_files,
                 official_ground_truth_read=False, S_enabled=False,
                 confidence_source_order="raw experts reordered to action ORDER",
                 probability_order="action ORDER model x action ORDER candidate",
                 probability_semantics="conditional mean and known fraction stored separately float32")
    write(store / "index.json", index)
    receipt = dict(status="completed", schema=SCHEMA, scenes=len(rows), groups=cursor,
                   bytes=total_bytes, baseline_confusions=totals.tolist(),
                   baseline_miou_percent=miou_percent(totals[2]),
                   source_config_sha256=sha(source / "config.json"),
                   replay_completed_sha256=sha(replay_root / "completed.json"),
                   index_sha256=sha(store / "index.json"), data_code_sha256=sha(__file__),
                   official_ground_truth_read=False, every_written_array_verified_exact=True,
                   all_q_exactly_matches_oof=True, all_baselines_exactly_reconstructed=True,
                   elapsed_seconds=time.time() - started, time_unix=time.time())
    write(store / "completed.json", receipt)
    print("completed", receipt["groups"], receipt["baseline_miou_percent"], flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--replay-root", required=True)
    p.add_argument("--storage-roots", nargs="+", required=True)
    args = p.parse_args()
    prepare(args.store, args.source, args.replay_root, args.storage_roots)

if __name__ == "__main__":
    main()
