"""CPU-only independent replay of the complete joint-MS75 data contract.

Builds all prespecified deterministic epoch plans, then reads actual crops from the
first plan. Known optical labels must survive missing SAR and SAR masks.
No GPU model, held raster, optical feature cache or training output is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcmmsline3.ms75_sar_joint_data import MS75SARJointDataset, digest
from tcmmsline3.ms75_rebuild_data import read_cached_scene
from tools.evaluate_target_dino import _write_json

SCHEMA = "convnext_sar_independent_input_replay_v1_20260930"


def choose_probe_indices(dataset, samples):
    """Cover every train scene first, then fill a deterministic spread of rows."""
    if type(samples) is not int or samples < len(dataset.scene_ids) or samples > len(dataset):
        raise ValueError("Replay must include at least one crop per training scene")
    selected, seen_scenes = set(), set()
    for block, (scene, _, _) in enumerate(dataset.blocks):
        if scene not in seen_scenes:
            selected.add(block * 16)
            seen_scenes.add(scene)
    # Fill by evenly spaced ranks of a fixed shuffled full plan, not labels or
    # SAR quality. Prefix coverage above intentionally overrepresents rare IDs.
    rng = np.random.default_rng(20260930)
    remaining = np.array([i for i in range(len(dataset)) if i not in selected], np.int64)
    rng.shuffle(remaining)
    selected.update(int(i) for i in remaining[:samples - len(selected)])
    return sorted(selected)


def _bucket():
    return dict(crops=0, known_pixels=0, total_pixels=0, sar_valid_native_pixels=0,
                sar_supported_optical_pixels=0, label_histogram=[0] * 24,
                all_ignore_crops=0, redraws=0, fallbacks=0)


def _accumulate(bucket, sample):
    labels = sample["label"].numpy()
    known = labels < 24
    histogram = np.bincount(labels[known], minlength=24)
    bucket["crops"] += 1
    bucket["known_pixels"] += int(known.sum())
    bucket["total_pixels"] += labels.size
    bucket["sar_valid_native_pixels"] += int(sample["sar_valid"].sum())
    bucket["sar_supported_optical_pixels"] += int(sample["optical_valid"].sum())
    bucket["label_histogram"] = (np.asarray(bucket["label_histogram"]) + histogram).tolist()
    bucket["all_ignore_crops"] += int(not known.any())
    bucket["redraws"] += sample["redraw_count"]
    bucket["fallbacks"] += int(sample["fallback_used"])


def _finish(bucket):
    return {**bucket, "known_fraction": bucket["known_pixels"] / bucket["total_pixels"] if bucket["total_pixels"] else None,
            "observed_class_ids": [c for c, n in enumerate(bucket["label_histogram"]) if n]}


def audit(config, *, samples=256, progress=None):
    from tools.train_convnext_sar import validate_config, verify_files
    started = time.time()
    identity = validate_config(config, devices=False)
    verify_files(config)
    plans, scene_total, target_total, kind_total = [], Counter(), Counter(), Counter()
    data = None
    try:
        for epoch in range(config["epochs"]):
            current = MS75SARJointDataset(config, epoch=epoch)
            try:
                summary = current.audit()
                if len(current) != config["updates_per_epoch"] * 64:
                    raise AssertionError("Epoch sample budget changed")
                if set(summary["scene_block_counts"]) != set(current.scene_ids):
                    raise AssertionError("Epoch lost training scene coverage")
                if set(summary["scene_block_counts"]) & set(current.metadata["held_ids"]):
                    raise AssertionError("Held scene entered a training plan")
                counts = Counter(kind for _,kind,_ in current.blocks)
                expected = dict(scene=960, rare=576, hard=192, pond=96, boundary=96)
                if counts != expected:
                    raise AssertionError("Wrong epoch50/30/10/5/5 block counts")
                for first in range(0, len(current.blocks), 4):
                    if len({s for s, _, _ in current.blocks[first:first + 4]}) != 4:
                        raise AssertionError("Effective batch lacks four distinct scenes")
                targets = Counter(str(c) for _,kind,c in current.blocks if c >= 0)
                summary["target_class_crop_counts"] = {str(c): 16 * targets[str(c)] for c in range(24)}
                summary["scene_crop_counts"] = {s: 16 * n for s,n in summary["scene_block_counts"].items()}
                summary["kind_crop_counts"] = {k: 16 * n for k,n in counts.items()}
                summary["global_crop_indices"] = dict(first=0, last=len(current) - 1, unique=True,
                    rank0_count=len(current)//2, rank1_count=len(current)//2,
                    partition="each64 row interval -> first32 rank0, last32 rank1; no padding")
                scene_total.update(summary["scene_crop_counts"])
                target_total.update(summary["target_class_crop_counts"])
                kind_total.update(summary["kind_crop_counts"])
                plans.append(summary)
                if progress:
                    progress(dict(event="epoch_plan_checked", epoch=epoch, plan_sha256=summary["plan_sha256"]))
            finally:
                current.close()
        data = MS75SARJointDataset(config, epoch=0)
        indices = choose_probe_indices(data, samples)
        overall, by_kind, by_availability = _bucket(), {}, {"sar_available": _bucket(), "sar_missing": _bucket()}
        observed_scenes, rows = set(), []
        # Use a separate one-scene label cache to independently reconstruct
        # each target. It never observes the candidate's returned label array.
        label_scene, original_labels = None, None
        for number, index in enumerate(indices):
            sample = data[index]
            scene = sample["scene_id"]
            if scene not in data.scene_ids or scene in data.metadata["held_ids"]:
                raise AssertionError("Observed a non-training crop")
            observed_scenes.add(scene)
            if label_scene != scene:
                _, original_labels = read_cached_scene(data.metadata, scene)
                label_scene = scene
            top, left, n = sample["top"], sample["left"], data.n
            original = np.asarray(original_labels[top:top+n, left:left+n], dtype=np.int64)
            if sample["flip_code"] & 1:
                original = original[:, ::-1]
            if sample["flip_code"] & 2:
                original = original[::-1]
            if not np.array_equal(original, sample["label"].numpy()):
                raise AssertionError("Principal GT was changed, filtered or misaligned")
            if sample["known_pixels"] != int((original < 24).sum()) or sample["known_pixels"] < 1:
                raise AssertionError("Known-label count invalid or all-ignore crop")
            if not sample["has_sar"] and any(bool(sample[k].any()) for k in ("sar_db", "sar_valid", "optical_valid")):
                raise AssertionError("Missing SAR is not represented by zero tensors/masks")
            if not np.isfinite(sample["sar_db"].numpy()).all():
                raise AssertionError("Nonfinite SAR input")
            if sample["index"] != index:
                raise AssertionError("Epoch-local crop index changed")
            kind = sample["sampling_kind"]
            by_kind.setdefault(kind, _bucket())
            for bucket in (overall, by_kind[kind], by_availability["sar_available" if sample["has_sar"] else "sar_missing"]):
                _accumulate(bucket, sample)
            rows.append(dict(index=index, scene_id=scene, top=top, left=left, kind=kind,
                sampling_class=sample["sampling_class"], has_sar=sample["has_sar"],
                known_pixels=sample["known_pixels"], known_fraction=sample["known_pixels"] / (n*n),
                sar_optical_valid_pixels=int(sample["optical_valid"].sum()),
                redraw_count=sample["redraw_count"], fallback_used=sample["fallback_used"],
                label_preserved=True, flip_code=sample["flip_code"]))
            if progress and (number + 1) % 16 == 0:
                progress(dict(event="crop_replay_progress", completed=number + 1, total=samples))
        if observed_scenes != set(data.scene_ids) or not all(bucket["crops"] > 0 for bucket in by_availability.values()):
            raise AssertionError("Replay missed training scenes or SAR-availability groups")
        result = dict(schema=SCHEMA, status="passed", config_sha256=identity,
            duration_seconds=time.time() - started, gpu_model_constructed=False,
            held_rasters_or_labels_read=False, epoch_count=len(plans), epoch_plans=plans,
            all_epoch_scene_crop_counts=dict(scene_total), all_epoch_target_class_crop_counts=dict(target_total),
            all_epoch_kind_crop_counts=dict(kind_total), all_epoch_crops=sum(scene_total.values()),
            replay=dict(samples=samples, sampling="one row per scene, then fixed-seed random fill; diagnostic not training distribution",
                indices_sha256=digest(indices), observed_scene_ids=sorted(observed_scenes),
                overall=_finish(overall), by_kind={k:_finish(v) for k,v in by_kind.items()},
                by_sar_availability={k:_finish(v) for k,v in by_availability.items()}, rows=rows),
            assertions=dict(all_96_scenes_each_epoch=True, no_held_sampling=True,
                exact_fractions_each_epoch=True, four_distinct_scenes_each_update=True,
                all_ignore_crops_zero=True, principal_GT_bitwise_preserved=True,
                missing_SAR_keeps_optical_supervision=True),
            interpretation="A finite replay validates input mechanics and reports class coverage; zero counts in this replay do not mean a class is absent from training.")
        result["payload_sha256"] = digest(result)
        return result
    finally:
        if data is not None:
            data.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=256, choices=(128, 256))
    parser.add_argument("--output", help="Defaults to training output/input_audit.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = audit(config, samples=args.samples,
        progress=lambda row: print(json.dumps(row, sort_keys=True), flush=True))
    output = Path(args.output) if args.output else Path(config["output_dir"]) / "input_audit.json"
    _write_json(output, result)
    print(json.dumps(dict(status=result["status"], output=str(output),
        payload_sha256=result["payload_sha256"], samples=args.samples,
        overall=result["replay"]["overall"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
