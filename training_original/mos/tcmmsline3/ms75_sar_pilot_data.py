"""Shared HR-MS75/SAR pilot crops, with a label/SAR-only replay before fit.

No arm is selected here. Optical, real-SAR and zero-SAR controls consume the
same sealed rows, GT mask and augmentations. Only original train96 scenes with
SAR are eligible. Source groups are diagnostics, not sampling weights.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .light_sar_data import (
    _checked_aux, _load_points, _sha, apply_flips, load_prepared_manifest,
    open_sar_scene,
)
from .ms75_rebuild_data import (
    HARD_CLASS_IDS, RARE_CLASS_IDS, GlobalViews, _check_paths,
    load_manifest, read_cached_scene, read_scaled_views,
)
from .target_dino_data import SPECTRAL_AUGMENTATION
from .target_ms75_data import MODEL_INPUT_KEYS, _augment_views, _flip_views

SCHEMA = "ms75_sar_pilot_shared_sampling_v1_20260929"
WATER = (10, 12, 13, 14)
BLOCK_SIZE, EFFECTIVE_BATCH = 16, 64
KINDS = ("scene", "rare", "hard", "pond", "boundary")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)


def _new_output(output):
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("pilot output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_labels(metadata, scene):
    # Deliberately do not call read_cached_scene: replay never loads MS spectra.
    if scene not in metadata["selection_train_ids"] or scene in metadata["held_ids"]:
        raise ValueError("pilot labels must belong to train96")
    row = metadata["scenes"][scene]
    _check_paths(metadata, row)
    labels = np.load(row["label"], mmap_mode="r", allow_pickle=False)
    if labels.dtype != np.uint8 or labels.shape != tuple(row["shape"]):
        raise ValueError("pilot label geometry/dtype changed")
    return labels


def known_water_boundary(labels):
    """Distinct endpoints of known 4-neighbour water/nonwater pairs."""
    labels = np.asarray(labels)
    known = labels < 24
    water = np.isin(labels, WATER)
    edge = np.zeros(labels.shape, bool)
    h = known[:, :-1] & known[:, 1:] & (water[:, :-1] != water[:, 1:])
    v = known[:-1] & known[1:] & (water[:-1] != water[1:])
    edge[:, :-1] |= h
    edge[:, 1:] |= h
    edge[:-1] |= v
    edge[1:] |= v
    return edge


def boundary_reservoir(labels, *, capacity=4096, seed=1337, stripe_rows=256):
    """Uniform reservoir via independent priorities, bounded memory with halo.

    Each endpoint occurs once. Keep globally smallest IID priorities; no class,
    SAR, model output or downstream effect determines the selected coordinates.
    """
    if type(capacity) is not int or capacity < 1 or stripe_rows < 1:
        raise ValueError("positive reservoir capacity and stripe size required")
    rng = np.random.default_rng(seed)
    h, _ = labels.shape
    keys, points, total = np.empty(0), np.empty((0, 2), np.int64), 0
    for first in range(0, h, stripe_rows):
        last = min(first + stripe_rows, h)
        a, b = max(0, first - 1), min(h, last + 1)
        mask = known_water_boundary(labels[a:b])[first-a:last-a]
        yy, xx = np.nonzero(mask)
        new = np.column_stack((yy + first, xx))
        priorities = rng.random(len(new))
        total += len(new)
        keys = np.concatenate((keys, priorities))
        points = np.concatenate((points, new))
        if len(keys) > capacity:
            keep = np.argpartition(keys, capacity-1)[:capacity]
            keys, points = keys[keep], points[keep]
    if len(points):
        points = points[np.lexsort((points[:, 1], points[:, 0]))]
    return points.astype(np.int64, copy=False), int(total)


def _crop_boundary(labels, top, left, size):
    h, w = labels.shape
    a, b = max(0, top-1), min(h, top+size+1)
    c, d = max(0, left-1), min(w, left+size+1)
    # Halo preserves true endpoints at crop edges; no 255-as-land convention.
    return known_water_boundary(labels[a:b, c:d])[top-a:top-a+size, left-c:left-c+size]


def _boundary_fallback(labels, points, reader, size, sar_size, minimum, limit=64):
    h, w = labels.shape
    if not len(points):
        return None, 0
    selected = np.unique(np.linspace(0, len(points)-1, min(limit, len(points))).round().astype(int))
    tested = 0
    for index in selected:
        cy, cx = map(int, points[index])
        top, left = int(np.clip(cy-size//2, 0, h-size)), int(np.clip(cx-size//2, 0, w-size))
        target = labels[top:top+size, left:left+size]
        tested += 1
        if np.count_nonzero(target < 24) < minimum:
            continue
        _, _, optical = _checked_aux(reader.read_patch(top, left, size, modality="sar", out_size=sar_size), size, sar_size)
        if np.count_nonzero((target < 24) & optical[0]) >= minimum and optical[0, cy-top, cx-left]:
            return dict(top=top, left=left, anchor=[cy, cx], class_id=-1), tested
    return None, tested


def prepare_pilot_support(base_prepared, output, *, source_by_scene,
                          boundary_capacity=4096, workers=1, progress=None):
    """Reuse authenticated SAR moments/support; add train-only boundary anchors.

    Existing class reservoirs and fallback witnesses are borrowed read-only.
    No held labels, MS spectra, OOF predictions or former 28-scene pool are read.
    """
    base = load_prepared_manifest(base_prepared)
    raw = load_manifest(base["raw_manifest"])
    scenes = tuple(base["adapt_scene_ids"])
    if not set(scenes).issubset(source_by_scene):
        raise ValueError("source diagnostics must cover every SAR training scene")
    if raw.get("strict_protocol", True) and len(scenes) != 83:
        raise ValueError("formal pilot must retain all 83 SAR training scenes")
    out = _new_output(output)
    n, sn = base["crop_size"], base["sar_size"]
    minimum = math.ceil(n*n*base["min_supervised_fraction"])
    def one(scene):
        labels = _read_labels(raw, scene)
        row = raw["scenes"][scene]
        if hashlib.sha256(memoryview(labels)).hexdigest() != row["label_sha256"]:
            raise ValueError("original mapped GT payload changed")
        points = _load_points(base["scenes"][scene]["points"], base["scenes"][scene]["points_sha256"])
        for c, xy in points.items():
            if len(xy) and not np.all(labels[xy[:, 0], xy[:, 1]] == c):
                raise ValueError("borrowed class anchor no longer agrees with GT")
        scene_seed = int(hashlib.sha256(scene.encode()).hexdigest()[:16], 16)
        boundary, total = boundary_reservoir(labels, capacity=boundary_capacity, seed=scene_seed)
        path = out / f"{scene}.boundary.npz"
        np.savez_compressed(path, points=boundary)
        reader = open_sar_scene(base["data_root"], scene)
        try:
            if reader.registration_sha256 != base["scenes"][scene]["registration_sha256"]:
                raise ValueError("SAR registration binding changed")
            fallback, tested = _boundary_fallback(labels, boundary, reader, n, sn, minimum)
        finally:
            reader.close()
        value = dict(boundary_points=str(path), boundary_sha256=_sha(path),
                     boundary_population=total, boundary_reservoir_count=len(boundary),
                     boundary_fallback=fallback, boundary_witnesses_tested=tested,
                     boundary_unavailable_reason=None if fallback else "no_witness_in_fixed_candidate_set_not_global_infeasibility",
                     source=str(source_by_scene[scene]),
                     borrowed_excluded_target_classes=base["scenes"][scene]["excluded_target_classes"])
        if progress:
            progress(dict(event="pilot_boundary_prepared", scene=scene, **value))
        return scene, value
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = dict(pool.map(one, scenes))
    value = dict(schema=SCHEMA, base_prepared=str(Path(base_prepared).resolve()),
                 base_prepared_sha256=_sha(base_prepared), scene_ids=list(scenes), scenes=rows,
                 boundary_capacity=boundary_capacity, raw_manifest_sha256=base["raw_manifest_sha256"],
                 spectral_image_reads=False, held_label_reads=False, source_weighting=False,
                 supervision="known_GT_AND_actual_crop_optical_SAR_valid", water_class_ids=list(WATER),
                 class_scene_support={str(c): dict(natural=sum(raw["scenes"][s]["histogram"][c] > 0 for s in scenes),
                     viable_fixed_fallback=len(base["class_scenes"][str(c)])) for c in range(24)},
                 sampling_fractions=dict(scene=.5, rare=.3, hard=.1, pond=.05, boundary=.05))
    value["payload_sha256"] = digest(value)
    path = out / "prepared_manifest.json"
    _write_json(path, value)
    return path


def load_pilot_support(prepared):
    value = dict(prepared) if isinstance(prepared, dict) else json.loads(Path(prepared).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("payload_sha256") != digest({k: v for k, v in value.items() if k != "payload_sha256"}):
        raise ValueError("pilot support identity changed")
    if _sha(value["base_prepared"]) != value["base_prepared_sha256"]:
        raise ValueError("borrowed support manifest changed")
    base = load_prepared_manifest(value["base_prepared"])
    if value["scene_ids"] != base["adapt_scene_ids"] or set(value["scenes"]) != set(value["scene_ids"]):
        raise ValueError("pilot must retain exact original SAR train96 universe")
    return value, base


def make_pilot_plan(prepared, *, updates=415, epoch=0, seed=1337):
    """Exact 50/30/10/5/5; four distinct scenes per 64-crop update."""
    support, base = load_pilot_support(prepared)
    scenes = tuple(support["scene_ids"])
    if (type(updates) is not int or updates < 5 or updates % 5 or type(epoch) is not int or epoch < 0
            or type(seed) is not int or not 0 <= seed < 2**32 or len(scenes) < 4):
        raise ValueError("whole groups of five positive updates and a valid seed required")
    if updates * 2 < len(scenes):
        raise ValueError("natural block budget cannot cover all scenes")
    rng = random.Random(seed + epoch * 10000019)
    targeted = ["rare"]*(updates*6//5) + ["hard"]*(updates*2//5) + ["pond"]*(updates//5) + ["boundary"]*(updates//5)
    rng.shuffle(targeted)
    queue, blocks, natural_coverage = deque(), [], set()
    class_scenes = {int(c): ss for c, ss in base["class_scenes"].items() if ss}
    boundary_scenes = [s for s in scenes if support["scenes"][s]["boundary_fallback"] is not None]
    for u in range(updates):
        chosen, current = set(), []
        for _ in range(2):
            if not queue:
                fresh = list(scenes)
                rng.shuffle(fresh)
                queue.extend(fresh)
            for _ in range(len(queue)):
                scene = queue.popleft()
                if scene not in chosen:
                    break
                queue.append(scene)
            else:
                scene = rng.choice([s for s in scenes if s not in chosen])
            chosen.add(scene)
            natural_coverage.add(scene)
            current.append((scene, "scene", -1))
        for kind in targeted[u*2:u*2+2]:
            if kind == "boundary":
                eligible = {-1: [s for s in boundary_scenes if s not in chosen]}
            else:
                ids = RARE_CLASS_IDS if kind == "rare" else HARD_CLASS_IDS if kind == "hard" else (13,)
                eligible = {c: [s for s in class_scenes.get(c, ()) if s not in chosen] for c in ids}
            eligible = {c: ss for c, ss in eligible.items() if ss}
            if not eligible:
                raise ValueError(f"no distinct-scene {kind} block available; no silent budget fallback")
            c = rng.choice(sorted(eligible))
            scene = rng.choice(eligible[c])
            chosen.add(scene)
            current.append((scene, kind, c))
        rng.shuffle(current)
        blocks.extend([dict(scene=s, kind=k, class_id=c) for s, k, c in current])
    if natural_coverage != set(scenes):
        raise ValueError("natural scene cycle lost scene coverage")
    value = dict(schema=SCHEMA, prepared_payload_sha256=support["payload_sha256"], epoch=epoch, seed=seed,
                 updates=updates, samples=updates*EFFECTIVE_BATCH, block_size=BLOCK_SIZE,
                 effective_batch=EFFECTIVE_BATCH, blocks=blocks, source_weighting=False)
    value["plan_sha256"] = digest(value)
    return value


def _validate_plan(support, plan):
    if (plan.get("schema") != SCHEMA or plan.get("prepared_payload_sha256") != support["payload_sha256"]
            or plan.get("plan_sha256") != digest({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan["samples"] != plan["updates"]*64 or len(plan["blocks"]) != plan["updates"]*4):
        raise ValueError("pilot plan identity/budget changed")
    expected = {"scene": 2*plan["updates"], "rare": plan["updates"]*6//5,
                "hard": plan["updates"]*2//5, "pond": plan["updates"]//5, "boundary": plan["updates"]//5}
    if dict(Counter(b["kind"] for b in plan["blocks"])) != expected:
        raise ValueError("pilot plan sampling mixture changed")
    if {b["scene"] for b in plan["blocks"]} != set(support["scene_ids"]):
        raise ValueError("pilot plan escaped original SAR training scenes")
    for first in range(0, len(plan["blocks"]), 4):
        if len({b["scene"] for b in plan["blocks"][first:first+4]}) != 4:
            raise ValueError("pilot update must have four distinct scenes")


def _hist(target, mask=None):
    selected = target < 24
    if mask is not None:
        selected &= mask
    return np.bincount(target[selected], minlength=24).astype(np.int64).tolist()


def _resolve_row(support, base, plan, index, labels, points, boundary, reader):
    block = plan["blocks"][index // BLOCK_SIZE]
    scene, kind, c = block["scene"], block["kind"], block["class_id"]
    rng = random.Random(plan["seed"] + plan["epoch"]*10000019 + index*65537)
    n, sn = base["crop_size"], base["sar_size"]
    h, w = labels.shape
    minimum = math.ceil(n*n*base["min_supervised_fraction"])
    rejections = Counter()
    for attempt in range(base["max_attempts"] + 1):
        fallback = attempt == base["max_attempts"]
        anchor = None
        if fallback:
            witness = (support["scenes"][scene]["boundary_fallback"] if kind == "boundary" else
                       base["scenes"][scene]["fallbacks"]["scene" if kind == "scene" else f"class_{c}"])
            top, left = witness["top"], witness["left"]
            if kind != "scene":
                anchor = tuple(witness["anchor"])
        elif kind == "scene":
            top, left = rng.randrange(h-n+1), rng.randrange(w-n+1)
        else:
            candidates = boundary if kind == "boundary" else points[c]
            cy, cx = map(int, candidates[rng.randrange(len(candidates))])
            anchor = cy, cx
            top = int(np.clip(cy-n//2+rng.randint(-n//4, n//4), 0, h-n))
            left = int(np.clip(cx-n//2+rng.randint(-n//4, n//4), 0, w-n))
        target = np.asarray(labels[top:top+n, left:left+n])
        if np.count_nonzero(target < 24) < minimum:
            rejections["insufficient_known_GT"] += 1
            continue
        _, native, optical = _checked_aux(reader.read_patch(top, left, n, modality="sar", out_size=sn), n, sn)
        valid = (target < 24) & optical[0]
        if np.count_nonzero(valid) < minimum:
            rejections["insufficient_GT_SAR_intersection"] += 1
            continue
        boundary_mask = _crop_boundary(labels, top, left, n)
        if anchor is not None:
            cy, cx = anchor
            ay, ax = cy-top, cx-left
            if not (0 <= ay < n and 0 <= ax < n and valid[ay, ax]
                    and (boundary_mask[ay, ax] if kind == "boundary" else target[ay, ax] == c)):
                rejections["target_anchor_invalid"] += 1
                continue
        flip = int(rng.random() < .5) + 2*int(rng.random() < .5)
        return dict(index=index, update=index//64, scene=scene, source=support["scenes"][scene]["source"],
                    kind=kind, class_id=c, top=top, left=left, anchor=list(anchor) if anchor else [-1, -1],
                    flip_code=flip, augmentation_seed=plan["seed"] + plan["epoch"]*10000019 + index*65537 + 917000003,
                    redraw_count=attempt, fallback_used=fallback, rejections=dict(rejections),
                    known_hist=_hist(target), valid_hist=_hist(target, optical[0]),
                    boundary_hist=_hist(target, boundary_mask), valid_boundary_hist=_hist(target, boundary_mask & optical[0]),
                    optical_valid_count=int(optical.sum()), native_valid_count=int(native.sum()),
                    optical_valid_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(optical))).hexdigest(),
                    native_valid_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(native))).hexdigest(),
                    original_label_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(target))).hexdigest())
    raise RuntimeError(f"pilot crop has no valid fixed fallback: {scene} {kind} {index}; no skipped updates allowed")


def summarize_replay(rows, plan):
    if [r["index"] for r in rows] != list(range(plan["samples"])):
        raise ValueError("replay must contain every crop exactly once in plan order")
    scene_rows, source_rows, kinds = {}, {}, {}
    def new():
        return dict(crops=0, fallback_count=0, redraw_count=0, known_hist=np.zeros(24, np.int64),
                    valid_hist=np.zeros(24, np.int64), boundary_hist=np.zeros(24, np.int64),
                    valid_boundary_hist=np.zeros(24, np.int64), rejections=Counter(), coordinates=set(), updates=set())
    total = new()
    for row in rows:
        for bucket in (total, scene_rows.setdefault(row["scene"], new()), source_rows.setdefault(row["source"], new()),
                       kinds.setdefault(row["kind"], new())):
            bucket["crops"] += 1
            bucket["fallback_count"] += int(row["fallback_used"])
            bucket["redraw_count"] += row["redraw_count"]
            for field in ("known_hist", "valid_hist", "boundary_hist", "valid_boundary_hist"):
                bucket[field] += row[field]
            bucket["rejections"].update(row["rejections"])
            bucket["coordinates"].add((row["scene"], row["top"], row["left"]))
            bucket["updates"].add(row["update"])
    update_sources = []
    for i in range(0, len(rows), 64):
        block = rows[i:i+64]
        update_sources.append(len({r["source"] for r in block}))
        if len({r["scene"] for r in block}) != 4:
            raise ValueError("replayed update does not contain four distinct scenes")
    def finish(v):
        return {**{k: (x.tolist() if isinstance(x, np.ndarray) else dict(x) if isinstance(x, Counter) else x)
                    for k, x in v.items() if k not in ("coordinates", "updates")},
                "unique_coordinates": len(v["coordinates"]), "coordinate_repeat_fraction": 1-len(v["coordinates"])/v["crops"],
                "optimizer_updates_touched": len(v["updates"])}
    return dict(schema=SCHEMA, plan_sha256=plan["plan_sha256"], totals=finish(total),
                scenes={s: finish(v) for s, v in scene_rows.items()}, sources={s: finish(v) for s, v in source_rows.items()},
                kinds={s: finish(v) for s, v in kinds.items()},
                independent_source_counts_per_update=update_sources,
                independent_source_count_histogram=dict(Counter(map(str, update_sources))),
                class_scene_presence=[int(sum(v["valid_hist"][c] > 0 for v in scene_rows.values())) for c in range(24)],
                class_source_presence=[int(sum(v["valid_hist"][c] > 0 for v in source_rows.values())) for c in range(24)],
                spectral_image_reads=False, labels_are_continuous_pixel_GT=True,
                count_semantics="pixel exposures, repeated crops and spatial overlap are not deduplicated ground area")


def replay_plan(prepared, plan, output, *, workers=1, progress=None):
    """Materialize every accepted crop without reading MS or fitting a model."""
    support, base = load_pilot_support(prepared)
    _validate_plan(support, plan)
    raw = load_manifest(base["raw_manifest"])
    out = _new_output(output)
    by_scene = {s: [] for s in support["scene_ids"]}
    for index in range(plan["samples"]):
        by_scene[plan["blocks"][index//16]["scene"]].append(index)
    def one(scene):
        labels = _read_labels(raw, scene)
        record = base["scenes"][scene]
        points = _load_points(record["points"], record["points_sha256"])
        b = support["scenes"][scene]
        if _sha(b["boundary_points"]) != b["boundary_sha256"]:
            raise ValueError("boundary anchor binding changed")
        with np.load(b["boundary_points"], allow_pickle=False) as file:
            boundary = file["points"]
        reader = open_sar_scene(base["data_root"], scene)
        try:
            if reader.registration_sha256 != record["registration_sha256"]:
                raise ValueError("SAR registration changed during replay")
            values = [_resolve_row(support, base, plan, i, labels, points, boundary, reader) for i in by_scene[scene]]
        finally:
            reader.close()
        if progress:
            progress(dict(event="pilot_scene_replayed", scene=scene, crops=len(values)))
        return values
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [row for part in pool.map(one, support["scene_ids"]) for row in part]
    rows.sort(key=lambda r: r["index"])
    summary = summarize_replay(rows, plan)
    rows_path = out / "crops.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
    _write_json(out / "summary.json", summary)
    _write_json(out / "schedule.json", plan)
    result = dict(schema=SCHEMA, prepared=str(Path(prepared).resolve()), prepared_sha256=_sha(prepared),
                  crops=str(rows_path), crops_sha256=_sha(rows_path), summary_sha256=_sha(out/"summary.json"),
                  schedule_sha256=_sha(out/"schedule.json"), plan_sha256=plan["plan_sha256"], samples=len(rows),
                  crop_size=base["crop_size"], sar_size=base["sar_size"], effective_batch=64,
                  scene_ids=support["scene_ids"], epoch=plan["epoch"], seed=plan["seed"])
    result["payload_sha256"] = digest(result)
    destination = out / "replay_manifest.json"
    _write_json(destination, result)
    return destination


class MS75SARPilotDataset(Dataset):
    """Consume a sealed replay; there is deliberately no arm-specific sampler."""
    def __init__(self, replay_manifest, *, views_cache=None, start_sample=0, augmentation=True,
                 short_input_size=512, global_input_size=512):
        replay = json.loads(Path(replay_manifest).read_text(encoding="utf-8"))
        if replay.get("schema") != SCHEMA or replay.get("payload_sha256") != digest({k:v for k,v in replay.items() if k != "payload_sha256"}):
            raise ValueError("replay manifest identity changed")
        if _sha(replay["prepared"]) != replay["prepared_sha256"] or _sha(replay["crops"]) != replay["crops_sha256"]:
            raise ValueError("sealed shared crops/support changed")
        self.support, self.base = load_pilot_support(replay["prepared"])
        self.metadata = load_manifest(self.base["raw_manifest"])
        self.rows = [json.loads(line) for line in Path(replay["crops"]).read_text(encoding="utf-8").splitlines()]
        if (len(self.rows) != replay["samples"] or [r["index"] for r in self.rows] != list(range(len(self.rows)))
                or {r["scene"] for r in self.rows} != set(self.support["scene_ids"])):
            raise ValueError("sealed crop row coverage changed")
        if type(start_sample) is not int or not 0 <= start_sample <= len(self.rows) or start_sample % 64:
            raise ValueError("resume must start at a complete effective64 update")
        self.replay, self.start_sample, self.augmentation = replay, start_sample, bool(augmentation)
        self.n, self.sn = replay["crop_size"], replay["sar_size"]
        if self.metadata.get("strict_protocol", True) and (self.n, self.sn, short_input_size, global_input_size) != (512, 166, 512, 512):
            raise ValueError("formal pilot must use original HR512 three views and SAR166")
        self.short_input_size, self.global_input_size = short_input_size, global_input_size
        self.global_views = GlobalViews(self.metadata, views_cache, size=global_input_size)
        self._cache, self._aux = OrderedDict(), OrderedDict()

    def __len__(self):
        return len(self.rows) - self.start_sample

    def _reader(self, scene):
        reader = self._aux.pop(scene, None)
        if reader is None:
            reader = open_sar_scene(self.base["data_root"], scene)
            if reader.registration_sha256 != self.base["scenes"][scene]["registration_sha256"]:
                reader.close()
                raise ValueError("SAR registration changed after replay")
        self._aux[scene] = reader
        while len(self._aux) > 4:
            self._aux.popitem(last=False)[1].close()
        return reader

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        row = self.rows[self.start_sample+int(index)]
        scene, top, left = row["scene"], row["top"], row["left"]
        cached = self._cache.pop(scene, None)
        if cached is None:
            cached = read_cached_scene(self.metadata, scene)
        self._cache[scene] = cached
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        image, labels = cached
        sample, original = read_scaled_views(image, labels, top=top, left=left, source_size=self.n,
            output_size=self.n, short_input_size=self.short_input_size, global_input_size=self.global_input_size,
            global_raw=self.global_views.get(scene, image), normalization=self.metadata["normalization"])
        sar, native, optical = _checked_aux(self._reader(scene).read_patch(top, left, self.n, modality="sar", out_size=self.sn), self.n, self.sn)
        for key, array in (("native_valid", native), ("optical_valid", optical), ("original_label", original.astype(np.uint8))):
            if hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest() != row[key+"_sha256"]:
                raise ValueError("actual crop label/support differs from sealed replay: "+key)
        target = original.copy()
        target[~optical[0]] = 255
        code = row["flip_code"] if self.augmentation else 0
        sample = _flip_views(sample, bool(code & 1), bool(code & 2))
        if self.augmentation:
            sample = _augment_views(sample, rng=random.Random(row["augmentation_seed"]),
                augmentation=SPECTRAL_AUGMENTATION, normalization=self.metadata["normalization"])
        sample.update(sar_db=apply_flips(sar, code), sar_valid=apply_flips(native, code),
                      optical_valid=apply_flips(optical, code), label=apply_flips(target, code),
                      original_label=apply_flips(original, code))
        result = {k: torch.from_numpy(np.ascontiguousarray(v)) for k,v in sample.items()}
        result.update(scene_id=scene, source_id=row["source"], index=row["index"], top=top, left=left,
                      sampling_kind=row["kind"], sampling_class=row["class_id"], flip_code=code,
                      redraw_count=row["redraw_count"], fallback_used=row["fallback_used"])
        return result

    def close(self):
        for reader in getattr(self, "_aux", {}).values():
            reader.close()
        self._aux, self._cache = OrderedDict(), OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=OrderedDict(), _aux=OrderedDict())
        return state

    def __del__(self):
        self.close()


__all__ = ["prepare_pilot_support", "load_pilot_support", "make_pilot_plan", "replay_plan",
           "summarize_replay", "MS75SARPilotDataset", "known_water_boundary", "boundary_reservoir", "MODEL_INPUT_KEYS"]
