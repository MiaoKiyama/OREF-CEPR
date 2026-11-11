"""QS-OREF fitting on authenticated OOF120 or replay120 inputs."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import contextlib
import gc
import hashlib
from pathlib import Path
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from native24_original82_data import read, write, save, sha
import training_common as common
from native24_object_graph_model import (
    ARMS, SEED, active_parameter_count, arm_spec, loss_value, make_model, mask_raw,
    masked_normalization, shared_parameter_hash,
)
from native24_object_graph_policy import ROUTES, fit_four_routes, frozen_four_routes

SOURCES = ("replay120", "oof120")
CHECKPOINTS = ("CE_8000", "CE_24000", "CE_60000", "CE_120000",
               "CE_softIoU_3000", "CE_softIoU_6000")
ENDPOINTS = ("CE_120000", "CE_softIoU_6000")
GRAPH_META = dict(geometry_width=12, relation_count=3, relation_width=38)


class DrawAudit(common.DrawAudit):
    def snapshot(self):
        result = super().snapshot()
        domains = sorted(set(self.domains.tolist()))
        result["pool_draws"] = {
            name: int(self.scene_draws[self.domains == name].sum()) for name in domains}
        result["checkpoint_interval_pool_draws"] = {
            name: int(self.interval_scene_draws[self.domains == name].sum()) for name in domains}
        assert sum(result["pool_draws"].values()) == result["total_draws"]
        return result


def job_root(c, source, arm):
    assert source in SOURCES and arm in ARMS
    return Path(c["output"]) / "jobs" / source / arm


def config_sha(c):
    return sha(Path(c["output"]) / "config.json")






def checkpoint_dir(c, source, arm, name):
    assert name in CHECKPOINTS
    return job_root(c, source, arm) / "checkpoints" / name


@contextlib.contextmanager
def task_lock(path):
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextlib.contextmanager
def fitting_scope(c, source):
    if source not in SOURCES:
        raise ValueError("Unsupported training source: " + source)
    if c["pools"][source]["official_truth_allowed"] is not False:
        raise ValueError("Target truth is forbidden during fitting")
    if source == "replay120":
        from native24_replay120_qc_worker import official_truth_forbidden
        with official_truth_forbidden(c):
            yield
    else:
        from native24_pool120_qc_worker import official_truth_forbidden
        with official_truth_forbidden(c):
            yield


@contextlib.contextmanager
def calibration_slot(c, out, name):
    import fcntl
    directory = Path(c["output"]) / "calibration_slots"
    directory.mkdir(exist_ok=True)
    selected = None
    while selected is None:
        available = next(int(line.split()[1]) / 2**20
                         for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        if available >= float(c.get("cq_calibration_min_available_gib", 22)):
            for index in range(int(c.get("calibration_workers", 2))):
                candidate = (directory / f"lane{index}.lock").open("a")
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    selected = candidate
                    break
                except BlockingIOError:
                    candidate.close()
        if selected is None:
            common.stamp(out, "waiting_for_calibration", checkpoint=name,
                         available_gib=available)
            time.sleep(10)
    try:
        yield
    finally:
        fcntl.flock(selected, fcntl.LOCK_UN)
        selected.close()


class Inputs:
    """Observable input augmentation; never constructs geometry from targets."""
    def __init__(self, c, bank, source, arm):
        self.c, self.bank, self.source, self.arm = c, bank, source, arm
        self.rows = bank.rows
        self.ends = np.asarray([r["local_end"] for r in self.rows], np.int64)
        self.starts = np.asarray([r["local_start"] for r in self.rows], np.int64)
        self.graphs, self.sar = {}, {}

    def domain(self, row):
        return "replay" if self.source == "replay120" else "oof"

    def raw(self, indices):
        value = self.bank.raw(indices)
        if self.source == "replay120":
            from native24_object_graph_features import sar_scene
            owners = np.searchsorted(self.ends, indices, side="right")
            for number in np.unique(owners):
                row = self.rows[number]
                sid = row["scene"]
                if sid not in self.sar:
                    self.sar[sid] = sar_scene(self.c, sid, "replay")
                    assert self.sar[sid].shape == (
                        row["local_end"] - row["local_start"], 7)
                select = np.flatnonzero(owners == number)
                value[select, 207:214] = self.sar[sid][indices[select] - self.starts[number]]
        return value

    def graph(self, indices, structure=None):
        from native24_object_graph_features import SceneGraph
        count = len(indices)
        output = dict(
            geometry=np.empty((count, 7, 12), np.float32),
            relations=np.empty((count, 7, 3, 38), np.float32),
            relation_mask=np.empty((count, 7, 3), np.float32),
        )
        if structure is None:
            structure = ("shuffled" if arm_spec(self.arm)["shuffled"] else
                         "no_edges" if not arm_spec(self.arm)["edges"] else "full")
        owners = np.searchsorted(self.ends, indices, side="right")
        for number in np.unique(owners):
            row = self.rows[number]
            key = (self.domain(row), row["scene"])
            if key not in self.graphs:
                self.graphs[key] = SceneGraph(self.c, key[1], key[0])
            selected = np.flatnonzero(owners == number)
            values = self.graphs[key].get(indices[selected] - self.starts[number],
                                          structure=structure)
            for name, array in output.items():
                array[selected] = values[name]
        assert all(np.isfinite(a).all() for a in output.values())
        return output


def training_bank(c, source):
    if source not in SOURCES:
        raise ValueError("Unsupported training source: " + source)
    if source == "replay120":
        from native24_replay120_qc_data import TrainBank120
        bank = TrainBank120(c)
    else:
        from native24_pool120_qc_data import TrainBank120
        bank = TrainBank120(c)
    assert bank.ids == c["pools"][source]["fit_ids"]
    return bank


class Training:
    def __init__(self, c, source, arm, out, torch, bank=None, device="cuda"):
        self.c, self.source, self.arm = c, source, arm
        self.out, self.torch, self.device = Path(out), torch, device
        self.bank = bank if bank is not None else training_bank(c, source)
        self.pack = self.bank.target_pack()
        self.inputs = Inputs(c, self.bank, source, arm)
        self.mass = self.pack["counts"][:, 1:].sum(1, dtype=np.float32)
        self.valid = np.flatnonzero(self.mass > 0)
        assert len(self.valid)
        self.sqrtmean = float(np.sqrt(self.mass[self.valid]).mean())
        sample = np.random.default_rng(SEED).choice(
            self.bank.n, min(self.bank.n, int(c["normalization_samples"])), replace=False)
        raw = self.inputs.raw(sample)
        mean = raw.mean(0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(raw.std(0, dtype=np.float64), .05).astype(np.float32)
        self.mean, self.scale = masked_normalization(mean, scale, arm)
        del raw
        self.graph_norm = dict(
            geometry_mean=np.zeros(12, np.float32),
            geometry_scale=np.ones(12, np.float32),
            relations_mean=np.zeros(38, np.float32),
            relations_scale=np.ones(38, np.float32),
        )
        if arm_spec(arm)["graph"]:
            # All graph structures use the identical unshuffled fitting statistics.
            graph = self.inputs.graph(sample, structure="full")
            for key, axes in (("geometry", (0, 1)), ("relations", (0, 1, 2))):
                values = graph[key]
                self.graph_norm[key + "_mean"] = values.mean(axes, dtype=np.float64).astype(np.float32)
                self.graph_norm[key + "_scale"] = np.maximum(
                    values.std(axes, dtype=np.float64), .05).astype(np.float32)
            del graph
        from native24_original82_policy import fit_prior
        self.prior = fit_prior(self.pack)
        self.fixed = torch.as_tensor(self.pack["fixed"][1:, 1:], dtype=torch.float32, device=device)
        self.audit = dict(
            source=source, arm=arm, fit_ids=self.bank.ids,
            normalization_ids=self.bank.ids, prior_ids=self.bank.ids,
            calibration_ids=self.bank.ids, checkpoint_selection_ids=self.bank.ids,
            official_truth_used=False, original_folds_are_reporting_only=True,
            groups=self.bank.n, valid_groups=len(self.valid), sqrtmean=self.sqrtmean,
            normalization_indices_sha256=hashlib.sha256(sample.astype("<i8").tobytes()).hexdigest(),
            mean_sha256=hashlib.sha256(self.mean.tobytes()).hexdigest(),
            scale_sha256=hashlib.sha256(self.scale.tobytes()).hexdigest(),
            graph_norm_sha256={key: hashlib.sha256(value.tobytes()).hexdigest()
                               for key, value in self.graph_norm.items()},
            Q_enabled=arm_spec(arm)["Q"], S_enabled=arm_spec(arm)["S"],
            replay_S_restored_from_original_observable_sidecar=source == "replay120",
            sampler="Dedicated CPU torch.Generator, continuous across optimizer reset",
            time_unix=time.time(),
        )
        write(self.out / "data_audit.json", self.audit)

    def cpu_batch(self, generator):
        positions = self.torch.randint(len(self.valid), (int(self.c["batch_size"]),),
                                       generator=generator, device="cpu").numpy()
        indices = self.valid[positions]
        x = (mask_raw(self.inputs.raw(indices), self.arm) - self.mean) / self.scale
        counts = self.pack["counts"][indices, 1:].astype(np.float32)
        weight = (np.sqrt(self.mass[indices]) / self.sqrtmean).astype(np.float32)
        labels = self.pack["labels"][indices].astype(np.int64) - 1
        graph = None
        if arm_spec(self.arm)["graph"]:
            values = self.inputs.graph(indices)
            graph = normalize_graph(values, self.graph_norm)
        return indices, (x, counts, weight, labels, graph)

    def transfer(self, values):
        x, counts, weight, labels, graph = values
        t = self.torch
        x, counts, weight, labels = [
            t.as_tensor(a, device=self.device) for a in (x, counts, weight, labels)]
        target = counts / counts.sum(1, keepdim=True).clamp_min(1)
        if graph is not None:
            graph = tuple(t.as_tensor(a, device=self.device) for a in graph)
        return x, target, counts, weight, labels, graph


def normalize_graph(values, norm):
    return (
        (values["geometry"] - norm["geometry_mean"]) / norm["geometry_scale"],
        (values["relations"] - norm["relations_mean"]) / norm["relations_scale"],
        values["relation_mask"],
    )


class Prefetch:
    """One continuous queue spans both phases; no sampled minibatch is discarded."""
    def __init__(self, run, generator, count):
        self.run, self.generator, self.remaining = run, generator, count
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = self.pool.submit(run.cpu_batch, generator)

    def next(self):
        assert self.remaining > 0
        indices, values = self.pending.result()
        self.remaining -= 1
        if self.remaining:
            self.pending = self.pool.submit(self.run.cpu_batch, self.generator)
        return indices, self.run.transfer(values)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def predict_posterior(model, inputs, mean, scale, graph_norm, torch, device="cuda"):
    posterior = np.empty((inputs.bank.n, 24), np.float32)
    model.eval()
    with torch.inference_mode():
        for row in inputs.rows:
            for start in range(row["local_start"], row["local_end"], 8192):
                end = min(row["local_end"], start + 8192)
                indices = np.arange(start, end, dtype=np.int64)
                x = (mask_raw(inputs.raw(indices), inputs.arm) - mean) / scale
                graph = None
                if arm_spec(inputs.arm)["graph"]:
                    graph = tuple(torch.as_tensor(a, device=device) for a in
                                  normalize_graph(inputs.graph(indices), graph_norm))
                posterior[start:end] = model(
                    torch.as_tensor(x, device=device), graph=graph).softmax(1).cpu().numpy()
    assert np.isfinite(posterior).all()
    np.testing.assert_allclose(posterior.sum(1), 1, rtol=1e-5, atol=1e-6)
    return posterior


def summarize(arrays, selected):
    from native24_original82_reference import miou
    baseline = arrays["baseline"][selected].sum(0)
    base_score = float(100 * miou(baseline))
    base_valid = baseline[1:, 1:].astype(np.float64)
    base_diag = base_valid.diagonal()
    base_correct = int(base_diag.sum())
    output = dict(baseline=dict(
        miou_percent=base_score,
        class_iou_percent=(100 * base_diag / np.maximum(
            base_valid.sum(0) + base_valid.sum(1) - base_diag, 1)).tolist(),
        correct_pixels=base_correct, class_correct_pixels=base_diag.astype(np.int64).tolist(),
    ))
    for name in arrays:
        if name not in ROUTES:
            continue
        cm = arrays[name][selected].sum(0)
        stats = arrays[name + "_stats"][selected].sum(0)
        value = float(100 * miou(cm))
        valid = cm[1:, 1:].astype(np.float64)
        diagonal = valid.diagonal()
        output[name] = dict(
            miou_percent=value, delta_vs_baseline_pp=value - base_score,
            class_iou_percent=(100 * diagonal / np.maximum(
                valid.sum(0) + valid.sum(1) - diagonal, 1)).tolist(),
            total_rescued=int(stats[1, 1:].sum()),
            total_broken_correct=int(stats[0, 1:].sum()),
            class_rescued_pixels=stats[1, 1:].tolist(),
            class_broken_correct_pixels=stats[0, 1:].tolist(),
            baseline_correct_pixels=base_correct,
            class_baseline_correct_pixels=base_diag.astype(np.int64).tolist(),
            broken_correct_rate_percent=float(100 * stats[0, 1:].sum() / max(base_correct, 1)),
            class_broken_correct_rate_percent=(
                100 * stats[0, 1:] / np.maximum(base_diag, 1)).tolist(),
        )
    return dict(scene_count=len(selected), **output)


def score_routes(bank, routes, out):
    from native24_original82_reference import score_route
    arrays = {"baseline": []}
    for name in routes:
        arrays[name], arrays[name + "_stats"] = [], []
    ids, domains, folds = [], [], []
    for row in bank.rows:
        d = bank.scene(row, targets=True)
        labels, counts = d["labels"], d["counts"]
        baseline = d["baselines"][2]
        np.testing.assert_array_equal(
            baseline, score_route(d["fixed"], counts, labels, np.full(len(labels), 2, np.uint8)))
        arrays["baseline"].append(baseline)
        selected = slice(row["local_start"], row["local_end"])
        for name, route in routes.items():
            cm = score_route(d["fixed"], counts, labels, route[selected])
            stats = common.correction_stats(counts, labels, route[selected])
            np.testing.assert_array_equal(cm.sum(1), baseline.sum(1))
            np.testing.assert_array_equal(np.diag(cm - baseline)[1:], (stats[1] - stats[0])[1:])
            arrays[name].append(cm)
            arrays[name + "_stats"].append(stats)
        ids.append(row["scene"])
        domains.append(row["domain"])
        folds.append(int(row["original_fold"]))
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    save(Path(out) / "confusions.npz", **arrays, scene_ids=np.asarray(ids),
         domains=np.asarray(domains), original_folds=np.asarray(folds, np.int8))
    result = {"fitting_pool": summarize(arrays, np.arange(len(ids)))}
    for domain in sorted(set(domains)):
        result[domain] = summarize(arrays, np.flatnonzero(np.asarray(domains) == domain))
    result["folds"] = {
        str(f): summarize(arrays, np.flatnonzero(np.asarray(folds) == f)) for f in range(5)}
    return result, arrays


def evaluate_checkpoint(c, run, model, name, updates, draw_audit, initial_hash, loss_curve):
    out = checkpoint_dir(c, run.source, run.arm, name)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with calibration_slot(c, run.out, name):
        common.stamp(run.out, "predicting_fitting_pool", checkpoint=name, total_updates=updates)
        p = predict_posterior(model, run.inputs, run.mean, run.scale,
                              run.graph_norm, run.torch, run.device)
        common.stamp(run.out, "calibrating_C_and_I", checkpoint=name, total_updates=updates)
        policy, routes = fit_four_routes(p, run.pack, run.prior, include_I=name in ENDPOINTS)
        if name in ENDPOINTS:
            observed = {key: run.pack[key] for key in ("labels", "mass", "fixed_obs", "pixels")}
            replay, _ = frozen_four_routes(p, observed, policy)
            for key in ROUTES:
                np.testing.assert_array_equal(routes[key], replay[key])
            del replay
        del p
        gc.collect()
        results, arrays = score_routes(run.bank, routes, out)
        np.testing.assert_array_equal(arrays["baseline"].sum(0), run.pack["baselines"][2])
        state = dict(
            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            mean=run.mean, scale=run.scale, graph_norm=run.graph_norm,
            graph_meta=GRAPH_META, source=run.source, arm=run.arm, checkpoint=name,
            training_ids=run.bank.ids, seed=SEED,
        )
        run.torch.save(state, out / "model.pt")
        write(out / "policy.json", policy)
        receipt = dict(
            status="completed", source=run.source, arm=run.arm, checkpoint=name,
            phase=name.rsplit("_", 1)[0], step=int(name.rsplit("_", 1)[1]),
            total_updates=updates, results=results,
            C_fitting_miou_percent=policy["C_fitting_miou_percent"],
            I_selected={key: value["iterations"] for key, value in policy.get("I", {}).items()},
            model_sha256=sha(out / "model.pt"), policy_sha256=sha(out / "policy.json"),
            confusions_sha256=sha(out / "confusions.npz"),
            route_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in routes.items()},
            input_audit=run.audit, draw_audit=draw_audit,
            initial_parameters_sha256=initial_hash, loss_curve=loss_curve,
            config_sha256=config_sha(c), seconds=time.time() - started, time_unix=time.time(),
        )
        write(out / "completed.json", receipt)
        del routes, state, arrays
        gc.collect()
    return receipt


def verify_training(c, source, arm):
    root = job_root(c, source, arm)
    receipt = read(root / "training_completed.json")
    assert receipt["status"] == "completed" and receipt["config_sha256"] == config_sha(c)
    assert receipt["source"] == source and receipt["arm"] == arm
    assert receipt["optimizer_updates"] == 126000
    assert set(receipt["checkpoints"]) == set(CHECKPOINTS)
    audit = receipt["input_audit"]
    expected = c["pools"][source]["fit_ids"]
    for key in ("fit_ids", "normalization_ids", "prior_ids",
                "calibration_ids", "checkpoint_selection_ids"):
        assert audit[key] == expected
    assert receipt["official_truth_used"] == (False)
    soft = read(root / "soft_stage_start.json")
    assert sha(root / "soft_stage_start.json") == receipt["soft_stage_start_sha256"]
    assert soft["starting_checkpoint"] == receipt["best_CE_checkpoint"]
    assert soft["cumulative_updates"] == 120000
    assert soft["optimizer_reset"] is True and soft["sampler_reset"] is False
    assert soft["selection_criterion"] == "same_pool_C_calibrated_best_CE"
    assert soft["model_file_sha256"] == receipt["checkpoints"][
        receipt["best_CE_checkpoint"]]["model_sha256"]
    for name, summary in receipt["checkpoints"].items():
        out = checkpoint_dir(c, source, arm, name)
        assert sha(out / "completed.json") == summary["completed_sha256"]
        done = read(out / "completed.json")
        assert done["config_sha256"] == config_sha(c)
        assert done["source"] == source and done["arm"] == arm and done["checkpoint"] == name
        assert sha(out / "model.pt") == done["model_sha256"] == summary["model_sha256"]
        assert sha(out / "policy.json") == done["policy_sha256"] == summary["policy_sha256"]
    return receipt


def train(c, source, arm, torch):
    out = job_root(c, source, arm)
    if (out / "training_completed.json").exists():
        return verify_training(c, source, arm)
    assert not (out / "started.json").exists(), "Partial attempt requires explicit recovery."
    assert c["seed"] == SEED and c["batch_size"] == 4096
    assert c["cq_ce_steps"] == 120000 and c["cq_soft_steps"] == 6000
    write(out / "started.json", dict(source=source, arm=arm, pid=os.getpid(),
                                     config_sha256=config_sha(c), time_unix=time.time()))
    with fitting_scope(c, source):
        run = Training(c, source, arm, out, torch)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = make_model(torch, run.mean, run.scale, arm, GRAPH_META).to("cuda")
        initial_hash = shared_parameter_hash(model)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        draws = DrawAudit(run.bank)
        prefetch = Prefetch(run, generator, 126000)
        best_score, best_ce = -np.inf, None
        history, curve, updates = {}, [], 0
        begun = time.time()
        try:
            for phase, steps, checks, rate in (
                ("CE", 120000, c["cq_ce_checkpoints"], .001),
                ("CE_softIoU", 6000, c["cq_soft_checkpoints"], .00008),
            ):
                if phase != "CE":
                    assert best_ce is not None and best_ce.startswith("CE_")
                    chosen_model = checkpoint_dir(c, source, arm, best_ce) / "model.pt"
                    assert sha(chosen_model) == history[best_ce]["model_sha256"]
                    state = torch.load(chosen_model, map_location="cpu", weights_only=False)
                    model.load_state_dict(state["state"])
                    assert updates == draws.snapshot()["total_updates"] == 120000
                    write(out / "soft_stage_start.json", dict(
                        source=source, arm=arm, starting_checkpoint=best_ce,
                        model_file_sha256=sha(chosen_model),
                        loaded_state_sha256=common.parameter_hash(model),
                        C_fitting_miou_percent=best_score,
                        selection_criterion="same_pool_C_calibrated_best_CE",
                        optimizer_reset=True, sampler_reset=False,
                        cumulative_updates=updates,
                        consumed_draw_indices_sha256=draws.snapshot()["cumulative_indices_sha256"],
                        continuous_prefetch_queue_retained=True, time_unix=time.time(),
                    ))
                    del state
                optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=.0001)
                for step in range(1, steps + 1):
                    model.train()
                    indices, batch = prefetch.next()
                    draws.update(indices)
                    loss, ce, auxiliary, soft = loss_value(
                        model, batch, phase, run.fixed, len(run.valid), torch, arm_spec(arm)["S"])
                    assert torch.isfinite(loss).item()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    updates += 1
                    if step % 250 == 0:
                        row = dict(phase=phase, step=step, total_updates=updates,
                                   loss=float(loss.detach()), CE=float(ce), SAR_aux=float(auxiliary),
                                   softIoU=float(soft), elapsed=time.time() - begun)
                        curve.append(row)
                        common.stamp(out, "training", **row)
                    if step in checks:
                        name = f"{phase}_{step}"
                        record = evaluate_checkpoint(
                            c, run, model, name, updates, draws.snapshot(), initial_hash, curve)
                        history[name] = dict(
                            C_fitting_miou_percent=record["C_fitting_miou_percent"],
                            completed_sha256=sha(checkpoint_dir(c, source, arm, name) / "completed.json"),
                            model_sha256=record["model_sha256"], policy_sha256=record["policy_sha256"])
                        if phase == "CE" and record["C_fitting_miou_percent"] > best_score:
                            best_score, best_ce = record["C_fitting_miou_percent"], name
                        draws.reset_interval()
                        write(out / "history.json", history)
                del optimizer
        finally:
            prefetch.close()
        audit = draws.snapshot()
        assert updates == audit["total_updates"] == 126000
        assert audit["total_draws"] == 126000 * 4096
        receipt = dict(
            status="completed", source=source, arm=arm, seed=SEED,
            optimizer_updates=updates, best_CE_checkpoint=best_ce,
            fixed_final_checkpoint="CE_softIoU_6000",
            attribution_checkpoints=list(ENDPOINTS), checkpoints=history,
            initial_parameters_sha256=initial_hash,
            draw_indices_sha256=audit["cumulative_indices_sha256"],
            draw_audit=audit, total_draws=audit["total_draws"], input_audit=run.audit,
            shared_parameters=sum(p.numel() for n, p in model.named_parameters()
                                  if not n.startswith(("geometry_", "relation_", "q_attribute_"))),
            parameters=sum(p.numel() for p in model.parameters()),
            active_path_parameters=active_parameter_count(model, arm),
            matched_budget="same optimizer updates and sampled groups; compute is not equal",
            config_sha256=config_sha(c), official_truth_used=False,
            soft_stage_start_sha256=sha(out / "soft_stage_start.json"),
            seconds=time.time() - begun, time_unix=time.time(),
        )
        write(out / "training_completed.json", receipt)
        common.stamp(out, "training_completed", total_updates=updates)
        return receipt














class SmokeSceneBank:
    """Disposable one-scene view of genuine source inputs/targets for timing."""
    def __init__(self, bank, scene):
        row = next(dict(r) for r in bank.rows if r["scene"] == scene)
        self.base, self.source_row = bank, row
        self.offset = row["local_start"]
        self.n = row["local_end"] - row["local_start"]
        self.rows = [dict(row, local_start=0, local_end=self.n)]
        self.ids, self.pixels = [scene], row["pixels"]

    def raw(self, indices):
        return self.base.raw(np.asarray(indices, np.int64) + self.offset)

    def scene(self, row, targets=True):
        return self.base.scene(self.source_row, targets=targets)

    def target_pack(self):
        d = self.scene(self.rows[0])
        return dict(counts=d["counts"], labels=d["labels"], mass=d["mass"],
                    fixed=d["fixed"], fixed_obs=d["fixed_prediction_counts"],
                    baselines=d["baselines"], pixels=self.pixels)


def smoke(c, source, arm, torch, output, scene=None, updates=8):
    output = Path(output).resolve()
    jobs = (Path(c["output"]) / "jobs").resolve()
    assert output != jobs and jobs not in output.parents
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "passed.json").exists(), "Use a fresh disposable smoke path."
    temporary = dict(c, normalization_samples=min(c["normalization_samples"], 2000))
    with fitting_scope(c, source):
        bank = training_bank(c, source)
        if scene is not None:
            bank = SmokeSceneBank(bank, scene)
        started = time.perf_counter()
        run = Training(temporary, source, arm, output, torch, bank=bank)
        preparation = time.perf_counter() - started
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = make_model(torch, run.mean, run.scale, arm, GRAPH_META).to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        cpu_seconds, transfer_seconds, gpu_seconds, loss_values = [], [], [], []
        graph_seconds, raw_seconds, seen_scenes = [], [], set()
        initial = shared_parameter_hash(model)
        for update in range(updates):
            positions = torch.randint(len(run.valid), (c["batch_size"],), generator=generator).numpy()
            indices = run.valid[positions]
            owners = np.searchsorted(run.inputs.ends, indices, side="right")
            seen_scenes.update(run.bank.rows[i]["scene"] for i in np.unique(owners))
            start = time.perf_counter()
            run.inputs.raw(indices)
            raw_seconds.append(time.perf_counter() - start)
            if arm_spec(arm)["graph"]:
                start = time.perf_counter()
                run.inputs.graph(indices)
                graph_seconds.append(time.perf_counter() - start)
            start = time.perf_counter()
            _, values = run.cpu_batch(generator)
            cpu_seconds.append(time.perf_counter() - start)
            start = time.perf_counter()
            batch = run.transfer(values)
            torch.cuda.synchronize()
            transfer_seconds.append(time.perf_counter() - start)
            start = time.perf_counter()
            loss, _, _, _ = loss_value(
                model, batch, "CE" if update < updates // 2 else "CE_softIoU",
                run.fixed, len(run.valid), torch, arm_spec(arm)["S"])
            assert torch.isfinite(loss).item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            torch.cuda.synchronize()
            gpu_seconds.append(time.perf_counter() - start)
            loss_values.append(float(loss.detach()))
        flops = dict(status="unavailable")
        try:
            from torch.utils.flop_counter import FlopCounterMode
            model.zero_grad(set_to_none=True)
            with FlopCounterMode(display=False) as counter:
                probe_loss = loss_value(
                    model, batch, "CE", run.fixed, len(run.valid), torch, arm_spec(arm)["S"])[0]
                forward = int(counter.get_total_flops())
                probe_loss.backward()
                forward_backward = int(counter.get_total_flops())
            flops = dict(
                status="available", batch_size=c["batch_size"],
                neural_CE_forward_flops=forward,
                neural_CE_forward_backward_flops=forward_backward,
                includes_S_aux=arm_spec(arm)["S"],
                excludes="optimizer, CPU data/graph assembly, C/I policy calibration",
            )
        except (ImportError, AttributeError, RuntimeError, NotImplementedError) as exc:
            flops = dict(status="unavailable", reason=type(exc).__name__ + ": " + str(exc))
        result = dict(
            status="passed", disposable=True, source=source, arm=arm,
            smoke_scene=scene, batch_size=c["batch_size"], updates=updates,
            preparation_seconds=preparation, scenes_touched=len(seen_scenes),
            source_scenes=len(bank.ids), input_audit=run.audit,
            raw_fetch_seconds=raw_seconds, graph_fetch_seconds=graph_seconds,
            cpu_batch_seconds=cpu_seconds, transfer_seconds=transfer_seconds,
            gpu_update_seconds=gpu_seconds, loss_values=loss_values,
            estimated_sequential_steps_per_second=1 / float(
                np.mean(cpu_seconds + transfer_seconds + gpu_seconds) * 3),
            initial_parameters_sha256=initial,
            parameters=sum(p.numel() for p in model.parameters()),
            active_path_parameters=active_parameter_count(model, arm),
            flop_probe=flops, matched_budget="updates/draws, not equal compute",
            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            official_truth_used=False, time_unix=time.time(),
        )
        write(output / "passed.json", result)
        print({k: result[k] for k in (
            "status", "source", "arm", "scenes_touched", "preparation_seconds",
            "estimated_sequential_steps_per_second", "peak_gpu_allocated_gib")}, flush=True)
        return result





def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("train", "smoke"), required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--arm", choices=ARMS, default="graphQS")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu-slot", type=int, default=0)
    p.add_argument("--smoke-out")
    p.add_argument("--smoke-scene")
    p.add_argument("--smoke-updates", type=int, default=8)
    args = p.parse_args()
    from native24_object_graph_prepare import verify, validate_config
    c = read(args.config)
    validate_config(c)
    c["_path"] = str(Path(args.config).resolve())
    if args.mode == "smoke" and not args.smoke_out:
        p.error("--smoke-out is required for an isolated diagnostic run")
    verify(c)
    out = Path(args.smoke_out) if args.mode == "smoke" else job_root(c, args.source, args.arm)
    out.mkdir(parents=True, exist_ok=True)
    with task_lock(out / "worker.lock"):
        torch = common.configure(c, args.gpu, args.cpu_slot % 8)
        if args.mode == "train":
            train(c, args.source, args.arm, torch)
        else:
            smoke(c, args.source, args.arm, torch, out, args.smoke_scene, args.smoke_updates)

if __name__ == "__main__":
    main()
