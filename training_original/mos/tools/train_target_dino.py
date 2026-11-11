"""Fair train/held selection followed by a fresh full-train target DINO fit.

No official test raster is read. Every stage starts from authenticated generic
pretraining unless resuming that exact stage's atomic optimizer checkpoint.
Only the best selection EMA and one rolling recovery are retained. The full
fit keeps the original 70-epoch LR horizon even when its selected duration is
shorter. Formal training cannot silently fall back to a different batch size.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcmmsline3.losses import lovasz_softmax_loss, soft_dice_loss

CANDIDATE_EPOCHS = (2, 4, 6, 8, 10, 12, 16, 18, 20, 24, 30, 40, 50, 60, 70)
PROTOCOL = {
    "schema": "target_dino_training_v1",
    "seed": 1337, "horizon_epochs": 70, "effective_batch": 64,
    "epoch_crops_per_scene": 320, "warmup_epochs": 2000 / 480,
    "loss": {"ce": 1.0, "dice": 0.5, "lovasz": 0.25, "reduction": "per_image_mean"},
    "loss_dtype": "float32", "autocast": "bfloat16", "autocast_cache": False,
    "optimizer": "AdamW", "head_lr": 0.0002, "layer_decay": 0.9,
    "weight_decay": 0.05, "gradient_clip": 1.0, "ema_decay": 0.999,
    "ema_first_successful_update": "copy_raw_model",
    "v3l": {"backbone_lr_multiplier": 0.05, "permanent": "rgb_stem,blocks0:12,prefix",
             "staged": "blocks12:24,norm", "unfreeze_epoch_zero_based": 5},
    "v2l": {"backbone_lr_multiplier": 0.1, "trainable": "entire_model"},
    "candidate_epochs": list(CANDIDATE_EPOCHS),
    "selection": "maximum_whole_held_global_fixed_class_miou_tie_earlier",
    "full_init": "original_pretrained_not_selection_checkpoint",
    "full_scheduler": "seventy_epoch_horizon_not_compressed_to_selected_duration",
    "official_test_auto": False,
}
_STOP_REQUESTED = False


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_config(config: dict) -> None:
    anchors = {"schema_version": 1, "seed": 1337, "max_epochs": 70,
               "candidate_epochs": list(CANDIDATE_EPOCHS), "auto_full": True}
    for key, expected in anchors.items():
        if config.get(key) != expected:
            raise ValueError(f"fixed target training protocol requires {key}={expected!r}")
    if config.get("dataset") not in ("gid5", "whu") or config.get("variant") not in ("v3l", "v2l"):
        raise ValueError("unsupported dataset or expert")
    # Physical batch may be reduced only by an explicit, recorded new config.
    if (config.get("batch_size"), config.get("accumulation_steps")) not in ((4, 16), (2, 32)):
        raise ValueError("physical batch/accumulation must be 4x16 or explicitly configured 2x32")
    if config.get("batch_size") == 2 and not config.get("batch2_oom_evidence"):
        raise ValueError("batch2 requires explicit OOM-smoke evidence in the committed config")
    for key in ("num_workers", "torch_threads", "eval_batch_size", "eval_workers"):
        value = config.get(key)
        if type(value) is not int or value < (0 if key.endswith("workers") else 1):
            raise ValueError(f"invalid {key}")
    if config["eval_batch_size"] not in (2, 4):
        raise ValueError("evaluation batch must be fixed at 2 or 4")
    if not config.get("source_files") or len(set(config["source_files"])) != len(config["source_files"]):
        raise ValueError("an explicit unique source manifest is required")
    required = ("output_dir", "data_manifest", "pretrained_file", "pretrained_sha256")
    if any(not config.get(key) for key in required):
        raise ValueError("missing pinned training path")


def build_binding(config: dict) -> tuple[dict, dict]:
    from tcmmsline3.target_dino_data import load_manifest
    validate_config(config)
    metadata = load_manifest(config["data_manifest"])
    if metadata["dataset"] != config["dataset"] or metadata["seed"] != config["seed"]:
        raise ValueError("config and training manifest disagree")
    expected = {"gid5": (120, 96, 24, 30, 5), "whu": (80, 64, 16, 20, 7)}[config["dataset"]]
    observed = tuple(len(metadata[k]) for k in ("train_ids", "selection_train_ids", "held_ids", "test_ids"))
    if (*observed, metadata["num_classes"]) != expected or not metadata.get("strict_protocol", True):
        raise ValueError("formal training needs the full authenticated target split")
    sources = {str(Path(path).resolve()): sha256(path) for path in config["source_files"]}
    if str(Path(__file__).resolve()) not in sources:
        raise ValueError("trainer itself must be source-bound")
    if sha256(config["pretrained_file"]) != config["pretrained_sha256"]:
        raise ValueError("generic pretraining checksum mismatch")
    binding = {"config": copy.deepcopy(config), "config_sha256": canonical_hash(config),
               "protocol": PROTOCOL, "protocol_sha256": canonical_hash(PROTOCOL),
               "source_files": sources, "data_manifest_sha256": sha256(config["data_manifest"]),
               "data_request_sha256": metadata["request_sha256"],
               "pretrained_sha256": config["pretrained_sha256"],
               "splits": {k: metadata[k] for k in ("train_ids", "selection_train_ids", "held_ids", "test_ids")}}
    binding["binding_sha256"] = canonical_hash(binding)
    return binding, metadata


def schedule_shape(scene_count: int) -> dict:
    samples = scene_count * 320
    if samples % 64:
        raise ValueError("epoch sample count is not divisible by effective batch")
    updates = samples // 64
    # Integer nearest warmup is declared once; WHU320 gives1333, WHU4001667.
    warmup = int(math.floor(updates * (2000 / 480) + 0.5))
    return {"samples_per_epoch": samples, "updates_per_epoch": updates,
            "horizon_updates": updates * 70, "warmup_updates": warmup}


def lr_multiplier(update: int, horizon: int, warmup: int) -> float:
    if update < 0 or not 0 < warmup < horizon:
        raise ValueError("invalid scheduler position")
    if update < warmup:
        return (update + 1) / warmup
    progress = min(max((update - warmup) / (horizon - warmup), 0), 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def per_image_segmentation_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Batch-independent Dice/Lovasz reductions preserve physical-batch fairness."""
    logits = logits.float()
    if labels.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("target/logit shapes differ")
    if torch.any((labels != 255) & ((labels < 0) | (labels >= logits.shape[1]))):
        raise ValueError("invalid target class ID")
    total = logits.sum() * 0.0
    for i in range(logits.shape[0]):
        target = labels[i:i + 1].clone().long()
        target[target == 255] = -1
        if not (target >= 0).any():
            continue
        sample = logits[i:i + 1]
        ce = F.cross_entropy(sample, target, ignore_index=-1)
        total = total + ce + 0.5 * soft_dice_loss(sample, target, logits.shape[1])
        total = total + 0.25 * lovasz_softmax_loss(sample, target)
    return total / logits.shape[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


class TensorEMA:
    """Tensor-only EMA; no second GPU model or module execution state."""
    def __init__(self, model: torch.nn.Module):
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        raw = model.state_dict()
        if raw.keys() != self.state.keys():
            raise RuntimeError("model state changed after EMA construction")
        for name, value in raw.items():
            if self.updates == 0 or not value.is_floating_point():
                self.state[name].copy_(value.detach())
            else:
                self.state[name].lerp_(value.detach(), 1 - PROTOCOL["ema_decay"])
        self.updates += 1

    def load(self, state: dict, updates: int) -> None:
        if state.keys() != self.state.keys():
            raise ValueError("EMA state keys changed")
        for name, value in state.items():
            if value.shape != self.state[name].shape or value.dtype != self.state[name].dtype:
                raise ValueError("EMA state shape/dtype changed")
            self.state[name].copy_(value)
        self.updates = int(updates)


@contextmanager
def use_ema(model: torch.nn.Module, ema: TensorEMA):
    """Swap parameters in-place; restore all module modes, grad flags and RNG."""
    raw = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    modes = [(module, module.training) for module in model.modules()]
    grads = [(p, p.requires_grad) for p in model.parameters()]
    random_state = rng_state()
    try:
        model.load_state_dict(ema.state, strict=True)
        model.eval()
        yield model
    finally:
        model.load_state_dict(raw, strict=True)
        for module, training in modes:
            module.training = training
        for parameter, requires_grad in grads:
            parameter.requires_grad_(requires_grad)
        restore_rng(random_state)


def _tensor_bytes(value: Any, seen: set | None = None) -> int:
    seen = set() if seen is None else seen
    if torch.is_tensor(value):
        key = (value.device.type, value.device.index, value.untyped_storage().data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v, seen) for v in value)
    return 0


def checkpoint_roundtrip_validate(payload: dict, reloaded: dict) -> None:
    for key in ("schema", "role", "binding_sha256", "stage", "state"):
        if payload.get(key) != reloaded.get(key):
            raise RuntimeError(f"checkpoint roundtrip changed {key}")
    for key in ("model", "ema", "state_dict"):
        if key not in payload:
            continue
        if payload[key].keys() != reloaded[key].keys():
            raise RuntimeError("checkpoint tensor names changed")
        for name, value in payload[key].items():
            candidate = reloaded[key][name]
            if value.shape != candidate.shape or value.dtype != candidate.dtype:
                raise RuntimeError("checkpoint tensor layout changed")
            if value.is_floating_point() and not torch.isfinite(candidate).all():
                raise RuntimeError("checkpoint has nonfinite tensors")
    if "optimizer" in payload:
        before, after = payload["optimizer"], reloaded["optimizer"]
        if before["param_groups"] != after["param_groups"] or before["state"].keys() != after["state"].keys():
            raise RuntimeError("optimizer checkpoint layout changed")
        for key, state in after["state"].items():
            for name, value in state.items():
                original = before["state"][key][name]
                if torch.is_tensor(value) and (value.shape != original.shape or
                        (value.is_floating_point() and not torch.isfinite(value).all())):
                    raise RuntimeError("optimizer checkpoint corrupted")


def atomic_checkpoint(path: str | Path, payload: dict, *, reserve_bytes: int = 6 * 1024**3) -> None:
    """Old last survives until validated new last atomically replaces it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    estimate = _tensor_bytes(payload) + 64 * 1024**2
    if shutil.disk_usage(path.parent).free < estimate + reserve_bytes:
        raise RuntimeError(f"insufficient disk for atomic checkpoint plus {reserve_bytes / 1024**3:g}GiB reserve")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        restored = torch.load(temporary, map_location="cpu", weights_only=False)
        checkpoint_roundtrip_validate(payload, restored)
        del restored
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _state(stage: str, shape: dict) -> dict:
    return {"stage": stage, "epoch": 0, "next_index": 0, "updates": 0,
            "ema_updates": 0, "best_epoch": None, "best_miou": None,
            "evaluated_epochs": [], **shape}


def validate_progress(state: dict, stage: str, shape: dict) -> None:
    if state["stage"] != stage or any(state[k] != v for k, v in shape.items()):
        raise ValueError("resume stage/scheduler identity changed")
    epoch, offset, updates = state["epoch"], state["next_index"], state["updates"]
    if type(epoch) is not int or epoch < 0 or type(offset) is not int or not 0 <= offset < shape["samples_per_epoch"] or offset % 64:
        raise ValueError("resume is not on a valid optimizer boundary")
    if updates != epoch * shape["updates_per_epoch"] + offset // 64 or state["ema_updates"] != updates:
        raise ValueError("successful updates, samples and EMA count disagree")
    if any(e not in CANDIDATE_EPOCHS or e > epoch for e in state["evaluated_epochs"]):
        raise ValueError("invalid evaluated epoch receipt")


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)


def _loader(config: dict, stage: str, epoch: int, offset: int):
    from tcmmsline3.target_dino_data import TargetDataset
    norm = {"divisor": 255.0, **config["normalization"]}
    dataset = TargetDataset(config["data_manifest"], stage=stage, epoch=epoch,
                            start_index=offset, normalization=norm)
    generator = torch.Generator().manual_seed(config["seed"] + epoch * 1009)
    kwargs = {"dataset": dataset, "batch_size": config["batch_size"], "shuffle": False,
              "drop_last": False, "num_workers": config["num_workers"], "pin_memory": True,
              "worker_init_fn": _worker_init, "generator": generator}
    if config["num_workers"]:
        kwargs.update(prefetch_factor=2, persistent_workers=False)
    return dataset, DataLoader(**kwargs)


def _optimizer(model, config):
    from tcmmsline3.target_dino_model import parameter_groups
    optimizer = torch.optim.AdamW(parameter_groups(model, config), lr=2e-4,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def _payload(model, optimizer, ema, state, binding, stage):
    return {"schema": PROTOCOL["schema"], "role": "rolling_recovery", "stage": stage,
            "binding_sha256": binding["binding_sha256"], "binding": binding,
            "state": copy.deepcopy(state), "model": model.state_dict(), "ema": ema.state,
            "optimizer": optimizer.state_dict(), "rng": rng_state()}


def _compact(model_state, state, binding, stage, role):
    return {"schema": PROTOCOL["schema"], "role": role, "stage": stage,
            "binding_sha256": binding["binding_sha256"], "binding": binding,
            "state": copy.deepcopy(state), "state_dict": model_state}


def _progress(output: Path, stage: str, state: dict, status="running", **extra) -> None:
    row = {"status": status, "stage": stage, "time_unix": time.time(),
           "pid": os.getpid(), **state, **extra}
    atomic_json(output / "progress.json", row)
    print(json.dumps(row, allow_nan=False), flush=True)


def _restore(last: Path, model, optimizer, ema, binding, stage, shape):
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != PROTOCOL["schema"] or checkpoint.get("role") != "rolling_recovery" or checkpoint.get("binding_sha256") != binding["binding_sha256"] or checkpoint.get("stage") != stage:
        raise ValueError("resume checkpoint provenance mismatch")
    state = checkpoint["state"]
    validate_progress(state, stage, shape)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    ema.load(checkpoint["ema"], state["ema_updates"])
    restore_rng(checkpoint["rng"])
    del checkpoint
    return state


def _signal_handler(signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(json.dumps({"status": "stop_requested", "signal": signum,
                      "action": "save_after_clean_optimizer_boundary"}), flush=True)


def _train_update(model, optimizer, ema, batches, shape: dict, update: int, config: dict) -> dict:
    multiplier = lr_multiplier(update, shape["horizon_updates"], shape["warmup_updates"])
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier
    optimizer.zero_grad(set_to_none=True)
    count, total_loss = 0, 0.0
    indices = []
    for batch in batches:
        ms = batch["ms"].cuda(non_blocking=True)
        target = batch["label"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = model(ms)["logits"]
        with torch.autocast("cuda", enabled=False):
            loss = per_image_segmentation_loss(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss; no optimizer/EMA/scheduler advance")
        (loss * (len(ms) / 64)).backward()
        total_loss += float(loss.detach()) * len(ms) / 64
        count += len(ms)
        indices.extend(int(x) for x in batch["index"])
    if count != 64 or indices != list(range(indices[0], indices[0] + 64)):
        raise RuntimeError("effective optimizer update did not consume exactly64 ordered crops")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    floating = [v.detach() for v in model.state_dict().values() if v.is_floating_point()]
    # A single device synchronization validates every post-step tensor; avoid
    # one costly CUDA sync per parameter and never checkpoint a poisoned model.
    if not torch.isfinite(torch.stack(torch._foreach_norm(floating))).all():
        raise FloatingPointError("nonfinite post-step model; abort and resume last clean checkpoint")
    ema.update(model)
    if not torch.isfinite(torch.stack(torch._foreach_norm(
            [v for v in ema.state.values() if v.is_floating_point()]))).all():
        raise FloatingPointError("nonfinite EMA; abort and resume last clean checkpoint")
    optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss, "gradient_norm": float(norm), "lr_multiplier": multiplier,
            "first_crop_index": indices[0], "last_crop_index": indices[-1]}


def _updates(loader, accumulation: int):
    iterator = iter(loader)
    while True:
        try:
            first = next(iterator)
        except StopIteration:
            return
        # A generator avoids holding all64 normalized images in pinned memory.
        def batches(first=first):
            yield first
            for _ in range(accumulation - 1):
                try:
                    yield next(iterator)
                except StopIteration as error:
                    raise RuntimeError("partial effective batch is forbidden") from error
        yield batches()


def run_stage(config: dict, binding: dict, metadata: dict, stage: str, duration: int,
              *, selection: dict | None = None) -> dict:
    from tcmmsline3.target_dino_model import build_model, configure_train_stage
    from tools.evaluate_target_dino import evaluate_scene_set
    output = Path(config["output_dir"])
    stage_dir = output / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    complete_file = stage_dir / "completed.json"
    if complete_file.is_file():
        completed = json.loads(complete_file.read_text())
        if completed.get("binding_sha256") != binding["binding_sha256"] or completed.get("duration_epochs") != duration:
            raise ValueError("completed stage identity does not match request")
        final = stage_dir / ("best_ema.pt" if stage == "selection" else "final_ema.pt")
        if not final.is_file() or sha256(final) != completed["checkpoint_sha256"]:
            raise ValueError("completed stage checkpoint is absent or changed")
        return completed
    ids = metadata["selection_train_ids"] if stage == "selection" else metadata["train_ids"]
    shape = schedule_shape(len(ids))
    last = stage_dir / "last.pt"
    seed_everything(config["seed"])
    # A resumed stage restores every tensor; fresh full stage always reloads
    # original pretraining and a newly initialized target head, never best EMA.
    model = build_model(config, load_pretrained=not last.is_file()).cuda()
    configure_train_stage(model, 0)
    optimizer = _optimizer(model, config)
    ema = TensorEMA(model)
    state = _restore(last, model, optimizer, ema, binding, stage, shape) if last.is_file() else _state(stage, shape)
    if selection is not None:
        expected_selection = canonical_hash(selection)
        if state.get("selection_sha256", expected_selection) != expected_selection:
            raise ValueError("full resume epoch-selection provenance changed")
        state["selection_sha256"] = expected_selection
    if state["epoch"] > duration:
        raise ValueError("checkpoint passed committed stage duration")

    def save():
        validate_progress(state, stage, shape)
        atomic_checkpoint(last, _payload(model, optimizer, ema, state, binding, stage))

    def evaluate_pending():
        epoch = state["epoch"]
        if stage != "selection" or epoch not in CANDIDATE_EPOCHS or epoch in state["evaluated_epochs"] or state["next_index"]:
            return
        _progress(output, stage, state, status="held_evaluation")
        with use_ema(model, ema):
            metrics = evaluate_scene_set(model, config["data_manifest"], metadata["held_ids"],
                                        normalization={"divisor": 255.0, **config["normalization"]},
                                        batch_size=config["eval_batch_size"], workers=config["eval_workers"],
                                        output_dir=stage_dir / "validation" / f"epoch{epoch:03d}", device="cuda",
                                        provenance={"binding_sha256": binding["binding_sha256"],
                                                    "stage": stage, "epoch": epoch,
                                                    "successful_updates": state["updates"],
                                                    "ema_updates": ema.updates, "weights": "EMA"})
        score = float(metrics["miou"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError("held evaluation returned invalid global fixed-class mIoU")
        metrics = {**metrics, "epoch": epoch, "successful_updates": state["updates"],
                   "binding_sha256": binding["binding_sha256"], "stage": stage}
        atomic_json(stage_dir / "validation" / f"epoch{epoch:03d}" / "candidate.json", metrics)
        if state["best_miou"] is None or score > state["best_miou"] or (score == state["best_miou"] and epoch < state["best_epoch"]):
            state["best_epoch"], state["best_miou"] = epoch, score
            atomic_checkpoint(stage_dir / "best_ema.pt", _compact(ema.state, state, binding, stage, "best_held_ema"))
        state["evaluated_epochs"].append(epoch)
        save()
        _progress(output, stage, state, status="held_evaluation_completed", held_miou=score)

    try:
        evaluate_pending()
        while state["epoch"] < duration:
            if _STOP_REQUESTED:
                save()
                _progress(output, stage, state, status="paused_at_optimizer_boundary")
                raise InterruptedError("clean training stop requested")
            epoch = state["epoch"]
            model.train()
            configure_train_stage(model, epoch)
            dataset, loader = _loader(config, stage, epoch, state["next_index"])
            if dataset.samples_per_epoch != shape["samples_per_epoch"]:
                raise ValueError("data sample budget changed")
            _progress(output, stage, state)
            last_log = time.monotonic()
            for batches in _updates(loader, config["accumulation_steps"]):
                before = time.monotonic()
                step = _train_update(model, optimizer, ema, batches, shape, state["updates"], config)
                if step["first_crop_index"] != state["next_index"]:
                    raise RuntimeError("data cursor disagrees with optimizer cursor")
                state["updates"] += 1
                state["ema_updates"] = ema.updates
                state["next_index"] += 64
                end_epoch = state["next_index"] == shape["samples_per_epoch"]
                if end_epoch:
                    state["epoch"] += 1
                    state["next_index"] = 0
                validate_progress(state, stage, shape)
                if state["updates"] % 500 == 0 or end_epoch or _STOP_REQUESTED:
                    save()
                if time.monotonic() - last_log >= 30 or end_epoch:
                    _progress(output, stage, state, step_seconds=time.monotonic() - before, **step)
                    last_log = time.monotonic()
                if _STOP_REQUESTED:
                    break
            del loader, dataset
            if _STOP_REQUESTED:
                _progress(output, stage, state, status="paused_at_optimizer_boundary")
                raise InterruptedError("clean training stop requested")
            if state["epoch"] != epoch + 1 or state["next_index"]:
                raise RuntimeError("data loader ended before committed epoch budget")
            evaluate_pending()
        if stage == "selection":
            if state["evaluated_epochs"] != list(CANDIDATE_EPOCHS) or state["best_epoch"] is None:
                raise RuntimeError("cannot select without every committed candidate evaluation")
            artifact = stage_dir / "best_ema.pt"
        else:
            artifact = stage_dir / "final_ema.pt"
            atomic_checkpoint(artifact, _compact(ema.state, state, binding, stage, "final_full_train_ema"))
        result = {"status": "completed", "stage": stage, "binding_sha256": binding["binding_sha256"],
                  "duration_epochs": duration, "selected_epoch": state["best_epoch"] if stage == "selection" else duration,
                  "best_held_miou": state["best_miou"], "checkpoint": str(artifact.resolve()),
                  "checkpoint_sha256": sha256(artifact), "state": state,
                  "official_test_auto": False, "time_unix": time.time()}
        atomic_json(complete_file, result)
        # This exact, validated rolling file belongs to this new run and is
        # redundant only after a recoverable compact checkpoint + receipt exist.
        if stage == "selection" and last.exists():
            last.unlink()
        _progress(output, stage, state, status="stage_completed")
        return result
    finally:
        del model, optimizer, ema
        gc.collect()
        torch.cuda.empty_cache()


def run_smoke(config: dict, binding: dict, metadata: dict) -> dict:
    from tcmmsline3.target_dino_model import build_model, configure_train_stage
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection").exists() or (output / "full").exists():
        raise RuntimeError("smoke cannot overlap formal training outputs")
    seed_everything(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    model = build_model(config, load_pretrained=True).cuda()
    model.train()
    # Worst-case v3 smoke includes the staged upper backbone and Adam moments.
    configure_train_stage(model, 5)
    optimizer, ema = _optimizer(model, config), TensorEMA(model)
    shape = schedule_shape(len(metadata["selection_train_ids"]))
    dataset, loader = _loader(config, "selection", 0, 0)
    started = time.monotonic()
    update = _train_update(model, optimizer, ema, next(_updates(loader, config["accumulation_steps"])), shape, 0, config)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    # Independent inference at native512 geometry, not a target metric or any
    # validation-based tuning. The smoke model is discarded on process exit.
    with use_ema(model, ema), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        sample = dataset[0]["ms"].unsqueeze(0).cuda()
        logits = model(sample)["logits"]
        if tuple(logits.shape) != (1, metadata["num_classes"], 512, 512) or not torch.isfinite(logits).all():
            raise RuntimeError("smoke native512 inference failed")
    state = _state("smoke", shape)
    state.update(updates=1, ema_updates=1, next_index=64)
    temporary = output / "smoke_roundtrip.pt"
    try:
        atomic_checkpoint(temporary, _payload(model, optimizer, ema, state, binding, "smoke"))
        result = {"status": "passed", "binding_sha256": binding["binding_sha256"],
                  "dataset": config["dataset"], "variant": config["variant"],
                  "effective_batch": 64, "physical_batch": config["batch_size"],
                  "accumulation_steps": config["accumulation_steps"],
                  "worst_case_epoch_zero_based": 5, "updates": 1, "ema_updates": 1,
                  "optimizer_state_entries": len(optimizer.state),
                  "native_logits_shape": list(logits.shape), "update_seconds": elapsed,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                  "checkpoint_roundtrip": "passed", "formal_outputs_written": False,
                  "time_unix": time.time(), **update}
        atomic_json(output / "smoke.json", result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        if temporary.exists():
            temporary.unlink()


def run(config: dict, binding: dict, metadata: dict) -> dict:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "runtime_binding.json"
    if existing.is_file() and json.loads(existing.read_text()) != binding:
        raise ValueError("output directory belongs to another source/config/data identity")
    atomic_json(existing, binding)
    smoke_path = output / "smoke.json"
    if not smoke_path.is_file():
        raise RuntimeError("formal training requires a successful independent smoke")
    smoke = json.loads(smoke_path.read_text())
    if smoke.get("status") != "passed" or smoke.get("binding_sha256") != binding["binding_sha256"]:
        raise ValueError("smoke and formal runtime identities differ")
    selected = run_stage(config, binding, metadata, "selection", 70)
    if _STOP_REQUESTED:
        raise InterruptedError("stop requested before fresh full fit")
    epoch = selected["selected_epoch"]
    if epoch not in CANDIDATE_EPOCHS:
        raise RuntimeError("selected epoch is outside committed candidates")
    final = run_stage(config, binding, metadata, "full", epoch, selection=selected)
    result = {"status": "completed", "binding_sha256": binding["binding_sha256"],
              "selection": selected, "full": final, "official_test_auto": False,
              "message": "Ready for later unified official evaluation; no test inference was launched.",
              "time_unix": time.time()}
    atomic_json(output / "completed.json", result)
    atomic_json(output / "progress.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    torch.set_num_threads(config["torch_threads"])
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("target training requires CUDA with native BF16")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    binding, metadata = build_binding(config)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _signal_handler)
    try:
        if args.smoke:
            run_smoke(config, binding, metadata)
        else:
            run(config, binding, metadata)
    except InterruptedError:
        raise SystemExit(130)
    except BaseException as error:
        output = Path(config["output_dir"])
        atomic_json(output / ("smoke.json" if args.smoke else "failure.json"),
                    {"status": "failed", "binding_sha256": binding["binding_sha256"],
                     "error_type": type(error).__name__, "error": str(error), "time_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
