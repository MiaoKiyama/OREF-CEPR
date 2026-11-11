"""Two-rank, train96-only continuation of the complete clean MS75 graph.

The original trainer and model are immutable dependencies. All stage-union
parameters/buffers, optimizer state and both rank RNG streams are recoverable.
DDP wraps the *objective*, so unused auxiliary outputs cannot fool its graph
traversal. A global update is exactly64 planned crops, split32/32 without pad.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import distributed as dist, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from tools import train_target_dino as engine

SCHEMA = "ms75_sar_joint_train96_ddp_v1"
GPU_UUIDS = ("GPU-64ff02fb-e035-2e17-9e55-961de4dd9f75",
             "GPU-3460188f-cb0e-f065-26d0-348e74ebe26c")
BASE_FINGERPRINT = "1fea3ea364a9cec0995d81fdc26357cc6f42f821a0179cb7a7408b9a776b1426"
FIXED = dict(schema=SCHEMA, seed=1337, epochs=12, updates_per_epoch=480,
             effective_batch=64, world_size=2, warmup_epochs=1,
             checkpoint_epochs=[2, 4, 8, 12], physical_gpus=[4, 5],
             optical_lr=1e-5, sar_lr=1e-4, tail_lr=3e-6, weight_decay=.01,
             lr_warmup_updates=240, ema_decay=.999, gradient_clip=1.,
             sar_aux_weight=.1, anchor_weight=.05, anchor_confidence=.7,
             anchor_temperature=2., auto_full=False, official_test_auto=False)
LOSS_OVERRIDES = dict(cross_entropy_weight=1., dice_weight=.5, lovasz_weight=.25,
                     context_aux_weight=.2, spectral_aux_weight=.2,
                     short_aux_weight=.15, global_aux_weight=.15,
                     family_ce_weight=.2, hierarchy_consistency_weight=.05,
                     family_child_weight=.4, family_contrast_weight=.05,
                     fixed_anchor_no_harm_weight=0., force_main_losses_fp32=True)
OPTICAL_KEYS = ("ms", "ms_short", "ms_short_valid", "ms_global", "ms_global_valid",
                "local_box_short", "local_box_global")


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def subset(model, keys, device="cpu"):
    state = model.state_dict()
    return {key: state[key].detach().to(device).clone() for key in keys}


def apply_delta(model, values):
    state = model.state_dict()
    if set(values) != set(model.mutable_keys()):
        raise ValueError("stage-union delta tensor inventory changed")
    for key, value in values.items():
        if value.shape != state[key].shape or value.dtype != state[key].dtype:
            raise ValueError("delta shape/dtype differs: " + key)
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite delta: " + key)
    state.update(values)
    model.load_state_dict(state, strict=True)


def validate_config(config, *, devices=True):
    for key, value in FIXED.items():
        if config.get(key) != value:
            raise ValueError("fixed joint protocol mismatch: " + key)
    if config.get("gpu_uuids") != list(GPU_UUIDS):
        raise ValueError("GPU4/5 UUID identity changed")
    if config.get("micro_batch") not in (1, 2, 4) or 32 % config["micro_batch"]:
        raise ValueError("physical microbatch must divide each rank's32 crops")
    for key in ("num_workers", "torch_threads", "save_every_updates"):
        if type(config.get(key)) is not int or config[key] < (0 if key == "num_workers" else 1):
            raise ValueError("invalid " + key)
    if config["num_workers"] > 6 or config["torch_threads"] > 2:
        raise ValueError("CPU budget exceeds the fixed per-rank allowance")
    if devices and os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(GPU_UUIDS):
        raise ValueError("CUDA_VISIBLE_DEVICES must bind exactly physical GPU4,5 by UUID")
    for key in ("base_checkpoint", "base_config", "croma_checkpoint", "croma_source", "output_dir", "data"):
        if not config.get(key):
            raise ValueError("missing joint configuration: " + key)
    if config.get("base_config_fingerprint") != BASE_FINGERPRINT:
        raise ValueError("only the authenticated clean96 base is allowed")
    if not isinstance(config.get("files"), dict) or not config["files"]:
        raise ValueError("source and asset SHA bindings are required")
    for key in ("base_checkpoint", "base_config", "croma_checkpoint", "croma_source"):
        if config[key] not in config["files"]:
            raise ValueError("missing file SHA binding: " + key)
    return engine.canonical_hash(config)


def verify_files(config):
    for path, expected in config["files"].items():
        if engine.sha256(path) != expected:
            raise ValueError("bound file changed: " + path)
    if str(Path(__file__).resolve()) not in config["files"]:
        raise ValueError("trainer source itself is not SHA-bound")


def verify_physical_devices(config):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                            check=True, text=True, capture_output=True)
    mapping = {int(parts[0]): parts[1].strip() for line in result.stdout.splitlines()
               if (parts := line.split(",")) and len(parts) == 2}
    if [mapping.get(i) for i in (4, 5)] != config["gpu_uuids"]:
        raise ValueError("physical GPU index/UUID changed")


def load_base_config(config):
    import yaml
    base = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    if "_base_" in base or base["model"].get("architecture") != "ms75_multiscale":
        raise ValueError("base_config must be the resolved original MS75 configuration")
    if (base["data"].get("protocol_stage") != "ms75_clean_fold0_train96"
            or base["train"].get("fixed_anchor_teacher_enabled")
            or base["train"].get("ms75_anchor_checkpoint")):
        raise ValueError("full120/anchor-contaminated base configuration is forbidden")
    return base


def validate_base_payload(payload, config):
    required = dict(checkpoint_schema_version=1, checkpoint_role="ema_successful_optimizer_updates",
                    completion_kind="successful_optimizer_update_budget", budget_complete=True,
                    target_successful_optimizer_updates=192000, global_optimizer_step=192000,
                    ema_n_averaged=192000, config_fingerprint=config["base_config_fingerprint"])
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError("clean96 EMA provenance differs: " + key)
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise ValueError("clean96 EMA has no model state")


def make_optical(config, device="cpu"):
    from tcmmsline3.model import build_model
    base = copy.deepcopy(load_base_config(config))
    for encoder in ("local_encoder", "short_encoder", "context_encoder"):
        spec = base["model"]["ms75"][encoder]
        spec.update(pretrained=False, pretrained_file=None, pretrained_file_sha256=None,
                    prefer_official_url=False)
    model = build_model(base)
    payload = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    validate_base_payload(payload, config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device)


def make_model(config, device="cpu"):
    from tcmmsline3.ms75_sar_joint_model import MS75SARJoint
    return MS75SARJoint(make_optical(config), config["croma_checkpoint"],
                        croma_source=config["croma_source"]).to(device)


def loss_config(config):
    base = copy.deepcopy(load_base_config(config))
    base["loss"].update(LOSS_OVERRIDES)
    return base


class ExactRankSampler(Sampler):
    """One ordered contiguous32 half per update; no padding or repetition."""
    def __init__(self, *, updates, rank, world_size=2, start_update=0, effective_batch=64):
        if (type(rank) is not int or not 0 <= rank < world_size or world_size not in (1, 2)
                or effective_batch != 64 or not 0 <= start_update <= updates):
            raise ValueError("invalid exact rank sampler bounds")
        self.updates, self.rank, self.start = updates, rank, start_update
        self.local_batch = effective_batch // world_size

    def __len__(self):
        return (self.updates - self.start) * self.local_batch

    def __iter__(self):
        for update in range(self.start, self.updates):
            begin = update * 64 + self.rank * self.local_batch
            yield from range(begin, begin + self.local_batch)


def croma_region_targets(labels, optical_valid, *, num_classes=24, sar_size=166, patch=8):
    """Known optical pixel-centre counts in the padded native CROMA cells.

    This is explicitly discrete pixel-centre quadrature, not continuous area
    integration. No label interpolation, adaptive pooling or purity filtering.
    """
    if labels.ndim != 3 or labels.shape[-2:] != (512, 512):
        raise ValueError("native auxiliary supervision requires512 optical labels")
    if optical_valid.shape != (len(labels), 1, 512, 512) or optical_valid.dtype != torch.bool:
        raise ValueError("optical SAR support must be bool B1x512x512")
    if bool(((labels != 255) & ((labels < 0) | (labels >= num_classes))).any()):
        raise ValueError("auxiliary label IDs must be native classes or255")
    padded = math.ceil(sar_size / patch) * patch
    before = (padded - sar_size) // 2
    n = padded // patch
    coordinates = torch.arange(512, device=labels.device, dtype=torch.float64)
    axis = torch.floor((before + (coordinates + .5) * sar_size / 512) / patch).long()
    cells = (axis[:, None] * n + axis[None]).reshape(1, -1).expand(len(labels), -1)
    known = ((labels != 255) & optical_valid[:, 0]).reshape(len(labels), -1)
    combined = cells * num_classes + labels.reshape(len(labels), -1).clamp(0, num_classes - 1)
    counts = torch.zeros((len(labels), n*n*num_classes), device=labels.device, dtype=torch.float32)
    counts.scatter_add_(1, combined, known.float())
    counts = counts.reshape(len(labels), n, n, num_classes).permute(0, 3, 1, 2)
    mass = counts.sum(1, keepdim=True)
    return counts / mass.clamp_min(1), mass


def sar_soft_loss(logits, labels, optical_valid, native_observed):
    targets, mass = croma_region_targets(labels, optical_valid, num_classes=logits.shape[1])
    if logits.shape != targets.shape or native_observed.shape != mass.shape:
        raise ValueError("CROMA auxiliary logit/support lattice mismatch")
    eligible = (mass[:, 0] > 0) & native_observed[:, 0].bool()
    pixel = -(targets * logits.float().log_softmax(1)).sum(1)
    per_image = (pixel * eligible).flatten(1).sum(1) / eligible.flatten(1).sum(1).clamp_min(1)
    return per_image.mean(), eligible.float().mean().detach()


def anchor_loss(student, teacher, labels, *, confidence=.7, temperature=2.):
    """Keep only known pixels where the immutable clean anchor is correct."""
    if student.shape != teacher.shape or labels.shape != (len(student), *student.shape[2:]):
        raise ValueError("anchor shapes differ")
    teacher = teacher.detach().float()
    probability = teacher.softmax(1)
    confidence_value, predicted = probability.max(1)
    protect = (labels != 255) & (predicted == labels) & (confidence_value >= confidence)
    pixel = F.kl_div((student.float()/temperature).log_softmax(1),
                     (teacher/temperature).softmax(1), reduction="none").sum(1) * temperature**2
    per_image = (pixel * protect).flatten(1).sum(1) / protect.flatten(1).sum(1).clamp_min(1)
    return per_image.mean(), protect.float().mean().detach()


def compute_losses(output, batch, teacher, config, base_loss_config):
    from tcmmsline3.losses import segmentation_loss
    labels = batch["label"].long()
    if bool(((labels != 255) & ((labels < 0) | (labels >= 24))).any()):
        raise ValueError("invalid main target")
    # Only per-image normalization is permitted; physical microbatch cannot
    # change Dice, Lovasz, auxiliary-head or no-harm loss weighting.
    main = output["logits"].float().sum() * 0
    tracked = {}
    for i in range(len(labels)):
        target = labels[i:i+1].clone()
        target[target == 255] = -1
        sliced = {}
        for key, value in output.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim >= 3 and value.shape[0] == len(labels):
                value = value[i:i+1]
            if value.is_floating_point() and ("logits" in key or key == "family_contrast_embeddings"):
                value = value.float()
            sliced[key] = value
        if not bool((target >= 0).any()):
            continue
        item, parts = segmentation_loss(sliced, target, batch["sar_valid"][i:i+1, 0],
                                         base_loss_config, phase="joint")
        main = main + item / len(labels)
        for key in ("final_ce", "dice", "lovasz", "family_ce", "family_child",
                    "context_aux", "spectral_aux", "short_aux", "global_aux"):
            tracked[key] = tracked.get(key, item.detach()*0) + parts[key].detach() / len(labels)
    auxiliary, aux_support = sar_soft_loss(output["sar_aux_logits"], labels,
                                           batch["optical_valid"], output["sar_aux_valid"])
    preserve, protect_fraction = anchor_loss(output["logits"], teacher, labels,
                    confidence=config["anchor_confidence"], temperature=config["anchor_temperature"])
    total = main + config["sar_aux_weight"] * auxiliary + config["anchor_weight"] * preserve
    return total, dict(loss=total.detach(), main=main.detach(), sar_aux=auxiliary.detach(),
                      anchor_kl=preserve.detach(), aux_support=aux_support,
                      protected_fraction=protect_fraction, **tracked)


class TrainingObjective(nn.Module):
    def __init__(self, model, config, base_loss_config):
        super().__init__()
        self.model, self.config, self.base_loss_config = model, config, base_loss_config

    def forward(self, batch, teacher):
        from tcmmsline3.ms75_sar_joint_model import JOINT_INPUT_KEYS
        output = self.model(**{key: batch[key] for key in JOINT_INPUT_KEYS})
        with torch.autocast(device_type=batch["ms"].device.type, enabled=False):
            return compute_losses(output, batch, teacher, self.config, self.base_loss_config)


def validate_optimizer_inventory(model, groups):
    parameters = dict(model.named_parameters())
    flat = [parameter for group in groups for parameter in group["params"]]
    if len({id(p) for p in flat}) != len(flat):
        raise ValueError("optimizer repeats a parameter")
    expected = {id(value) for name, value in parameters.items() if name in set(model.mutable_keys())}
    if {id(p) for p in flat} != expected:
        raise ValueError("optimizer omits a stage-union parameter or includes a frozen one")


def stage_for_update(update, config):
    return "warmup" if update < config["warmup_epochs"]*config["updates_per_epoch"] else "joint"


def lr_factor(update, config):
    return engine.lr_multiplier(update, config["epochs"] * config["updates_per_epoch"],
                                config["lr_warmup_updates"])


def capture_rng(rank, device):
    return dict(rank=rank, python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device).cpu())


def restore_rng(state, rank, device):
    if state["rank"] != rank:
        raise ValueError("resume RNG rank differs")
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def make_loader(config, epoch, rank, start_update):
    from tcmmsline3.ms75_sar_joint_data import MS75SARJointDataset
    dataset = MS75SARJointDataset(config, epoch=epoch, start_sample=0)
    if len(dataset) != config["updates_per_epoch"]*64:
        raise ValueError("epoch dataset length differs from the fixed crop budget")
    sampler = ExactRankSampler(updates=config["updates_per_epoch"], rank=rank,
                               world_size=dist.get_world_size(), start_update=start_update)
    generator = torch.Generator().manual_seed(config["seed"] + 1009*epoch + rank)
    kwargs = dict(dataset=dataset, sampler=sampler, batch_size=config["micro_batch"],
                  num_workers=config["num_workers"], pin_memory=True, drop_last=False,
                  generator=generator, worker_init_fn=engine._worker_init)
    if config["num_workers"]:
        kwargs.update(prefetch_factor=2, persistent_workers=False)
    return dataset, DataLoader(**kwargs)


def device_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def gradient_summary(model, optimizer):
    result = {}
    for index, group in enumerate(optimizer.param_groups):
        values = [p.grad.detach().float().square().sum() for p in group["params"] if p.grad is not None]
        result[group.get("group_name", str(index))] = math.sqrt(float(sum(values))) if values else 0.
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("frozen parameter acquired a gradient")
    return result


def train_update(ddp, model, teacher, optimizer, iterator, update, epoch_update, rank, config,
                 *, device, audit=False):
    factor = lr_factor(update, config)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor
    optimizer.zero_grad(set_to_none=True)
    world = dist.get_world_size()
    local_batch = config["effective_batch"] // world
    steps = local_batch // config["micro_batch"]
    seen, accumulated = [], {}
    for micro in range(steps):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise ValueError("dataset ended inside a full64 global update") from exc
        ids = [int(v) for v in batch["index"]]
        if len(ids) != config["micro_batch"]:
            raise ValueError("partial physical microbatch is forbidden")
        seen.extend(ids)
        value = device_batch(batch, device)
        sync = nullcontext() if micro == steps-1 else ddp.no_sync()
        with sync:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                reference = teacher(**{k:value[k] for k in OPTICAL_KEYS})["logits"]
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                loss, parts = ddp(value, reference)
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise FloatingPointError("nonfinite loss on at least one rank")
            # DDP rank averaging plus local normalization yields global64.
            (loss * len(ids)/local_batch).backward()
        for key, item in parts.items():
            accumulated[key] = accumulated.get(key, torch.zeros_like(item)) + item * len(ids)/local_batch
        del value, reference, loss, parts, batch
    expected = list(range(epoch_update*64 + rank*local_batch, epoch_update*64 + (rank+1)*local_batch))
    if seen != expected:
        raise ValueError("rank did not consume its exact ordered planned crop IDs")
    gradients = gradient_summary(model, optimizer) if audit else None
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          config["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    model.project_constrained_parameters_()
    optimizer.zero_grad(set_to_none=True)
    names = sorted(accumulated)
    values = torch.stack([accumulated[k] for k in names])
    dist.all_reduce(values); values /= world
    return dict(zip(names, values.cpu().tolist())) | dict(lr_factor=factor, gradient_norm=float(norm),
                  gradients=gradients, rank_local_crops=len(seen))


@torch.no_grad()
def update_ema(model, keys, ema, decay, updates):
    state = model.state_dict()
    for key in keys:
        value = state[key].detach()
        if updates == 0 or not value.is_floating_point():
            ema[key].copy_(value)
        else:
            ema[key].lerp_(value, 1-decay)


def checkpoint_payload(model, keys, ema, optimizer, config, identity, frozen_hash, updates, rank_rng):
    return dict(schema=SCHEMA, role="joint_recovery", config_sha256=identity,
                base_sha256=config["files"][config["base_checkpoint"]], frozen_hash=frozen_hash,
                updates=updates, stage=stage_for_update(updates, config), mutable_keys=list(keys),
                model=subset(model, keys), ema={k:v.detach().cpu() for k,v in ema.items()},
                optimizer=optimizer.state_dict(), rank_rng=rank_rng, world_size=2)


def validate_recovery(payload, model, config, identity, frozen_hash):
    if (payload.get("schema") != SCHEMA or payload.get("role") != "joint_recovery"
            or payload.get("config_sha256") != identity or payload.get("frozen_hash") != frozen_hash
            or payload.get("base_sha256") != config["files"][config["base_checkpoint"]]
            or payload.get("mutable_keys") != list(model.mutable_keys()) or payload.get("world_size") != 2):
        raise ValueError("recovery identity or stage-union inventory changed")
    updates = payload.get("updates")
    if type(updates) is not int or not 0 <= updates <= config["epochs"]*config["updates_per_epoch"]:
        raise ValueError("invalid recovery update count")
    if payload.get("stage") != stage_for_update(updates, config):
        raise ValueError("recovery stage differs from fixed schedule")
    if [r.get("rank") for r in payload.get("rank_rng", [])] != [0, 1]:
        raise ValueError("recovery must contain both ordered rank RNG streams")


def assert_rank_hashes(value, description):
    hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, value)
    if any(item != hashes[0] for item in hashes):
        raise RuntimeError("DDP rank mismatch: " + description)
    return hashes[0]


def save_recovery(output, model, keys, ema, optimizer, config, identity, frozen_hash, updates, rank, device):
    rank_rng = [None] * 2
    dist.all_gather_object(rank_rng, capture_rng(rank, device))
    if rank == 0:
        engine.atomic_checkpoint(output/"last.pt", checkpoint_payload(model, keys, ema, optimizer,
                         config, identity, frozen_hash, updates, rank_rng))
    dist.barrier()


def no_sar_parity(model, batch):
    from tcmmsline3.ms75_sar_joint_model import JOINT_INPUT_KEYS
    was_training = model.training
    model.eval()
    value = {k:batch[k] for k in JOINT_INPUT_KEYS}
    value["sar_db"] = torch.zeros_like(value["sar_db"])
    value["sar_valid"] = torch.zeros_like(value["sar_valid"])
    value["optical_valid"] = torch.zeros_like(value["optical_valid"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        reference = model.forward_optical(**value)["logits"]
        actual = model(**value)["logits"]
    error = float((reference.float()-actual.float()).abs().max())
    model.train(was_training)
    if error != 0:
        raise RuntimeError("missing SAR does not equal the current optical path")
    return error


def run(config, *, smoke=False, resume=None, benchmark_updates=None,
        smoke_output="smoke.json", smoke_micro_batch=None):
    benchmark = benchmark_updates is not None
    if (benchmark or smoke_micro_batch is not None or smoke_output != "smoke.json") and not smoke:
        raise ValueError("benchmark/runtime overrides are smoke-only")
    if benchmark and (type(benchmark_updates) is not int or not 2 <= benchmark_updates <= 20):
        raise ValueError("benchmark must include2..20 complete global updates")
    if Path(smoke_output).name != smoke_output or not smoke_output.endswith(".json"):
        raise ValueError("smoke output must be a JSON filename inside the bound output directory")
    if benchmark and smoke_output == "smoke.json":
        raise ValueError("benchmarks need a distinct receipt filename")
    identity = validate_config(config, devices=False)
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if (world not in ((1, 2) if benchmark else (2,)) or rank != local_rank
            or not 0 <= rank < world):
        raise ValueError("formal/smoke joint run needs2 ranks; explicit benchmark may use GPU4 alone")
    expected_devices = ",".join(GPU_UUIDS[:world])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_devices:
        raise ValueError("runtime GPU UUIDs do not match the authorized physical4/5 mapping")
    runtime = copy.deepcopy(config)
    if smoke_micro_batch is not None:
        if smoke_micro_batch not in (1, 2, 4):
            raise ValueError("smoke microbatch must be1,2,4")
        runtime["micro_batch"] = smoke_micro_batch
    if config.get("cpu_affinity") and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(config["cpu_affinity"][rank]))
    torch.set_num_threads(config["torch_threads"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        _run_initialized(runtime, identity, rank, device, smoke, resume,
                         benchmark_updates=benchmark_updates, smoke_output=smoke_output,
                         formal_compatible=(not benchmark and smoke_micro_batch in (None, config["micro_batch"])))
    finally:
        dist.destroy_process_group()


def _run_initialized(config, identity, rank, device, smoke, resume, *,
                     benchmark_updates=None, smoke_output="smoke.json", formal_compatible=True):
    output = Path(config["output_dir"])
    if rank == 0:
        verify_physical_devices(config); verify_files(config)
        output.mkdir(parents=True, exist_ok=True)
        if (output/"completed.json").exists():
            raise FileExistsError("formal joint training is already completed")
        if smoke and ((output/"last.pt").exists() or resume):
            raise ValueError("smoke cannot alter an existing formal recovery")
        if not smoke:
            receipt = json.loads((output/"smoke.json").read_text())
            if (receipt.get("status") != "passed" or receipt.get("config_sha256") != identity
                    or not receipt.get("formal_compatible") or receipt.get("world_size") != 2
                    or receipt.get("micro_batch") != config["micro_batch"]):
                raise ValueError("formal training requires this config's two-stage DDP smoke")
            if not resume and (output/"last.pt").exists():
                raise FileExistsError("existing recovery requires explicit --resume")
    dist.barrier()
    engine.seed_everything(config["seed"])
    model = make_model(config, device)
    teacher = make_optical(config, device).requires_grad_(False).eval()
    keys = tuple(model.mutable_keys())
    frozen_keys = tuple(k for k in model.state_dict() if k not in set(keys))
    frozen_hash = state_hash(subset(model, frozen_keys))
    teacher_hash = state_hash(teacher.state_dict())
    groups = model.parameter_groups(optical_lr=config["optical_lr"], sar_lr=config["sar_lr"],
                  tail_lr=config["tail_lr"], weight_decay=config["weight_decay"])
    validate_optimizer_inventory(model, groups)
    for group in groups:
        group["initial_lr"] = group["lr"]
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)
    ema = subset(model, keys, device)
    completed, saved_rng = 0, None
    if resume:
        path = output/"last.pt" if resume == "last" else Path(resume)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_recovery(payload, model, config, identity, frozen_hash)
        apply_delta(model, payload["ema"]); apply_delta(model, payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        ema = {k:v.to(device) for k,v in payload["ema"].items()}
        completed, saved_rng = payload["updates"], payload["rank_rng"][rank]
        del payload
    objective = TrainingObjective(model, config, loss_config(config))
    ddp, previous_stage = None, None
    started, initial_update = time.monotonic(), completed
    target = (benchmark_updates or 2) if smoke else config["epochs"]*config["updates_per_epoch"]
    if completed == target and resume:
        # Re-finalize a finished last.pt only; no extra optimizer update.
        pass
    engine.seed_everything(config["seed"] + 100000 + rank)
    if saved_rng is not None:
        restore_rng(saved_rng, rank, device)
    audits, candidates, update_seconds = [], [], []
    dataset, iterator, loaded_epoch = None, None, None
    torch.cuda.reset_peak_memory_stats(device)
    while completed < target:
        epoch, epoch_update = divmod(completed, config["updates_per_epoch"])
        stage = ("joint" if benchmark_updates is not None else
                 (("warmup" if completed == 0 else "joint") if smoke else stage_for_update(completed, config)))
        if stage != previous_stage:
            if ddp is not None:
                del ddp
            model.set_stage(stage); model.train()
            ddp = DDP(objective, device_ids=[device.index], output_device=device.index,
                      find_unused_parameters=True, broadcast_buffers=False)
            previous_stage = stage
        if epoch != loaded_epoch:
            if dataset is not None:
                dataset.close()
            dataset, loader = make_loader(config, epoch, rank, epoch_update)
            plan_audit = dataset.audit()
            assert_rank_hashes(plan_audit["plan_sha256"], "epoch sampling plan")
            if rank == 0:
                plan_path = output/"plans"/f"epoch{epoch+1:03d}.json"
                if plan_path.exists():
                    if json.loads(plan_path.read_text()) != plan_audit:
                        raise ValueError("epoch sampling audit changed on resume")
                else:
                    engine.atomic_json(plan_path, plan_audit)
            iterator = iter(loader)
            loaded_epoch = epoch
        if smoke and completed == 0:
            # This independent iterator has its own generator; the real ordered
            # training iterator is untouched and all augmentations are keyed.
            probe = device_batch(next(iter(loader)), device)
            no_sar_parity(model, probe)
            from tcmmsline3.ms75_sar_joint_model import JOINT_INPUT_KEYS
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                baseline = teacher(**{k:probe[k] for k in OPTICAL_KEYS})["logits"]
                initial = model(**{k:probe[k] for k in JOINT_INPUT_KEYS})["logits"]
            if not torch.equal(baseline, initial):
                raise RuntimeError("zero-initialized joint model differs from immutable optical EMA")
            model.train()
        audited = smoke or completed in (0, 1, 480, 481)
        update_started = time.monotonic()
        metrics = train_update(ddp, model, teacher, optimizer, iterator, completed, epoch_update,
                               rank, config, device=device, audit=audited)
        update_ema(model, keys, ema, config["ema_decay"], completed)
        if smoke:
            torch.cuda.synchronize(device)
            duration = torch.tensor(time.monotonic()-update_started, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            update_seconds.append(float(duration))
        completed += 1
        if audited:
            audits.append(dict(update=completed, stage=stage, gradients=metrics["gradients"]))
        if rank == 0 and (smoke or completed % 10 == 0 or completed == initial_update+1):
            row = dict(status="smoke" if smoke else "training", updates=completed, total_updates=target,
                       epoch=(completed-1)//480+1, stage=stage, config_sha256=identity,
                       seconds_per_update=(time.monotonic()-started)/(completed-initial_update),
                       time_unix=time.time(), pid=os.getpid(), **metrics)
            engine.atomic_json(output/(Path(smoke_output).stem+"_progress.json" if smoke else "progress.json"), row)
            if not smoke:
                with (output/"metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
        if not smoke and completed % 480 == 0 and completed//480 in config["checkpoint_epochs"]:
            epoch_number = completed//480
            if rank == 0:
                path = output/"candidates"/f"epoch{epoch_number:03d}_ema_delta.pt"
                engine.atomic_checkpoint(path, dict(schema=SCHEMA, role="candidate_ema_delta", epoch=epoch_number,
                    updates=completed, config_sha256=identity, frozen_hash=frozen_hash,
                    base_sha256=config["files"][config["base_checkpoint"]], mutable_keys=list(keys),
                    state_dict={k:v.detach().cpu() for k,v in ema.items()}))
                candidates.append(dict(epoch=epoch_number, path=str(path), sha256=engine.sha256(path)))
            dist.barrier()
        if not smoke and (completed % config["save_every_updates"] == 0 or completed == target):
            save_recovery(output, model, keys, ema, optimizer, config, identity, frozen_hash, completed, rank, device)
    if dataset is not None:
        dataset.close()
    if state_hash(subset(model, frozen_keys)) != frozen_hash or state_hash(teacher.state_dict()) != teacher_hash:
        raise RuntimeError("immutable optical/SAR state or teacher changed")
    raw_digest = assert_rank_hashes(state_hash(subset(model, keys)), "all mutable parameters/buffers")
    ema_digest = assert_rank_hashes(state_hash(ema), "all EMA parameters/buffers")
    if smoke:
        no_sar_parity(model, probe)
        del probe
        # A clean, independently reconstructed CPU base is used for loading.
        if rank == 0:
            path = output/"smoke"/(Path(smoke_output).stem+"_delta.pt")
            engine.atomic_checkpoint(path, dict(schema=SCHEMA, role="smoke_delta",
                config_sha256=identity, updates=target, world_size=dist.get_world_size(),
                micro_batch=config["micro_batch"], base_sha256=config["files"][config["base_checkpoint"]],
                frozen_hash=frozen_hash, mutable_keys=list(keys), state_dict=subset(model, keys)))
            fresh = make_model(config)
            restored = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
            apply_delta(fresh, restored)
            if (state_hash(subset(fresh, keys)) != raw_digest
                    or state_hash(subset(fresh, frozen_keys)) != frozen_hash):
                raise RuntimeError("fresh-model strict delta roundtrip failed")
            delta_sha = engine.sha256(path)
            if benchmark_updates is not None or not formal_compatible:
                path.unlink()
            groups_by_stage = [record["gradients"] for record in audits]
            if len(groups_by_stage) < 2 or not all(any(v > 0 for v in row.values()) for row in groups_by_stage):
                raise RuntimeError("two-stage smoke has no actual learning gradients")
            for row in audits:
                sums = {prefix: sum(v for k,v in row["gradients"].items() if k.startswith(prefix))
                        for prefix in ("optical_heads", "optical_tail", "new_modules", "sar_tail")}
                if (sums["new_modules"] <= 0 or sums["sar_tail"] <= 0
                        or (row["stage"] == "warmup" and (sums["optical_heads"] != 0 or sums["optical_tail"] != 0))
                        or (row["stage"] == "joint" and (sums["optical_heads"] <= 0 or sums["optical_tail"] <= 0))):
                    raise RuntimeError("required stage-specific module gradients are absent")
            engine.atomic_json(output/smoke_output, dict(status="passed", config_sha256=identity,
                updates=target, stages=[row["stage"] for row in audits], raw_sha256=raw_digest, ema_sha256=ema_digest,
                world_size=dist.get_world_size(), micro_batch=config["micro_batch"],
                formal_compatible=formal_compatible, benchmark=benchmark_updates is not None,
                update_seconds=update_seconds,
                steady_seconds_per_update=sum(update_seconds[1:])/len(update_seconds[1:]),
                frozen_hash=frozen_hash, teacher_unchanged=True, rank_states_equal=True,
                strict_roundtrip=True, missing_sar_current_optical_exact=True,
                initial_clean_optical_exact=True, gradients=audits,
                smoke_delta=str(path) if path.exists() else None,
                smoke_delta_sha256=delta_sha if path.exists() else None,
                elapsed_seconds=time.monotonic()-started,
                peak_gib=torch.cuda.max_memory_allocated(device)/1024**3))
    elif rank == 0:
        # Resume can start after earlier candidates; reconstruct the manifest
        # from all fixed filenames rather than dropping the previous epochs.
        candidates = []
        for epoch_number in config["checkpoint_epochs"]:
            path = output/"candidates"/f"epoch{epoch_number:03d}_ema_delta.pt"
            if not path.is_file():
                raise FileNotFoundError("missing precommitted candidate: " + str(path))
            candidates.append(dict(epoch=epoch_number, path=str(path), sha256=engine.sha256(path)))
        verify_files(config)
        receipt = dict(status="training_completed_evaluation_pending", schema=SCHEMA, config_sha256=identity,
                       updates=completed, candidate_checkpoints=candidates, frozen_hash=frozen_hash,
                       raw_sha256=raw_digest, ema_sha256=ema_digest, rank_states_equal=True,
                       teacher_unchanged=True, elapsed_this_run_seconds=time.monotonic()-started,
                       auto_full=False, official_test_auto=False)
        engine.atomic_json(output/"completed.json", receipt)
        engine.atomic_json(output/"progress.json", receipt)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", nargs="?", const="last")
    parser.add_argument("--benchmark-updates", type=int)
    parser.add_argument("--smoke-output", default="smoke.json")
    parser.add_argument("--smoke-micro-batch", type=int)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), smoke=args.smoke, resume=args.resume,
        benchmark_updates=args.benchmark_updates, smoke_output=args.smoke_output,
        smoke_micro_batch=args.smoke_micro_batch)


if __name__ == "__main__":
    main()
