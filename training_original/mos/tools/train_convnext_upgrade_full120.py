"""Isolated two-rank full120 context/pyramid/paired ConvNeXt SAR training.

DDP wraps the objective; each global64 is split evenly without padding. All
stage-union state, optimizer and both rank RNG streams are recoverable. The
MS75 baseline and DINO pair are evaluation assets only and never loaded here.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
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
from torch.utils.data import DataLoader
from tools import train_target_dino as engine
from tools.train_ms75_sar_joint import (ExactRankSampler, apply_delta as _apply_delta, assert_rank_hashes,
    capture_rng, device_batch, gradient_summary, restore_rng, sar_soft_loss,
    state_hash, subset, update_ema, validate_optimizer_inventory)

SCHEMA = "convnext_upgrade_full120_ddp_v1_20261007"
GPU_UUID_BY_INDEX = {
    2: "GPU-25ab91ae-82b2-a99e-8d13-48ba3bfec46f",
    3: "GPU-183f3cce-cb37-5f1c-49dc-f1db79331e11",
    4: "GPU-64ff02fb-e035-2e17-9e55-961de4dd9f75",
    5: "GPU-3460188f-cb0e-f065-26d0-348e74ebe26c",
    6: "GPU-c8faf4b1-5638-90c6-fd97-fcc8eaa3ccbb",
    7: "GPU-aaf2e373-ee45-4aad-5907-1c1ec8c4cf8d",
}
VARIANT_GPUS = {"context": [2, 3], "pyramid": [4, 5], "paired": [6, 7]}
CONVNEXT_SHA256 = "61fa432da5aa0e71bc8e49ce4179f4a4d1d95e4c084a81705c476316bfa3eb33"
SMOKE_UPDATES = 4
FIXED = dict(schema=SCHEMA, seed=1337, epochs=30, updates_per_epoch=600,
    effective_batch=64, world_size=2, warmup_epochs=2,
    checkpoint_epochs=[4, 6]+list(range(8,21))+[22, 24, 27, 30],
    optical_lr=1e-5, sar_lr=1e-4, tail_lr=3e-6, weight_decay=.01,
    lr_warmup_updates=600, scheduler_epochs=30, ema_decay=.999, gradient_clip=1.,
    optical_aux_weight=.2, spectral_aux_weight=.1, sar_aux_weight=.1,
    auto_full=False, official_test_auto=False)


def validate_config(config, *, devices=True):
    for key, value in FIXED.items():
        if config.get(key) != value:
            raise ValueError("fixed ConvNeXt upgrade protocol mismatch: " + key)
    if config.get("variant") not in VARIANT_GPUS:
        raise ValueError("expected context, pyramid or paired variant")
    indices = config.get("physical_gpus")
    if indices != VARIANT_GPUS[config["variant"]]:
        raise ValueError("physical GPU pair differs from this variant's authorization")
    if config.get("gpu_uuids") != [GPU_UUID_BY_INDEX[i] for i in indices]:
        raise ValueError("physical GPU UUID identity changed")
    if config.get("micro_batch") not in (1, 2, 4, 8):
        raise ValueError("microbatch must divide each rank32")
    for key, upper, minimum in (("num_workers",3,0),("torch_threads",2,1)):
        if type(config.get(key)) is not int or not minimum <= config[key] <= upper:
            raise ValueError("invalid CPU allowance: " + key)
    if type(config.get("save_every_updates")) is not int or config["save_every_updates"] < 1:
        raise ValueError("invalid recovery interval")
    if devices and os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(config["gpu_uuids"]):
        raise ValueError("CUDA_VISIBLE_DEVICES must bind the configured physical pair by UUID")
    for key in ("convnext_checkpoint","convnext_source","croma_checkpoint","croma_source","output_dir","data"):
        if not config.get(key):
            raise ValueError("missing upgrade configuration: " + key)
    for key in ("checkpoint_dir", "candidate_dir"):
        if key in config and (not isinstance(config[key], str) or not config[key]):
            raise ValueError("storage override must be a nonempty path: " + key)
    if config.get("convnext_checkpoint_sha256") != CONVNEXT_SHA256:
        raise ValueError("only the authenticated DINOv3 ConvNeXt-L pretraining is allowed")
    files = config.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source, asset and data SHA bindings are required")
    for key in ("convnext_checkpoint","convnext_source","croma_checkpoint","croma_source"):
        if config[key] not in files:
            raise ValueError("missing asset/source SHA binding: " + key)
    if (files[config["convnext_checkpoint"]] != CONVNEXT_SHA256
            or files[config["convnext_source"]] != config.get("convnext_source_sha256")):
        raise ValueError("ConvNeXt source or pretraining binding differs")
    if (config["data"].get("seed") != config["seed"]
            or config["data"].get("updates_per_epoch") != config["updates_per_epoch"]):
        raise ValueError("sampling seed/budget differs from training protocol")
    for key in ("raw_manifest","views_cache","boundary_support"):
        if not config["data"].get(key) or config["data"][key] not in files:
            raise ValueError("missing data SHA binding: " + key)
    return engine.canonical_hash(config)


def pretraining_identity(config):
    return {key:dict(path=config[key],sha256=config["files"][config[key]])
            for key in ("convnext_checkpoint","convnext_source","croma_checkpoint","croma_source")}


def verify_files(config):
    for path, expected in config["files"].items():
        if engine.sha256(path) != expected:
            raise ValueError("bound file changed: " + path)
    for relative in ("tools/train_convnext_upgrade_full120.py", "tcmmsline3/convnext_sar_full120_data.py", "tcmmsline3/convnext_sar_model.py",
                     "tools/train_ms75_sar_joint.py", "tcmmsline3/ms75_sar_joint_data.py"):
        if str(ROOT/relative) not in config["files"]:
            raise ValueError("training dependency itself is not SHA-bound: " + relative)
    module = {"context": "convnext_context_model", "pyramid": "convnext_pyramid_model",
              "paired": "convnext_paired_model"}[config["variant"]]
    required = ["tcmmsline3/" + module + ".py"]
    if config["variant"] == "context":
        required.append("tcmmsline3/convnext_upgrade_data.py")
    for relative in required:
        if str(ROOT / relative) not in config["files"]:
            raise ValueError("variant dependency itself is not SHA-bound: " + relative)


def verify_physical_devices(config):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                            check=True, text=True, capture_output=True)
    mapping = {int(parts[0]):parts[1].strip() for line in result.stdout.splitlines()
               if (parts := line.split(",")) and len(parts) == 2}
    if [mapping.get(i) for i in config["physical_gpus"]] != config["gpu_uuids"]:
        raise ValueError("physical GPU index/UUID changed")


def make_model(config, device="cpu"):
    if config["variant"] == "context":
        from tcmmsline3.convnext_context_model import ConvNeXtContextSAR as Model
    elif config["variant"] == "pyramid":
        from tcmmsline3.convnext_pyramid_model import ConvNeXtPyramidSAR as Model
    elif config["variant"] == "paired":
        from tcmmsline3.convnext_paired_model import ConvNeXtPairedSAR as Model
    else:
        raise ValueError("unknown ConvNeXt upgrade variant")
    return Model(config, load_pretrained=True).to(device)


def device_batch(batch,device):
    # Copy only actual model inputs; the context variant owns a separate native
    # context view and does not consume the legacy short/global views.
    needed=('ms','sar_db','sar_valid','optical_valid','label','index',
            'ms_context','context_valid','local_box_context')
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items() if k in needed}


def validate_state_aliases(model, values):
    """Official norm/norms.3 aliases must never load contradictory EMA values."""
    aliases = {}
    for name, tensor in model.state_dict(keep_vars=True).items():
        aliases.setdefault(id(tensor), []).append(name)
    for names in aliases.values():
        included = [name for name in names if name in values]
        if included and len(included) != len(names):
            raise ValueError("delta omits a shared-state alias: " + str(names))
        if len(included) > 1 and any(not torch.equal(values[included[0]], values[name]) for name in included[1:]):
            raise ValueError("delta values conflict for shared-state aliases: " + str(names))


def apply_delta(model, values):
    validate_state_aliases(model, values)
    _apply_delta(model, values)


def stage_for_update(update, config):
    return "optical_warmup" if update < config["warmup_epochs"]*config["updates_per_epoch"] else "joint"


def lr_factor(update, config):
    return engine.lr_multiplier(update, config["scheduler_epochs"]*config["updates_per_epoch"], config["lr_warmup_updates"])


def compute_losses(output, batch, config, stage):
    """Own full-domain GT objective, independent of teacher probabilities/SAR masks."""
    from tcmmsline3.losses import soft_dice_loss, lovasz_softmax_loss
    if stage not in ("optical_warmup", "joint"):
        raise ValueError("unknown training stage")
    logits, labels = output["logits"].float(), batch["label"].long()
    if logits.shape != (len(labels),24,*labels.shape[-2:]) or labels.ndim != 3:
        raise ValueError("main logits/GT must share the native24 image geometry")
    if bool(((labels != 255) & ((labels < 0) | (labels >= 24))).any()):
        raise ValueError("invalid main target")
    zero = logits.sum()*0.
    pieces = {name:zero for name in ("main_ce","main_dice","main_lovasz","optical_aux","spectral_aux")}
    for i in range(len(labels)):
        target = labels[i:i+1].clone()
        target[target == 255] = -1
        if not bool((target >= 0).any()):
            continue
        own = logits[i:i+1]
        pieces["main_ce"] = pieces["main_ce"] + F.cross_entropy(own,target,ignore_index=-1)/len(labels)
        pieces["main_dice"] = pieces["main_dice"] + soft_dice_loss(own,target,24)/len(labels)
        pieces["main_lovasz"] = pieces["main_lovasz"] + lovasz_softmax_loss(own,target)/len(labels)
        for name in ("optical_aux", "spectral_aux"):
            if name+"_logits" in output:
                auxiliary = output[name+"_logits"][i:i+1].float()
                if auxiliary.shape[:2] != (1,24):
                    raise ValueError("auxiliary output must predict native24 classes")
                if auxiliary.shape[-2:] != target.shape[-2:]:
                    auxiliary = F.interpolate(auxiliary,size=target.shape[-2:],mode="bilinear",align_corners=False)
                pieces[name] = pieces[name] + F.cross_entropy(auxiliary,target,ignore_index=-1)/len(labels)
    main = pieces["main_ce"] + .5*pieces["main_dice"] + .25*pieces["main_lovasz"]
    auxiliary, support = zero, zero.detach()
    if stage == "joint":
        auxiliary, support = sar_soft_loss(output["sar_aux_logits"],labels,batch["optical_valid"],output["sar_aux_valid"])
    total = (main + config["optical_aux_weight"]*pieces["optical_aux"]
             + config["spectral_aux_weight"]*pieces["spectral_aux"] + config["sar_aux_weight"]*auxiliary)
    return total, {**{name:value.detach() for name,value in pieces.items()},
        "loss":total.detach(),"main":main.detach(),"sar_aux":auxiliary.detach(),"sar_aux_support":support.detach()}


class TrainingObjective(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model, self.config = model, config

    def forward(self, batch):
        paired = self.config["variant"] == "paired"
        output = self.model.forward_paired_batch(batch) if paired else self.model.forward_batch(batch)
        with torch.autocast(device_type=batch["ms"].device.type, enabled=False):
            if paired:
                return compute_paired_losses(output, batch, self.config, self.model.stage)
            return compute_losses(output, batch, self.config, self.model.stage)


def compute_paired_losses(output, batch, config, stage):
    """Off GT owns optical gradients; on GT owns only SAR gradients.

    Missing-SAR images occupy their original batch slot but add no on loss.
    This keeps exact global-batch weighting during microbatch accumulation.
    """
    off_loss, off_parts = compute_losses(output["off"], batch, config, "optical_warmup")
    on_loss = off_loss.detach().new_zeros(())
    on_parts = {}
    if stage == "joint":
        if output["on"] is None:
            raise ValueError("joint paired training requires the SAR-on branch")
        present = output["sar_present"]
        if present.dtype != torch.bool or present.shape != (len(batch["label"]),):
            raise ValueError("paired SAR presence must be a per-image boolean")
        on_batch = dict(batch)
        on_batch["label"] = torch.where(present[:, None, None], batch["label"],
                                        torch.full_like(batch["label"], 255))
        on_config = dict(config, optical_aux_weight=0., spectral_aux_weight=0.)
        on_loss, on_parts = compute_losses(output["on"], on_batch, on_config, "joint")
    elif stage != "optical_warmup":
        raise ValueError("unknown paired training stage")
    total = off_loss + on_loss
    parts = {"off_" + k: v for k, v in off_parts.items()}
    parts.update({"on_" + k: v for k, v in on_parts.items()})
    parts.update(loss=total.detach(), optical_off_loss=off_loss.detach(),
                 sar_on_loss=on_loss.detach(),
                 paired_present_fraction=output["sar_present"].float().mean().detach())
    return total, parts


def clip_parameter_groups(model, config):
    if config["variant"] != "paired":
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                              config["gradient_clip"], error_if_nonfinite=True)
        return {"gradient_norm": float(norm)}
    groups = model.gradient_parameter_sets()
    norms = {}
    for name, parameters in groups.items():
        active = [p for p in parameters if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(active, config["gradient_clip"], error_if_nonfinite=True)
        norms[name + "_gradient_norm"] = float(norm)
    norms["gradient_norm"] = math.hypot(norms["optical_gradient_norm"], norms["sar_gradient_norm"])
    return norms


def validate_stage_gradients(stage, gradients, *, require_sar=True):
    totals = {prefix:sum(value for key,value in gradients.items() if key.startswith(prefix))
              for prefix in ("optical_backbone","optical_new","sar_new","sar_tail")}
    if any(not math.isfinite(value) for value in totals.values()):
        raise FloatingPointError("nonfinite module gradient")
    if totals["optical_new"] <= 0:
        raise RuntimeError("optical decoder/spectral modules received no learning gradient")
    if stage == "optical_warmup":
        if any(totals[name] != 0 for name in ("optical_backbone","sar_new","sar_tail")):
            raise RuntimeError("a frozen warmup branch acquired gradients")
    elif stage == "joint":
        required = ("optical_backbone", "optical_new", "sar_new", "sar_tail") if require_sar else ("optical_backbone", "optical_new")
        if any(totals[name] <= 0 for name in required):
            raise RuntimeError("required joint branch has no learning gradient: " + str(totals))
    else:
        raise ValueError("unknown gradient audit stage")
    return totals


def make_loader(config, epoch, rank, start_update):
    from tcmmsline3.convnext_sar_full120_data import Full120Dataset, Full120RankSampler
    if config["variant"] == "context":
        from tcmmsline3.convnext_upgrade_data import UpgradeFull120Dataset
        dataset = UpgradeFull120Dataset(config, epoch=epoch, start_sample=0)
    else:
        dataset = Full120Dataset(config, epoch=epoch, start_sample=0)
    if len(dataset) != config["updates_per_epoch"]*64:
        raise ValueError("epoch dataset length differs from the fixed crop budget")
    sampler = Full120RankSampler(updates=config["updates_per_epoch"], rank=rank,
                               world_size=dist.get_world_size(), start_update=start_update)
    generator = torch.Generator().manual_seed(config["seed"] + 1009*epoch + rank)
    kwargs = dict(dataset=dataset, sampler=sampler, batch_size=config["micro_batch"],
                  num_workers=config["num_workers"], pin_memory=True, drop_last=False,
                  generator=generator, worker_init_fn=engine._worker_init)
    if config["num_workers"]:
        kwargs.update(prefetch_factor=2, persistent_workers=False)
    return dataset, DataLoader(**kwargs)


def train_update(ddp, model, optimizer, iterator, update, epoch_update, rank, config,
                 *, device, audit=False):
    factor = lr_factor(update, config)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor
    optimizer.zero_grad(set_to_none=True)
    world = dist.get_world_size()
    local_batch = config["effective_batch"] // world
    steps = local_batch // config["micro_batch"]
    seen, accumulated = [], {}
    sar_known_pixels = torch.zeros((), device=device, dtype=torch.int64) if audit else None
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
        if audit:
            sar_known_pixels += (value["optical_valid"][:, 0] & (value["label"] != 255)).sum()
        sync = nullcontext() if micro == steps-1 else ddp.no_sync()
        with sync:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                loss, parts = ddp(value)
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise FloatingPointError("nonfinite loss on at least one rank")
            # DDP rank averaging plus local normalization yields global64.
            (loss * len(ids)/local_batch).backward()
        for key, item in parts.items():
            accumulated[key] = accumulated.get(key, torch.zeros_like(item)) + item * len(ids)/local_batch
        del value, loss, parts, batch
    expected = list(range(epoch_update*64 + rank*local_batch, epoch_update*64 + (rank+1)*local_batch))
    if seen != expected:
        raise ValueError("rank did not consume its exact ordered planned crop IDs")
    gradients = gradient_summary(model, optimizer) if audit else None
    if audit:
        dist.all_reduce(sar_known_pixels)
        validate_stage_gradients(model.stage, gradients, require_sar=bool(sar_known_pixels > 0))
    norms = clip_parameter_groups(model, config)
    optimizer.step()
    if hasattr(model,"project_constrained_parameters_"):
        model.project_constrained_parameters_()
    optimizer.zero_grad(set_to_none=True)
    names = sorted(accumulated)
    values = torch.stack([accumulated[k] for k in names])
    dist.all_reduce(values); values /= world
    return dict(zip(names, values.cpu().tolist())) | norms | dict(lr_factor=factor,
                  gradients=gradients, rank_local_crops=len(seen),
                  audit_sar_known_pixels=int(sar_known_pixels) if audit else None)


def checkpoint_payload(model, keys, ema, optimizer, config, identity, frozen_hash, updates, rank_rng):
    return dict(schema=SCHEMA,role="convnext_upgrade_recovery",variant=config["variant"],config_sha256=identity,
        pretraining_identity=pretraining_identity(config),frozen_hash=frozen_hash,
        updates=updates,stage=stage_for_update(updates,config),mutable_keys=list(keys),
        model=subset(model,keys),ema={k:v.detach().cpu() for k,v in ema.items()},
        optimizer=optimizer.state_dict(),rank_rng=rank_rng,world_size=dist.get_world_size())


def validate_recovery(payload, model, config, identity, frozen_hash):
    if (payload.get("schema") != SCHEMA or payload.get("role") != "convnext_upgrade_recovery"
            or payload.get("variant") != config["variant"]
            or payload.get("config_sha256") != identity or payload.get("frozen_hash") != frozen_hash
            or payload.get("pretraining_identity") != pretraining_identity(config)
            or payload.get("mutable_keys") != list(model.mutable_keys()) or payload.get("world_size") != dist.get_world_size()):
        raise ValueError("recovery identity, pretraining or stage-union inventory changed")
    updates = payload.get("updates")
    if type(updates) is not int or not 0 <= updates <= config["epochs"]*config["updates_per_epoch"]:
        raise ValueError("invalid recovery update count")
    if payload.get("stage") != stage_for_update(updates,config):
        raise ValueError("recovery stage differs from fixed schedule")
    if [row.get("rank") for row in payload.get("rank_rng",[])] != list(range(dist.get_world_size())):
        raise ValueError("recovery requires both ordered rank RNG streams")


def save_recovery(output, model, keys, ema, optimizer, config, identity, frozen_hash, updates, rank, device):
    rank_rng = [None]*dist.get_world_size()
    dist.all_gather_object(rank_rng,capture_rng(rank,device))
    if rank == 0:
        write_rolling_recovery(output, checkpoint_payload(model,keys,ema,optimizer,
            config,identity,frozen_hash,updates,rank_rng))
    dist.barrier()


def write_rolling_recovery(output, payload):
    """Retain latest and previous, with one validated atomic-write temporary.

    A same-directory hard link preserves the old last without copying a third
    multi-GB recovery. Failure during the new save leaves both links readable.
    This only rotates this run's two owned recovery names.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    last, previous = output / "last.pt", output / "previous.pt"
    if last.exists():
        link = output / ("previous.pt." + str(os.getpid()) + ".link.tmp")
        try:
            os.link(last, link)
            os.replace(link, previous)
        finally:
            if link.exists():
                link.unlink()
    engine.atomic_checkpoint(last, payload)


def smoke_recovery_roundtrip(output, model, keys, ema, optimizer, config, identity, frozen_hash, rank, device):
    """Exercise the same serialized model/EMA/optimizer/RNG path used by resume."""
    from tools.sar_oct01_extension import tree_hash
    folder = output/('smoke_recovery_'+identity[:12])
    if rank == 0:
        folder.mkdir(exist_ok=False)
    dist.barrier()
    before = dict(raw=state_hash(subset(model,keys)), ema=state_hash(ema),
        optimizer=tree_hash(optimizer.state_dict()), rng=tree_hash(capture_rng(rank,device)))
    save_recovery(folder,model,keys,ema,optimizer,config,identity,frozen_hash,3,rank,device)
    payload = torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    validate_recovery(payload,model,config,identity,frozen_hash)
    apply_delta(model,payload['ema']); apply_delta(model,payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    ema.clear(); ema.update({k:v.to(device) for k,v in payload['ema'].items()})
    engine.seed_everything(9857+rank)
    restore_rng(payload['rank_rng'][rank],rank,device)
    after = dict(raw=state_hash(subset(model,keys)), ema=state_hash(ema),
        optimizer=tree_hash(optimizer.state_dict()), rng=tree_hash(capture_rng(rank,device)))
    if before != after:
        raise RuntimeError('Full120 recovery roundtrip changed raw/EMA/optimizer/RNG')
    return dict(status='passed',rank=rank,world_size=dist.get_world_size(),step=3,
                before=before,after=after,recovery_sha256=engine.sha256(folder/'last.pt'))


def check_ddp_global_mean(device, micro):
    """Actual NCCL/no_sync accumulation must equal the full64 reference gradient."""
    rank,world=dist.get_rank(),dist.get_world_size();local=64//world
    m=nn.Linear(1,1,bias=False,device=device)
    with torch.no_grad():m.weight.fill_(.25)
    d=DDP(m,device_ids=[device.index],broadcast_buffers=False)
    x=torch.arange(64,device=device,dtype=torch.float32).reshape(64,1)/64
    target=2*x
    for j in range(local//micro):
        ix=slice(rank*local+j*micro,rank*local+(j+1)*micro)
        with (nullcontext() if j==local//micro-1 else d.no_sync()):
            ((d(x[ix])-target[ix]).square().mean()*micro/local).backward()
    expected=(2*(.25*x-target)*x).mean()
    error=float((m.weight.grad[0,0]-expected).abs())
    if error>2e-7:raise RuntimeError('DDP effective64 gradient differs from reference')
    del d,m
    return dict(world_size=world,effective_batch=64,micro_batch=micro,max_abs_error=error)


def candidate_payload(model, values, config, identity, frozen_hash, *, role, updates, epoch=None):
    validate_state_aliases(model, values)
    payload = dict(schema=SCHEMA,role=role,variant=config["variant"],config_sha256=identity,updates=updates,
        pretraining_identity=pretraining_identity(config),frozen_hash=frozen_hash,
        convnext_checkpoint_sha256=config["convnext_checkpoint_sha256"],
        croma_checkpoint_sha256=config["files"][config["croma_checkpoint"]],
        mutable_keys=list(model.mutable_keys()),state_dict={k:v.detach().cpu() for k,v in values.items()})
    if epoch is not None:
        payload["epoch"] = epoch
    return payload


def no_sar_parity(model, batch):
    was_training = model.training
    model.eval()
    names = ("ms", "sar_db", "sar_valid", "optical_valid", "ms_context", "context_valid", "local_box_context")
    original = {k: batch[k] for k in names if k in batch}
    value = dict(original)
    value["sar_db"] = torch.zeros_like(value["sar_db"])
    value["sar_valid"] = torch.zeros_like(value["sar_valid"])
    value["optical_valid"] = torch.zeros_like(value["optical_valid"])
    with torch.no_grad(), torch.autocast(value["ms"].device.type, dtype=torch.bfloat16,
                                        enabled=value["ms"].is_cuda, cache_enabled=False):
        reference = model.forward_optical(**value)["logits"]
        actual = model(**value)["logits"]
        zero = model(**original, sar_mode="zero")["logits"]
    error = float((reference.float()-actual.float()).abs().max())
    zero_error = float((reference.float()-zero.float()).abs().max())
    model.train(was_training)
    if error != 0 or zero_error != 0:
        raise RuntimeError("missing/zero SAR does not equal the current optical path")
    return {"missing_max_abs_error": error, "zero_max_abs_error": zero_error}


def main_gradient_audit(model, dataset, config, device):
    """Pure main CE/Dice/Lovasz reaches SAR content/intensity without aux loss."""
    from torch.utils.data import default_collate
    if model.stage != "joint":
        raise ValueError("main gradient audit requires joint stage")
    probe = None
    for index in range(0, min(len(dataset), 256 * 16), 16):
        if not dataset.availability[dataset.blocks[index // 16][0]]:
            continue
        sample = dataset[index]
        if bool(sample["sar_valid"].any()) and bool((sample["label"] != 255).any()):
            probe = device_batch(default_collate([sample]), device)
            break
    if probe is None:
        raise RuntimeError("no real-SAR known-label train crop for main-only gradient audit")
    was_training = model.training
    model.eval()
    pairs = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda", cache_enabled=False):
                paired_output = model.forward_paired_batch(probe) if config["variant"] == "paired" else None
                output = paired_output["on"] if paired_output is not None else model.forward_batch(probe)
            # Suppress every independent auxiliary head; this is a separate
            # diagnostic graph and never changes an optimizer or training label.
            only_main = dict(config, optical_aux_weight=0., spectral_aux_weight=0., sar_aux_weight=0.)
            loss, _ = compute_losses({"logits": output["logits"]}, probe, only_main, "optical_warmup")
            gradients = torch.autograd.grad(loss, [parameter for _, parameter in pairs], allow_unused=True)
            isolation = None
            if paired_output is not None:
                ownership = model.gradient_parameter_sets()
                optical_ids = {id(p) for p in ownership["optical"]}
                sar_ids = {id(p) for p in ownership["sar"]}
                if any(g is not None for (_, p), g in zip(pairs, gradients) if id(p) in optical_ids):
                    raise RuntimeError("paired SAR-on main loss acquired an optical parameter gradient")
                off_loss, _ = compute_losses(paired_output["off"], probe, config, "optical_warmup")
                off_gradients = torch.autograd.grad(off_loss, [p for _, p in pairs], allow_unused=True)
                if any(g is not None for (_, p), g in zip(pairs, off_gradients) if id(p) in sar_ids):
                    raise RuntimeError("paired optical-off loss acquired a SAR parameter gradient")
                optical_norm = math.sqrt(sum(float(g.float().square().sum())
                    for (_, p), g in zip(pairs, off_gradients) if id(p) in optical_ids and g is not None))
                if optical_norm <= 0:
                    raise RuntimeError("paired optical-off loss has no optical gradient")
                isolation = dict(on_optical_all_none=True, off_sar_all_none=True,
                                 off_optical_gradient_norm=optical_norm,
                                 decoder_parameters_detached_but_input_differentiable=True)
        squared = {}
        for (name, _), gradient in zip(pairs, gradients):
            if gradient is not None:
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("nonfinite main-only gradient: " + name)
                squared[name] = float(gradient.float().square().sum())
        # Names are the public model topology, rather than optimizer groups
        # where a nonzero SAR auxiliary-head gradient could hide a blocked path.
        required = {
            "croma_tail": math.sqrt(sum(value for name, value in squared.items() if name.startswith("sar_encoder."))),
            "sar_interaction": math.sqrt(sum(value for name, value in squared.items() if name.startswith("interactions."))),
            "sar_intensity": math.sqrt(sum(value for name, value in squared.items() if name.startswith("sar_intensity_encoder."))),
        }
        if any(value <= 0 for value in required.values()):
            raise RuntimeError("main-only gradient fails to reach required SAR path: " + str(required))
        return dict(main_loss=float(loss.detach()), required=required,
            only_main_loss=True, auxiliary_losses_used=False, optimizer_step=False,
            paired_isolation=isolation,
            probe_index=int(probe["index"].item()), mode="eval_after_optical_warmup_and_two_joint_smoke_updates")
    finally:
        model.train(was_training)


def run(config, *, smoke=False, resume=None, benchmark_updates=None,
        smoke_output="smoke.json", smoke_micro_batch=None):
    benchmark = benchmark_updates is not None
    if (benchmark or smoke_micro_batch is not None or smoke_output != "smoke.json") and not smoke:
        raise ValueError("benchmark/runtime overrides are smoke-only")
    if benchmark and (type(benchmark_updates) is not int or not 10 <= benchmark_updates <= 100):
        raise ValueError("benchmark must include10..100 complete global updates")
    if Path(smoke_output).name != smoke_output or not smoke_output.endswith(".json"):
        raise ValueError("smoke output must be a JSON filename inside the bound output directory")
    if benchmark and smoke_output == "smoke.json":
        raise ValueError("benchmarks need a distinct receipt filename")
    identity = validate_config(config, devices=False)
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if (world != 2 or rank != local_rank
            or not 0 <= rank < world):
        raise ValueError("full120 upgrade is authorized for two ranks")
    expected_devices = ",".join(config["gpu_uuids"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_devices:
        raise ValueError("runtime GPU UUIDs do not match the authorized pair")
    runtime = copy.deepcopy(config)
    if smoke_micro_batch is not None:
        if smoke_micro_batch not in (1, 2, 4, 8):
            raise ValueError("smoke microbatch must be1,2,4,8")
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
    recovery_dir = Path(config.get("checkpoint_dir", output))
    candidate_dir = Path(config.get("candidate_dir", output / "candidates"))
    if rank == 0:
        verify_physical_devices(config)
        verify_files(config)
        output.mkdir(parents=True, exist_ok=True)
        recovery_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if (output / "completed.json").exists():
            raise FileExistsError("formal direction3 training is already completed")
        if smoke and ((recovery_dir / "last.pt").exists() or resume):
            raise ValueError("smoke cannot alter an existing formal recovery")
        if not smoke:
            receipt = json.loads((output / "smoke.json").read_text())
            if (receipt.get("status") != "passed" or receipt.get("config_sha256") != identity
                    or not receipt.get("formal_compatible") or receipt.get("world_size") != 2
                    or receipt.get("micro_batch") != config["micro_batch"]
                    or receipt.get("updates") != SMOKE_UPDATES
                    or receipt.get("stages") != ["optical_warmup", "joint", "joint", "joint"]
                    or receipt.get("main_only_gradients", {}).get("only_main_loss") is not True
                    or receipt.get("pretraining_identity") != pretraining_identity(config)):
                raise ValueError("formal training requires this config's four-update DDP smoke with recovery")
            if config["variant"] == "paired":
                isolation = receipt.get("main_only_gradients", {}).get("paired_isolation") or {}
                if not (isolation.get("on_optical_all_none") and isolation.get("off_sar_all_none")):
                    raise ValueError("paired training requires a passed disjoint-gradient audit")
            if not resume and (recovery_dir / "last.pt").exists():
                raise FileExistsError("existing recovery requires explicit --resume")
    dist.barrier()
    global_mean_audit=check_ddp_global_mean(device,config['micro_batch']) if smoke else None
    engine.seed_everything(config["seed"])
    model = make_model(config, device)
    keys = tuple(model.mutable_keys())
    frozen_keys = tuple(key for key in model.state_dict() if key not in set(keys))
    frozen_hash = state_hash(subset(model, frozen_keys))
    groups = model.parameter_groups(optical_lr=config["optical_lr"], sar_lr=config["sar_lr"],
        tail_lr=config["tail_lr"], weight_decay=config["weight_decay"])
    validate_optimizer_inventory(model, groups)
    for group in groups:
        group["initial_lr"] = group["lr"]
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)
    ema = subset(model, keys, device)
    completed, saved_rng = 0, None
    if resume:
        path = recovery_dir / "last.pt" if resume == "last" else Path(resume)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_recovery(payload, model, config, identity, frozen_hash)
        # Validate both tensor sets before the authoritative raw model is used.
        apply_delta(model, payload["ema"])
        apply_delta(model, payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        ema = {key: value.to(device) for key, value in payload["ema"].items()}
        completed, saved_rng = payload["updates"], payload["rank_rng"][rank]
        del payload
    objective = TrainingObjective(model, config)
    ddp, previous_stage = None, None
    started, initial_update = time.monotonic(), completed
    target = (benchmark_updates or SMOKE_UPDATES) if smoke else config["epochs"] * config["updates_per_epoch"]
    engine.seed_everything(config["seed"] + 100000 + rank)
    if saved_rng is not None:
        restore_rng(saved_rng, rank, device)
    audits, update_seconds = [], []
    recovery_audit = None
    dataset, iterator, loaded_epoch, probe = None, None, None, None
    boundary = config["warmup_epochs"] * config["updates_per_epoch"]
    torch.cuda.reset_peak_memory_stats(device)
    while completed < target:
        epoch, epoch_update = divmod(completed, config["updates_per_epoch"])
        stage = ("joint" if benchmark_updates is not None else
                 (("optical_warmup" if completed == 0 else "joint") if smoke
                  else stage_for_update(completed, config)))
        if stage != previous_stage:
            if ddp is not None:
                del ddp
            model.set_stage(stage)
            model.train()
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
                plan_path = output / "plans" / f"epoch{epoch+1:03d}.json"
                if plan_path.exists():
                    if json.loads(plan_path.read_text()) != plan_audit:
                        raise ValueError("epoch sampling audit changed on resume")
                else:
                    engine.atomic_json(plan_path, plan_audit)
            iterator = iter(loader)
            loaded_epoch = epoch
        if smoke and completed == 0:
            probe = device_batch(next(iter(loader)), device)
            no_sar_parity(model, probe)
        # Unlike the earlier D1/D2 implementation, all actual stage audits
        # are durably recorded, including updates961/962 between log intervals.
        audited = (smoke and (benchmark_updates is None or completed < SMOKE_UPDATES)) or completed in (0, 1, boundary, boundary + 1)
        update_started = time.monotonic()
        metrics = train_update(ddp, model, optimizer, iterator, completed, epoch_update,
                               rank, config, device=device, audit=audited)
        update_ema(model, keys, ema, config["ema_decay"], completed)
        if smoke:
            torch.cuda.synchronize(device)
            duration = torch.tensor(time.monotonic() - update_started, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            update_seconds.append(float(duration))
        completed += 1
        if smoke and not benchmark_updates and completed == 3:
            recovery_audit = smoke_recovery_roundtrip(recovery_dir,model,keys,ema,optimizer,config,
                identity,frozen_hash,rank,device)
        if audited:
            entry = dict(update=completed, stage=stage, gradients=metrics["gradients"], rank=rank,
                sar_known_pixels=metrics["audit_sar_known_pixels"],
                sar_gradient_required=stage == "joint" and metrics["audit_sar_known_pixels"] > 0)
            audits.append(entry)
            if not smoke:
                # Per-update/rank files are unambiguous after an exact resume;
                # a replay overwrites the same identity rather than duplicating it.
                engine.atomic_json(output / "stage_audits" / f"update{completed:06d}_rank{rank}.json",
                    dict(config_sha256=identity, time_unix=time.time(), **entry))
        if rank == 0 and (smoke or audited or completed % 10 == 0 or completed == initial_update + 1):
            row = dict(status="smoke" if smoke else "training", updates=completed, total_updates=target,
                epoch=(completed-1)//config["updates_per_epoch"]+1, stage=stage, config_sha256=identity,
                seconds_per_update=(time.monotonic()-started)/(completed-initial_update),
                time_unix=time.time(), pid=os.getpid(), **metrics)
            engine.atomic_json(output / (Path(smoke_output).stem + "_progress.json" if smoke else "progress.json"), row)
            if not smoke:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        if (not smoke and completed % config["updates_per_epoch"] == 0
                and completed//config["updates_per_epoch"] in config["checkpoint_epochs"]):
            epoch_number = completed // config["updates_per_epoch"]
            if rank == 0:
                path = candidate_dir / f"epoch{epoch_number:03d}_ema_delta.pt"
                engine.atomic_checkpoint(path, candidate_payload(model, ema, config, identity, frozen_hash,
                    role="candidate_ema_delta", updates=completed, epoch=epoch_number))
            dist.barrier()
        if not smoke and (completed % config["save_every_updates"] == 0 or completed == target):
            save_recovery(recovery_dir, model, keys, ema, optimizer, config, identity, frozen_hash, completed, rank, device)
    main_audit = None
    if smoke:
        del ddp
        main_audit = main_gradient_audit(model, dataset, config, device)
    if dataset is not None:
        dataset.close()
    if state_hash(subset(model, frozen_keys)) != frozen_hash:
        raise RuntimeError("immutable pretrained state changed")
    raw_digest = assert_rank_hashes(state_hash(subset(model, keys)), "all mutable parameters/buffers")
    ema_digest = assert_rank_hashes(state_hash(ema), "all EMA parameters/buffers")
    peak = torch.tensor(torch.cuda.max_memory_allocated(device)/1024**3, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if smoke:
        parity_audit = no_sar_parity(model, probe)
        del probe
        if rank == 0:
            path = output / "smoke" / (Path(smoke_output).stem + "_delta.pt")
            delta = candidate_payload(model, subset(model, keys), config, identity, frozen_hash,
                role="smoke_delta", updates=target)
            delta.update(world_size=dist.get_world_size(), micro_batch=config["micro_batch"])
            engine.atomic_checkpoint(path, delta)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config["seed"])
                fresh = make_model(config)
            restored = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
            apply_delta(fresh, restored)
            if (state_hash(subset(fresh, keys)) != raw_digest
                    or state_hash(subset(fresh, frozen_keys)) != frozen_hash):
                raise RuntimeError("fresh-model strict delta roundtrip failed")
            delta_sha = engine.sha256(path)
            for entry in audits:
                validate_stage_gradients(entry["stage"], entry["gradients"], require_sar=entry["sar_gradient_required"])
            if len(audits) < SMOKE_UPDATES:
                raise RuntimeError("smoke did not exercise three complete updates")
            engine.atomic_json(output / smoke_output, dict(status="passed", config_sha256=identity,
                updates=target, stages=[entry["stage"] for entry in audits],
                raw_sha256=raw_digest, ema_sha256=ema_digest, world_size=dist.get_world_size(),
                micro_batch=config["micro_batch"], formal_compatible=formal_compatible,
                benchmark=benchmark_updates is not None, update_seconds=update_seconds,
                steady_seconds_per_update=sum(update_seconds[5:] if benchmark_updates else update_seconds[1:])/len(update_seconds[5:] if benchmark_updates else update_seconds[1:]),
                frozen_hash=frozen_hash, immutable_pretraining_unchanged=True, rank_states_equal=True,
                strict_roundtrip=True, missing_sar_current_optical_exact=True, gradients=audits,
                zero_sar_current_optical_exact=True, no_sar_parity=parity_audit,
                recovery_roundtrip=recovery_audit,
                global_mean_audit=global_mean_audit,
                pretraining_identity=pretraining_identity(config),
                main_only_gradients=main_audit,
                teachers_loaded=False, ms75_base_loaded=False,
                smoke_delta=str(path) if path.exists() else None,
                smoke_delta_sha256=delta_sha if path.exists() else None,
                elapsed_seconds=time.monotonic()-started, peak_gib=float(peak)))
    elif rank == 0:
        candidates = []
        for epoch_number in config["checkpoint_epochs"]:
            path = candidate_dir / f"epoch{epoch_number:03d}_ema_delta.pt"
            if not path.is_file():
                raise FileNotFoundError("missing precommitted candidate: " + str(path))
            candidates.append(dict(epoch=epoch_number, path=str(path), sha256=engine.sha256(path)))
        verify_files(config)
        receipt = dict(status="training_completed_evaluation_pending", schema=SCHEMA,
            variant=config["variant"], recovery_checkpoint=str(recovery_dir / "last.pt"),
            candidate_directory=str(candidate_dir), final_ema=candidates[-1],
            config_sha256=identity, updates=completed, candidate_checkpoints=candidates,
            pretraining_identity=pretraining_identity(config), frozen_hash=frozen_hash,
            raw_sha256=raw_digest, ema_sha256=ema_digest, rank_states_equal=True,
            immutable_pretraining_unchanged=True, elapsed_this_run_seconds=time.monotonic()-started,
            stage_audit_directory=str(output / "stage_audits"),
            teachers_loaded=False, ms75_base_loaded=False, auto_full=False, official_test_auto=False)
        engine.atomic_json(output / "completed.json", receipt)
        engine.atomic_json(output / "progress.json", receipt)
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


