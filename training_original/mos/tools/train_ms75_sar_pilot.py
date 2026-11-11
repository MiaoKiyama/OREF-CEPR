"""Fixed-budget, matched P/R/Z continuation of an authenticated train96 MS75.

Only mutable tensors are checkpointed; each artifact is bound to the immutable
optical EMA, source files and sealed crop replay. No teacher is trained here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tools import train_target_dino as engine
from tcmmsline3.ms75_rebuild_model import build_model
from tcmmsline3.ms75_sar_pilot_model import MS75SARPilot, PILOT_INPUT_KEYS, FROZEN_PREFIXES
from tcmmsline3.ms75_sar_pilot_data import MS75SARPilotDataset

SCHEMA = "ms75_sar_pilot_fixed_budget_20260929_v1"


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        v = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def mutable_keys(model):
    # Include all counters/normalization buffers outside the frozen encoders.
    return tuple(k for k in model.state_dict()
                 if not (k.startswith("optical.") and k[8:].startswith(FROZEN_PREFIXES)))


def subset(model, keys, device="cpu"):
    values = model.state_dict()
    return {k: values[k].detach().to(device).clone() for k in keys}


def validate_delta(model, state):
    expected = set(mutable_keys(model))
    if set(state) != expected:
        raise ValueError("delta tensor inventory differs from authenticated model")
    full = model.state_dict()
    for name, value in state.items():
        if full[name].shape != value.shape or full[name].dtype != value.dtype:
            raise ValueError("delta tensor shape/dtype changed: " + name)
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite delta tensor: " + name)
        if name in ("sar_mean", "sar_std") and not torch.equal(full[name].cpu(), value.cpu()):
            raise ValueError("train-only SAR normalization changed")


def apply_delta(model, state):
    validate_delta(model, state)
    full = model.state_dict()
    # BatchNorm synthesizes absent num_batches_tracked during partial loading;
    # merge the authenticated mutable inventory into current frozen references
    # and require a full strict load, avoiding ambiguous missing-key semantics.
    full.update(state)
    model.load_state_dict(full, strict=True)


def verify_config(config):
    from tools.evaluate_ms75_sar_pilot import collect_aux_bindings
    fixed = dict(schema=SCHEMA, seed=1337, updates=1000, effective_batch=64,
                 optical_lr=1.5e-5, sar_lr=3e-4, weight_decay=.01, warmup_updates=50,
                 ema_decay=.99, auto_full=False, official_test_auto=False)
    for name, value in fixed.items():
        if config.get(name) != value:
            raise ValueError("sealed pilot protocol mismatch: " + name)
    if config["arm"] not in ("P", "R", "Z") or config["batch_size"] not in (1, 2, 4):
        raise ValueError("invalid pilot arm/batch")
    if 64 % config["batch_size"] or not 0 <= config["num_workers"] <= 8:
        raise ValueError("invalid effective batch or workers")
    if config["gpu"] not in (0, 2):
        raise ValueError("user authorized only physical GPU0/2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]:
        raise ValueError("physical GPU UUID binding differs")
    for path, digest in config["files"].items():
        if engine.sha256(path) != digest:
            raise ValueError("pilot file identity changed: " + path)
    for required in (config["base_checkpoint"], config["base_binding"], config["replay_manifest"],
                     config["views_cache"], str(Path(__file__).resolve())):
        if required not in config["files"]:
            raise ValueError("missing required file identity: " + required)
    if collect_aux_bindings(config["data_root"], config["train_sar_scene_ids"]) != config["train_aux_binding"]:
        raise ValueError("paired training SAR asset/registration identity changed")
    return engine.canonical_hash(config)


def make_model(config, device="cuda"):
    binding = json.loads(Path(config["base_binding"]).read_text())
    checkpoint = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=False)
    if (checkpoint.get("stage") != "selection" or checkpoint.get("role") != "best_held_ema"
            or checkpoint.get("binding_sha256") != binding["binding_sha256"]):
        raise ValueError("pilot requires the bound train96 selection EMA, never full120")
    if checkpoint["state"].get("best_epoch") != config["base_epoch"]:
        raise ValueError("base selection epoch changed")
    raw = json.loads(Path(config["raw_manifest"]).read_text())
    if (binding["splits"]["selection_train_ids"] != raw["selection_train_ids"]
            or binding["splits"]["held_ids"] != raw["held_ids"]):
        raise ValueError("optical base and SAR pilot development split differ")
    engine.seed_everything(config["seed"])
    base = build_model(binding["config"], load_pretrained=False)
    base.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model = MS75SARPilot(base, arm=config["arm"], sar_mean=config["sar_mean"], sar_std=config["sar_std"])
    return model.to(device)


def make_loader(config, start=0):
    dataset = MS75SARPilotDataset(config["replay_manifest"], views_cache=config["views_cache"], start_sample=start)
    if dataset.replay["samples"] != config["updates"]*64:
        raise ValueError("replay crop budget differs from fixed endpoint")
    generator = torch.Generator().manual_seed(config["seed"])
    kwargs = dict(dataset=dataset, batch_size=config["batch_size"], shuffle=False,
                  num_workers=config["num_workers"], pin_memory=True, generator=generator,
                  worker_init_fn=engine._worker_init)
    if config["num_workers"]:
        kwargs.update(prefetch_factor=2, persistent_workers=False)
    return dataset, DataLoader(**kwargs)


def inputs(batch):
    return {k: batch[k].cuda(non_blocking=True) for k in PILOT_INPUT_KEYS}


def gradient_summary(model):
    result = {}
    for prefix in ("optical.", "sar_encoder.", "interaction."):
        values = [p.grad.detach() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        result[prefix] = math.sqrt(sum(float(v.float().square().sum()) for v in values))
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("a frozen encoder acquired gradients")
    return result


def update(model, optimizer, batches, index, config):
    warmup = config["warmup_updates"]
    factor = (index+1)/warmup if index < warmup else .1 + .9*.5*(1+math.cos(math.pi*(index-warmup)/(config["updates"]-warmup)))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor
    optimizer.zero_grad(set_to_none=True)
    total, seen, branch_scale = 0., [], None
    audit = index < 2 or (index+1) % 50 == 0
    for batch in batches:
        value = inputs(batch)
        labels = batch["label"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            output = model(**value)
            logits = output["logits"]
        if audit and model.multimodal:
            branch_scale = {name:float(output[name].detach().float().square().mean().sqrt())
                            for name in ("pilot_sar_features", "pilot_sar_delta")}
        del output
        loss = engine.per_image_segmentation_loss(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite pilot loss")
        (loss * len(labels)/64).backward()
        total += float(loss.detach())*len(labels)/64
        seen.extend(int(v) for v in batch["index"])
    if seen != list(range(index*64, (index+1)*64)):
        raise ValueError("update did not consume exactly the sealed next64 crops")
    gradients = gradient_summary(model) if audit else None
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    model.project_constrained_parameters_()
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=total, gradient_norm=float(norm), lr_factor=factor, gradients=gradients,
                branch_rms_last_microbatch=branch_scale)


def run(config, smoke=False):
    torch.set_num_threads(config["torch_threads"])
    torch.cuda.set_per_process_memory_fraction(.75)
    identity = verify_config(config)
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    model = make_model(config)
    keys = mutable_keys(model)
    frozen_keys = tuple(k for k in model.state_dict() if k not in keys)
    frozen_hash = state_hash({k:v for k,v in model.state_dict().items() if k in frozen_keys})
    initial_hash = state_hash(subset(model, keys))
    initial_optical_hash = state_hash({k:v for k,v in model.state_dict().items() if k.startswith("optical.")})
    optimizer = torch.optim.AdamW(model.parameter_groups(optical_lr=config["optical_lr"], sar_lr=config["sar_lr"], weight_decay=config["weight_decay"]), betas=(.9,.999), eps=1e-8)
    ema = subset(model, keys, "cuda")
    completed = 0
    if not smoke:
        receipt = json.loads((output/"smoke.json").read_text())
        if receipt["status"] != "passed" or receipt["config_sha256"] != identity:
            raise ValueError("formal fit requires matching successful two-update smoke")
        if (output/"completed.json").exists():
            raise FileExistsError("pilot is already completed")
        if (output/"last.pt").exists():
            saved = torch.load(output/"last.pt", map_location="cpu", weights_only=False)
            if (saved.get("schema") != SCHEMA or saved["config_sha256"] != identity
                    or saved["frozen_hash"] != frozen_hash
                    or type(saved["updates"]) is not int or not 0 <= saved["updates"] <= config["updates"]):
                raise ValueError("resume identity differs")
            validate_delta(model, saved["ema"])
            apply_delta(model, saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            ema = {k:v.cuda() for k,v in saved["ema"].items()}
            completed = saved["updates"]
            engine.restore_rng(saved["rng"])
            del saved
    dataset, batches = make_loader(config, completed*64)
    if smoke:
        value = inputs(next(iter(batches)))
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            reference = model.optical(**{k:v for k,v in value.items() if k not in ("sar_db","sar_valid","optical_valid")})["logits"]
            initial = model(**value)["logits"]
            error = float((reference-initial).abs().max())
        if error != 0:
            raise RuntimeError("SAR zero-terminal initialization does not exactly retain optical output")
        del value, reference, initial
    if completed == 0:
        # Branch construction uses a different number of RNG draws for P.
        # Reset before the shared stochastic decoder/augmentation sequence.
        engine.seed_everything(config["seed"]+100000)
    model.train()
    started, first = time.monotonic(), completed
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(batches)
    target = 2 if smoke else config["updates"]
    audits = []
    while completed < target:
        # Consume one physical batch at a time. Workers can prepare the next
        # batch while CUDA computes, instead of stalling CUDA until all64
        # samples are resident. The same ordered64 gradients are accumulated.
        chunk = itertools.islice(iterator, 64//config["batch_size"])
        metrics = update(model, optimizer, chunk, completed, config)
        del chunk
        with torch.no_grad():
            state = model.state_dict()
            for key in keys:
                value = state[key].detach()
                if value.is_floating_point():
                    ema[key].lerp_(value, 1-config["ema_decay"])
                else:
                    ema[key].copy_(value)
        completed += 1
        if metrics["gradients"] is not None:
            audits.append(metrics["gradients"])
        if completed % 10 == 0 or smoke or completed == first+1:
            elapsed = time.monotonic()-started
            row = dict(status="smoke" if smoke else "training", arm=config["arm"], updates=completed,
                     total_updates=target, seconds_per_update=elapsed/(completed-first),
                     config_sha256=identity, time_unix=time.time(), pid=os.getpid(), **metrics)
            engine.atomic_json(output/("smoke_progress.json" if smoke else "progress.json"), row)
            if not smoke:
                with (output/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row,allow_nan=False)+"\n")
        if not smoke and (completed % 50 == 0 or completed == target):
            engine.atomic_checkpoint(output/"last.pt", dict(schema=SCHEMA, config_sha256=identity,
                frozen_hash=frozen_hash, updates=completed, model=subset(model, keys),
                ema={k:v.cpu() for k,v in ema.items()}, optimizer=optimizer.state_dict(), rng=engine.rng_state()))
    if state_hash({k:v for k,v in model.state_dict().items() if k in frozen_keys}) != frozen_hash:
        raise RuntimeError("frozen optical encoder parameters or buffers changed")
    verify_config(config)
    if not all(bool(torch.isfinite(v).all()) for v in ema.values() if v.is_floating_point()):
        raise FloatingPointError("nonfinite final EMA")
    if smoke:
        if (not all(math.isfinite(v) for row in audits for v in row.values())
                or audits[1]["optical."] <= 0
                or (config["arm"] != "P" and (audits[0]["interaction."] <= 0
                     or audits[0]["sar_encoder."] != 0 or audits[1]["interaction."] <= 0))
                or (config["arm"] == "R" and audits[1]["sar_encoder."] <= 0)):
            raise RuntimeError("main segmentation loss does not reach required trainable branches")
        state = subset(model, keys)
        temporary = output/"smoke_delta_roundtrip.pt"
        engine.atomic_checkpoint(temporary, dict(state_dict=state))
        fresh = make_model(config, device="cpu")
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)["state_dict"]
        apply_delta(fresh, loaded)
        roundtrip = (state_hash(subset(fresh, keys)) == state_hash(state)
                     and state_hash({k:v for k,v in fresh.state_dict().items() if k in frozen_keys}) == frozen_hash)
        if not roundtrip:
            raise RuntimeError("fresh-base delta checkpoint roundtrip failed")
        del fresh, loaded
        temporary.unlink()
        receipt = dict(status="passed", config_sha256=identity, updates=2, initial_max_error=error,
                       frozen_hash=frozen_hash, frozen_state_unchanged=True,
                       delta_roundtrip=roundtrip,
                       initial_mutable_sha256=initial_hash, gradients=audits,
                       initial_optical_sha256=initial_optical_hash,
                       seconds_per_update=(time.monotonic()-started)/2,
                       peak_gib=torch.cuda.max_memory_allocated()/1024**3)
        engine.atomic_json(output/"smoke.json", receipt)
    else:
        artifact = output/"final_ema_delta.pt"
        engine.atomic_checkpoint(artifact, dict(schema=SCHEMA, role="fixed_endpoint_ema_delta",
            config=config, config_sha256=identity, frozen_hash=frozen_hash, updates=completed,
            base_sha256=config["files"][config["base_checkpoint"]], state_dict={k:v.cpu() for k,v in ema.items()}))
        receipt = dict(status="training_completed_evaluation_pending", arm=config["arm"], updates=completed,
                       config_sha256=identity, final_ema=str(artifact), sha256=engine.sha256(artifact),
                       elapsed_seconds=time.monotonic()-started, initial_mutable_sha256=initial_hash,
                       frozen_state_unchanged=True, auto_full=False, official_test_auto=False)
        engine.atomic_json(output/"completed.json", receipt)
        engine.atomic_json(output/"progress.json", receipt)
    dataset.close()
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.smoke)


if __name__ == "__main__":
    main()
