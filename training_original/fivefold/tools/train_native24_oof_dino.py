"""Single-GPU, train96-only DINO24 fit for one sealed OOF fold.

The public pretrained backbone is the only initialization source. A recovery
may continue only this exact fold/configuration; held scenes are never loaded.
The data adapter must provide deterministic per-index crops and a fold audit.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "native24_oof_dino_single_gpu_v1"
STOP = False


def stop_handler(signum, frame):
    global STOP
    STOP = True


def scene_hash(ids):
    return hashlib.sha256(("\n".join(map(str, ids)) + "\n").encode()).hexdigest()


def validate_config(config):
    fixed = dict(schema=SCHEMA, seed=1337, classes=24, effective_batch=64,
                 micro_batch=4, samples_per_epoch=30720, updates_per_epoch=480,
                 scheduler_horizon=33600, warmup_updates=2000,
                 auto_evaluation=False, initialization="public_pretrained")
    for name, value in fixed.items():
        if type(config.get(name)) is not type(value) or config[name] != value:
            raise ValueError(f"OOF DINO requires {name}={value!r}")
    if config.get("variant") not in ("v2l", "v3l"):
        raise ValueError("expected DINOv2-L or DINOv3-L")
    epochs = 17
    if type(config.get("epochs")) is not int or config["epochs"] != epochs:
        raise ValueError("fixed duration changed")
    if type(config.get("fold_id")) is not int or config["fold_id"] not in range(5):
        raise ValueError("expected fold 0..4")
    train, held = [str(x) for x in config["train_scene_ids"]], [str(x) for x in config["held_scene_ids"]]
    if len(train) != 96 or len(set(train)) != 96 or len(held) != 24 or len(set(held)) != 24:
        raise ValueError("expected unique 96 training and 24 held scenes")
    if set(train) & set(held):
        raise ValueError("held scene in fitting range")
    if any(config.get(key) for key in ("source_checkpoint", "parent_checkpoint", "supervised_initialization")):
        raise ValueError("supervised initialization is forbidden")
    if type(config.get("num_workers")) is not int or not 0 <= config["num_workers"] <= 4:
        raise ValueError("num_workers must be in 0..4")
    if not config.get("cpu_affinity") or len(set(config["cpu_affinity"])) != len(config["cpu_affinity"]):
        raise ValueError("explicit unique CPU affinity required")
    if config.get("candidate_epochs", [epochs]) != [epochs]:
        raise ValueError("only the predetermined final EMA may be retained")
    if int(config.get("recovery_interval_updates", 240)) <= 0:
        raise ValueError("invalid recovery interval")
    for name in ("output_dir", "pretrained_file", "pretrained_sha256", "source_files",
                 "reference_config_file", "reference_config_sha256", "plans_manifest", "plans_manifest_sha256"):
        if not config.get(name):
            raise ValueError("missing sealed input: " + name)


def build_config(base, fold_data, paths):
    """Build a fresh config on the server after the immutable sources are copied.

    ``base`` is the authenticated former DINO24 configuration, used only for
    public model/pretraining/reference identifiers. ``fold_data`` is the new
    fold-only data description. No previous checkpoint or optimizer is copied.
    ``paths`` must include output_dir, project_root, cpu_affinity, and the
    persisted fold_data_file; source_files may be supplied by the launcher.
    """
    from tools import train_target_dino as engine
    variant = base["variant"]
    epochs = 17
    fold_file = paths.get("fold_data_file")
    if not fold_file:
        raise ValueError("persisted fold_data_file required")
    train_ids = fold_data.get("train_ids", fold_data.get("train_scene_ids"))
    held_ids = fold_data.get("held_ids", fold_data.get("held_scene_ids"))
    project = Path(paths["project_root"]).resolve()
    sources = paths.get("source_files")
    if sources is None:
        sources = {str(path.resolve()): engine.sha256(path)
                   for directory in (project / "tools", project / "tcmmsline3")
                   for path in sorted(directory.rglob("*.py"))}
    config = dict(schema=SCHEMA, variant=variant, fold_id=int(fold_data["fold_id"]),
        seed=1337, classes=24, effective_batch=64, micro_batch=4,
        samples_per_epoch=30720, updates_per_epoch=480, scheduler_horizon=33600,
        warmup_updates=2000, epochs=epochs, candidate_epochs=[epochs],
        auto_evaluation=False, initialization="public_pretrained",
        pretrained_file=base["pretrained_file"], pretrained_sha256=base["pretrained_sha256"],
        reference_config_file=paths.get("reference_config_file", base["reference_config_file"]),
        source_files=sources, data_files=copy.deepcopy(fold_data.get("files", {})),
        train_scene_ids=list(train_ids), held_scene_ids=list(held_ids),
        train_ids=list(train_ids), held_ids=list(held_ids),
        fold_data_file=str(fold_file), fold_data_sha256=engine.sha256(fold_file),
        plans_manifest=fold_data["plans_manifest"],
        plans_manifest_sha256=engine.sha256(fold_data["plans_manifest"]),
        output_dir=str(paths["output_dir"]), num_workers=int(paths.get("num_workers", 4)),
        cpu_affinity=list(paths["cpu_affinity"]), physical_gpu=paths.get("gpu"),
        recovery_interval_updates=240,
        checkpoint_reserve_bytes=int(paths.get("checkpoint_reserve_bytes", 2 * 1024**3)),
        checkpoint_lock_file=str(paths.get("checkpoint_lock_file", project.parent / "checkpoint.lock")),
        data=copy.deepcopy(fold_data["data"]),
        loss="per_image_CE+0.5Dice+0.25Lovasz_native24_ignore255", precision="bfloat16",
        outer_fold_id=int(fold_data["fold_id"]), inner_fold_id=None,
        expert_train_scene_hash=scene_hash(train_ids),
        normalization_provenance="fixed_existing_preprocessing_constants_no_fold_fitting")
    config["reference_config_sha256"] = engine.sha256(config["reference_config_file"])
    for key in ("weights25", "audit", "fold_plan_sha256", "normalization"):
        if key in fold_data:
            config[key] = copy.deepcopy(fold_data[key])
    if paths.get("gpu_uuid"):
        config["gpu_uuid"] = paths["gpu_uuid"]
    validate_config(config)
    return config


def check_binding(config, engine):
    validate_config(config)
    for path, expected in config["source_files"].items():
        if engine.sha256(path) != expected:
            raise ValueError("source changed: " + path)
    if str(Path(__file__).resolve()) not in config["source_files"]:
        raise ValueError("trainer source is not bound")
    files = [(config["pretrained_file"], config["pretrained_sha256"]),
             (config["reference_config_file"], config["reference_config_sha256"]),
             (config["plans_manifest"], config["plans_manifest_sha256"])]
    if config.get("data_audit_file"):
        files.append((config["data_audit_file"], config["data_audit_sha256"]))
    if config.get("fold_data_file"):
        files.append((config["fold_data_file"], config["fold_data_sha256"]))
    for path, expected in files:
        if engine.sha256(path) != expected:
            raise ValueError("bound data/pretraining changed: " + path)
    for path, expected in config.get("data_files", {}).items():
        if engine.sha256(path) != expected:
            raise ValueError("fold data manifest/index changed: " + path)
    return engine.canonical_hash(config)


class Global64Sampler:
    def __init__(self, size, start_update=0):
        if size <= 0 or size % 64 or not 0 <= start_update <= size // 64:
            raise ValueError("invalid global64 sampling interval")
        self.size, self.start_update = size, start_update

    def __iter__(self):
        return iter(range(self.start_update * 64, self.size))

    def __len__(self):
        return self.size - self.start_update * 64


def verify_seen(indices, scene_ids, completed, config):
    first = (completed % config["updates_per_epoch"]) * 64
    if indices != list(range(first, first + 64)) or len(scene_ids) != 64:
        raise RuntimeError("global64 crop inventory changed")
    blocks = [scene_ids[i:i + 16] for i in range(0, 64, 16)]
    if any(len(set(block)) != 1 for block in blocks) or len({block[0] for block in blocks}) != 4:
        raise RuntimeError("expected four distinct scene blocks per update")
    if not set(map(str, scene_ids)) <= set(map(str, config["train_scene_ids"])):
        raise RuntimeError("non-training scene reached optimizer")


def validate_dataset(dataset, config):
    if len(dataset) != config["samples_per_epoch"]:
        raise RuntimeError("wrong crop count")
    actual = getattr(dataset, "scene_ids", None)
    if actual is None and hasattr(dataset, "base"):
        actual = dataset.base.scene_ids
    if actual is None:
        actual = dataset.audit.get("train_scene_ids")
    if list(map(str, actual or [])) != list(map(str, config["train_scene_ids"])):
        raise RuntimeError("actual fitting scenes differ from bound train96 order")
    if not isinstance(dataset.audit, dict):
        raise RuntimeError("dataset must expose a sampling audit")


def make_loader(config, reference, epoch, completed, output, engine):
    import torch
    from torch.utils.data import DataLoader
    from tools.native24_oof_data import make_dino_dataset
    dataset = make_dino_dataset(config, reference, epoch)
    validate_dataset(dataset, config)
    audit = {**dataset.audit, "schema": "native24_oof_dino_sampling_v1",
             "fold_id": config["fold_id"], "epoch_zero_based": epoch,
             "train_scene_ids": config["train_scene_ids"],
             "train_scene_ids_sha256": scene_hash(config["train_scene_ids"]),
             "held_scene_ids_sha256": scene_hash(config["held_scene_ids"]),
             "source_label_zero_supervised": False, "artificial_padding_ignore": 255,
             "physical_batch": 4, "effective_batch": 64,
             "samples_per_epoch": 30720, "updates_per_epoch": 480}
    path = output / "plans" / f"epoch{epoch+1:03d}.json"
    if path.exists() and json.loads(path.read_text()) != audit:
        raise RuntimeError("regenerated epoch sampling plan differs")
    engine.atomic_json(path, audit)
    kwargs = dict(dataset=dataset,
        sampler=Global64Sampler(len(dataset), completed % config["updates_per_epoch"]),
        batch_size=16, num_workers=config["num_workers"], pin_memory=True,
        worker_init_fn=engine._worker_init,
        generator=torch.Generator().manual_seed(config["seed"] + 1009 * epoch))
    if config["num_workers"]:
        kwargs.update(prefetch_factor=1, persistent_workers=False, multiprocessing_context="spawn")
    return dataset, DataLoader(**kwargs), engine.canonical_hash(audit)


def train_update(model, optimizer, iterator, config, completed, engine, audit=False):
    import torch
    factor = engine.lr_multiplier(completed, config["scheduler_horizon"], config["warmup_updates"])
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device="cuda")
    indices, scenes, zero_pixels = [], [], 0
    for _ in range(4):
        batch = next(iterator)
        if len(batch["index"]) != 16:
            raise RuntimeError("incomplete sixteen-crop I/O block")
        indices += batch["index"].tolist()
        scenes += batch["scene_id"]
        for offset in range(0, 16, 4):
            ms = batch["ms"][offset:offset+4].to("cuda", non_blocking=True)
            labels = batch["label"][offset:offset+4].to("cuda", non_blocking=True)
            if audit:
                zero_pixels += int((labels == 255).sum())
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                logits = model(ms)["logits"]
            with torch.autocast("cuda", enabled=False):
                loss = engine.per_image_segmentation_loss(logits, labels)
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("nonfinite segmentation loss")
            (loss * (4 / 64)).backward()
            total += loss.detach() * (4 / 64)
            del ms, labels, logits, loss
        del batch
    verify_seen(indices, scenes, completed, config)
    details = None
    if audit:
        head = model.decoder.head[-1]
        upper = [p.grad for name, p in model.named_parameters()
                 if name.startswith("backbone.blocks.23.") and p.grad is not None]
        permanent = [name for name, p in model.named_parameters()
                     if not p.requires_grad and p.grad is not None]
        details = dict(ignored_pixels=zero_pixels,
            first_semantic_head_gradient=float(head.weight.grad[0].abs().sum()),
            semantic_head_gradient=float(head.weight.grad[1:].abs().sum()),
            upper_backbone_gradient=sum(float(g.abs().sum()) for g in upper),
            frozen_parameters_with_grad=permanent,
            sample_first=indices[0], sample_last=indices[-1], scene_ids=sorted(set(scenes)))
        if min(details["first_semantic_head_gradient"], details["semantic_head_gradient"]) <= 0 or permanent:
            raise RuntimeError("native24/frozen-parameter gradient audit failed")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=float(total), gradient_norm=float(norm), lr_factor=factor, gradient_audit=details)


@contextmanager
def checkpoint_lock(config):
    # The shared storage writer owns the lock.
    yield


def sampler_state(completed, config, plan_sha256):
    return dict(epoch_zero_based=completed // config["updates_per_epoch"],
                next_index=(completed % config["updates_per_epoch"]) * 64,
                updates_per_epoch=config["updates_per_epoch"],
                last_executed_plan_sha256=plan_sha256,
                train_scene_ids_sha256=scene_hash(config["train_scene_ids"]))


def save_recovery(path, model, optimizer, ema, config, identity, completed, plan_hash, engine):
    with checkpoint_lock(config):
        payload = dict(schema=SCHEMA, role="rolling_recovery", binding_sha256=identity,
            config=config, updates=completed, world_size=1, fold_id=config["fold_id"],
            model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            ema={k: v.detach().cpu() for k, v in ema.state.items()},
            ema_updates=ema.updates, optimizer=optimizer.state_dict(), rng=engine.rng_state(),
            sampler=sampler_state(completed, config, plan_hash),
            scheduler=dict(horizon=33600, warmup=2000, completed_updates=completed),
            scaler=None, precision="bfloat16_no_grad_scaler")
        engine.atomic_checkpoint(path, payload, reserve_bytes=int(config.get("checkpoint_reserve_bytes", 2 * 1024**3)))


def verify_recovery(payload, config, identity):
    completed = payload.get("updates")
    if (payload.get("schema") != SCHEMA or payload.get("role") != "rolling_recovery"
            or payload.get("binding_sha256") != identity or payload.get("world_size") != 1
            or payload.get("fold_id") != config["fold_id"]):
        raise ValueError("recovery fold/source identity mismatch")
    if type(completed) is not int or not 0 <= completed <= config["epochs"] * 480:
        raise ValueError("invalid recovery progress")
    if payload.get("ema_updates") != completed:
        raise ValueError("EMA count differs from successful updates")
    expected = sampler_state(completed, config, payload["sampler"]["last_executed_plan_sha256"])
    if payload["sampler"] != expected:
        raise ValueError("recovery sampler cursor changed")
    if payload.get("scheduler") != dict(horizon=33600, warmup=2000, completed_updates=completed):
        raise ValueError("scheduler recovery differs")
    return completed


def restore_recovery(payload, model, optimizer, ema, config, identity, engine, tree_hash):
    completed = verify_recovery(payload, config, identity)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    ema.load(payload["ema"], completed)
    engine.restore_rng(payload["rng"])
    for name, actual in (("model", model.state_dict()), ("optimizer", optimizer.state_dict()),
                         ("ema", ema.state), ("rng", engine.rng_state())):
        if tree_hash(payload[name]) != tree_hash(actual):
            raise RuntimeError("restored state differs: " + name)
    return completed


def run(args):
    import torch
    from tools import train_target_dino as engine
    from tools.sar_oct01_extension import tree_hash
    from tcmmsline3.native24_oof_dino import build_model, configure_train_stage, parameter_groups
    from tools.native24_fivefold_storage import save
    engine.atomic_checkpoint = save
    config = json.loads(Path(args.config).read_text())
    identity = check_binding(config, engine)
    if torch.cuda.device_count() != 1:
        raise ValueError("expose exactly one allocated GPU with CUDA_VISIBLE_DEVICES")
    if config.get("gpu_uuid") and os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]:
        raise ValueError("allocated GPU UUID changed")
    os.sched_setaffinity(0, set(config["cpu_affinity"]))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    # Fixed algorithms plus saved dropout RNG make interrupted steps repeatable.
    torch.backends.cudnn.deterministic = True
    output = Path(config["output_dir"]) / ("smoke" if args.smoke else "train")
    recovery = output / "last.pt"
    if args.resume and not recovery.exists():
        raise FileNotFoundError(recovery)
    if not args.resume and any((output / name).exists() for name in ("last.pt", "progress.json", "started.json", "completed.json")):
        raise FileExistsError("existing run requires --resume")
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError("--max-updates is a positive absolute update limit")
    output.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_handler)
    reference = json.loads(Path(config["reference_config_file"]).read_text())
    engine.seed_everything(config["seed"])
    # Fresh runs load only public backbone weights; resume loads its exact state.
    model = build_model(config, load_pretrained=not args.resume).to("cuda")
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise RuntimeError("unexpected batch-dependent normalization")
    optimizer = torch.optim.AdamW(parameter_groups(model, config), lr=2e-4, weight_decay=.05)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    ema = engine.TensorEMA(model)
    completed, restored_plan_hash = 0, None
    if args.resume:
        payload = torch.load(recovery, map_location="cpu", weights_only=False, mmap=True)
        completed = restore_recovery(payload, model, optimizer, ema, config, identity, engine, tree_hash)
        restored_plan_hash = payload["sampler"]["last_executed_plan_sha256"]
        del payload
        gc.collect()
    else:
        engine.seed_everything(101337)
    engine.atomic_json(output / "started.json", dict(schema=SCHEMA, binding_sha256=identity,
        config=config, pid=os.getpid(), time_unix=time.time(), resumed=args.resume,
        resumed_updates=completed, initialization="exact_same_fold_recovery" if args.resume else "public_pretrained",
        train_scene_ids_sha256=scene_hash(config["train_scene_ids"]),
        held_scene_ids_sha256=scene_hash(config["held_scene_ids"]),
        official30_evaluation=False, held_scene_evaluation=False,
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda))
    full_target = config["epochs"] * config["updates_per_epoch"]
    target = 3 if args.smoke else full_target
    if args.max_updates is not None:
        target = min(target, args.max_updates)
    loaded_epoch, iterator, previous_stage = None, None, None
    plan_hash = restored_plan_hash
    times, audits, started, start_updates = [], [], time.time(), completed
    last_saved = completed if args.resume else -1
    while completed < target:
        epoch = completed // config["updates_per_epoch"]
        stage_epoch = (0 if completed == 0 else 5) if args.smoke else epoch
        stage_key = config["variant"] == "v2l" or stage_epoch >= 5
        changed = stage_key != previous_stage
        if changed:
            stage = configure_train_stage(model, stage_epoch)
            model.train()
            previous_stage = stage_key
            engine.atomic_json(output / f"stage_{stage_epoch:03d}.json", stage)
        if loaded_epoch != epoch:
            if iterator is not None:
                del iterator, dl, dataset
                gc.collect()
            # Loader uses an independent generator; it cannot consume model RNG.
            dataset, dl, plan_hash = make_loader(config, reference, epoch, completed, output, engine)
            iterator, loaded_epoch = iter(dl), epoch
            if args.resume and completed == start_updates and completed % 480 and plan_hash != restored_plan_hash:
                raise RuntimeError("resume regenerated a different in-progress plan")
        before = time.monotonic()
        audit = args.smoke or changed or completed == start_updates
        result = train_update(model, optimizer, iterator, config, completed, engine, audit)
        if audit:
            details = result["gradient_audit"]
            if bool(stage_key) != (details["upper_backbone_gradient"] > 0):
                raise RuntimeError("upper-backbone gradient disagrees with freeze stage")
            audits.append(details)
        ema.update(model)
        completed += 1
        torch.cuda.synchronize()
        seconds = time.monotonic() - before
        times.append(seconds)
        if completed <= start_updates + 3 or completed % 10 == 0 or STOP:
            row = dict(status="smoke" if args.smoke else "training", variant=config["variant"],
                fold_id=config["fold_id"], pid=os.getpid(), time_unix=time.time(),
                completed_updates=completed, epoch=completed // 480, update_in_epoch=completed % 480,
                target_updates=full_target, run_limit_updates=target, step_seconds=seconds,
                recent_mean_seconds=sum(times[-20:]) / len(times[-20:]),
                elapsed_seconds=time.time()-started,
                gpu_peak_gib=torch.cuda.max_memory_allocated() / 1024**3, **result)
            engine.atomic_json(output / "progress.json", row)
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
        if completed % 480 == 0 or completed % int(config.get("recovery_interval_updates", 240)) == 0 or STOP:
            save_recovery(recovery, model, optimizer, ema, config, identity, completed, plan_hash, engine)
            last_saved = completed
        if STOP:
            break
    if completed != last_saved:
        save_recovery(recovery, model, optimizer, ema, config, identity, completed, plan_hash, engine)
    if iterator is not None:
        del iterator, dl, dataset
        gc.collect()
    if args.smoke:
        expected = {"model": tree_hash(model.state_dict()), "optimizer": tree_hash(optimizer.state_dict()),
                    "ema": tree_hash(ema.state), "rng": tree_hash(engine.rng_state())}
        payload = torch.load(recovery, map_location="cpu", weights_only=False, mmap=True)
        restore_recovery(payload, model, optimizer, ema, config, identity, engine, tree_hash)
        actual = {"model": tree_hash(model.state_dict()), "optimizer": tree_hash(optimizer.state_dict()),
                  "ema": tree_hash(ema.state), "rng": tree_hash(engine.rng_state())}
        if expected != actual:
            raise RuntimeError("smoke recovery roundtrip changed state")
        engine.atomic_json(output / "result.json", dict(status="passed" if completed == target and not STOP else "paused",
            binding_sha256=identity, fold_id=config["fold_id"], updates=completed,
            gradient_audits=audits, step_seconds=times, recovery_roundtrip=True,
            gpu_peak_gib=torch.cuda.max_memory_allocated() / 1024**3,
            smoke_does_not_initialize_formal_run=True))
    elif completed == full_target:
        candidate = output / "checkpoints" / f"ema_epoch{config['epochs']:03d}.pt"
        with checkpoint_lock(config):
            payload = dict(schema=SCHEMA, role="candidate_ema", binding_sha256=identity,
                config=config, fold_id=config["fold_id"], epoch=config["epochs"], updates=completed,
                num_classes=24, label_contract="semantic0..23; raw0_and_padding_ignore255",
                train_scene_ids_sha256=scene_hash(config["train_scene_ids"]),
                held_scene_ids_sha256=scene_hash(config["held_scene_ids"]),
                state_dict={k: v.detach().cpu() for k, v in ema.state.items()})
            engine.atomic_checkpoint(candidate, payload,
                reserve_bytes=int(config.get("checkpoint_reserve_bytes", 2 * 1024**3)))
            del payload
            saved = torch.load(candidate, map_location="cpu", weights_only=False, mmap=True)
            model.load_state_dict(saved["state_dict"], strict=True)
            if tree_hash(model.state_dict()) != tree_hash(saved["state_dict"]):
                raise RuntimeError("final EMA failed strict inference loading")
            del saved
            digest = engine.sha256(candidate)
        receipt = dict(status="completed", binding_sha256=identity, fold_id=config["fold_id"],
            epochs=config["epochs"], updates=completed, ema_file=str(candidate), ema_sha256=digest,
            ema_strict_load_verified=True, time_unix=time.time(), automatic_evaluation_started=False)
        engine.atomic_json(candidate.with_suffix(".json"), receipt)
        engine.atomic_json(output / "completed.json", receipt)
    else:
        engine.atomic_json(output / "paused.json", dict(status="paused", updates=completed,
            reason="signal" if STOP else "max_updates", binding_sha256=identity, time_unix=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-updates", type=int, help="absolute successful-update limit; save a resumable state")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        try:
            config = json.loads(Path(args.config).read_text())
            output = Path(config["output_dir"]) / ("smoke" if args.smoke else "train")
            output.mkdir(parents=True, exist_ok=True)
            (output / "failure.json").write_text(json.dumps(dict(time_unix=time.time(),
                traceback=traceback.format_exc()), indent=2))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
