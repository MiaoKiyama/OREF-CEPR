"""CPU-only supplemental EMA export for the two sealed 2026-10-15 runs.

Never changes a trainer, config, recovery, native candidate or native receipt.
Recovery inodes are pinned only while cloning EMA to private CPU memory.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import torch

SCHEMA = "native24_extra_candidates_v1"
CPU_AFFINITY = [46, 47, 48, 49, 102, 103, 104, 105]
PROJECT = Path("/archive/hot5/yeqi/projects/native24_retrain_20261015_v1")
JOBS = {
    "pyramid": dict(output="/archive/hot0/yeqi/native24_retrain_20261015_v1/pyramid",
                    native=[10, 11, 12], extra=[4, 6, 8, 9],
                    trainer="train_convnext_upgrade_full120.py",
                    schema="convnext_upgrade_full120_ddp_v1_20261007", role="convnext_upgrade_recovery"),
    "convnext": dict(output="/archive/hot6/yeqi/native24_retrain_20261015_v1/convnext",
                    native=[12, 13, 14], extra=[8, 10],
                    trainer="train_convnext_sar_full120.py",
                    schema="convnext_sar_full120_ddp_v1_20261002", role="convnext_sar_recovery"),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sync_directory(directory):
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def check_process(row):
    proc = Path("/proc") / str(row["rank_pid"])
    stat = (proc / "stat").read_text()
    fields = stat[stat.rfind(")") + 2:].split()
    require(int(fields[19]) == row["rank_start_ticks"] and fields[0] != "Z", "original training rank ended or PID was reused")
    command = (proc / "cmdline").read_bytes().split(b"\0")
    require(str(row["config"]).encode() in command, "training rank command no longer names the sealed config")
    require(any(b"native24_retrain_limited.py" in part for part in command), "unexpected training rank executable")


def prepare_manifest(path):
    path = Path(path).resolve()
    manifest_sha = sha256(path)
    manifest = json.loads(path.read_text())
    require(manifest.get("schema") == SCHEMA and manifest.get("evaluation_enabled") is False, "unexpected manifest/evaluation policy")
    rows = manifest.get("models", [])
    require(sorted(row.get("model_kind") for row in rows) == sorted(JOBS), "manifest must name exactly these two runs")
    for row in rows:
        kind = row["model_kind"]
        job = JOBS[kind]
        output = Path(row["output_dir"])
        require(output == Path(job["output"]) and Path(row["config"]) == output / "config.json", "unrecognized run paths")
        require(row["native_epochs"] == job["native"] and row["extra_epochs"] == job["extra"], "unauthorized candidate inventory")
        require(row["requested_epochs"] == sorted(job["native"] + job["extra"]), "merged epoch inventory differs")
        require(sha256(row["config"]) == row["config_file_sha256"], "sealed config bytes changed")
        config = json.loads(Path(row["config"]).read_text())
        require(canonical_hash(config) == row["config_identity"], "sealed config identity changed")
        require(config["output_dir"] == str(output) and config["checkpoint_epochs"] == job["native"], "native run output/candidates changed")
        require(config["save_every_updates"] == 300 and config["updates_per_epoch"] == 600, "recovery/epoch interval changed")
        require(config["auto_full"] is False and config["official_test_auto"] is False, "automatic evaluation must stay disabled")
        require(type(row["rank_pid"]) is int and type(row["rank_start_ticks"]) is int, "missing bound training process")
        source = PROJECT / kind / "tools" / job["trainer"]
        require(sha256(source) == config["files"].get(str(source)), "historical candidate builder source changed")
        smoke = Path(row["smoke_recovery"])
        require(smoke == output / ("smoke_recovery_" + row["config_identity"][:12]) / "last.pt", "smoke recovery path changed")
        smoke_receipt = json.loads((output / "smoke.json").read_text())
        require(smoke_receipt.get("status") == "passed" and smoke_receipt.get("config_sha256") == row["config_identity"],
                "smoke receipt does not authenticate this config")
        row["_frozen_hash"] = smoke_receipt["frozen_hash"]
        row["_smoke_sha256"] = sha256(output / "smoke.json")
        row["_config"] = config
        row["_source"] = source
        row["_directory"] = output / "extra_candidates"
        row["_directory"].mkdir(exist_ok=True)
        row["_manifest_sha256"] = manifest_sha
    return path, manifest_sha, rows


def validate_recovery(row, payload):
    config, job = row["_config"], JOBS[row["model_kind"]]
    require(payload.get("schema") == job["schema"] and payload.get("role") == job["role"], "unexpected recovery schema/role")
    require(payload.get("config_sha256") == row["config_identity"], "recovery belongs to another config")
    if row["model_kind"] == "pyramid":
        require(payload.get("variant") == "pyramid", "recovery is not Pyramid")
    expected_pretraining = {key: dict(path=config[key], sha256=config["files"][config[key]])
                            for key in ("convnext_checkpoint", "convnext_source", "croma_checkpoint", "croma_source")}
    require(payload.get("pretraining_identity") == expected_pretraining, "recovery pretraining identity differs")
    require(payload.get("frozen_hash") == row["_frozen_hash"], "frozen-state identity differs from sealed smoke")
    require(payload.get("world_size") == 2 and payload.get("optimizer", {}).get("state"), "incomplete optimizer/rank recovery")
    rng = payload.get("rank_rng", [])
    require([r.get("rank") for r in rng] == [0, 1] and all(set(("python", "numpy", "torch", "cuda")) <= set(r) for r in rng), "incomplete rank RNG recovery")
    require(type(payload.get("updates")) is int and payload["updates"] >= 0, "invalid recovery update")
    keys = payload.get("mutable_keys")
    require(isinstance(keys, list) and len(keys) == len(set(keys)), "invalid mutable-key inventory")
    require(set(payload.get("ema", {})) == set(keys) == set(payload.get("model", {})), "incomplete raw/EMA inventory")
    require(payload.get("stage") == ("optical_warmup" if payload["updates"] < 1200 else "joint"), "recovery stage disagrees with update")


def make_candidate(row, payload, epoch):
    """Match the old candidate_payload, cloning only EMA off the mapped inode."""
    config = row["_config"]
    candidate = dict(schema=payload["schema"], role="candidate_ema_delta",
                     config_sha256=payload["config_sha256"], updates=payload["updates"],
                     pretraining_identity=copy.deepcopy(payload["pretraining_identity"]),
                     frozen_hash=payload["frozen_hash"],
                     convnext_checkpoint_sha256=config["convnext_checkpoint_sha256"],
                     croma_checkpoint_sha256=config["files"][config["croma_checkpoint"]],
                     mutable_keys=list(payload["mutable_keys"]),
                     state_dict={key: value.detach().cpu().clone() for key, value in payload["ema"].items()})
    if row["model_kind"] == "pyramid":
        candidate["variant"] = "pyramid"
    if epoch is not None:
        candidate["epoch"] = epoch
    return candidate


def check_original_payload_semantics(row, candidate):
    """Execute only the two sealed original payload helpers on a state-only view."""
    source = ast.parse(row["_source"].read_text())
    selected = [node for node in source.body if isinstance(node, ast.FunctionDef)
                and node.name in ("candidate_payload", "validate_state_aliases", "pretraining_identity")]
    require(len(selected) == 3, "historical candidate payload helpers changed")
    namespace = dict(torch=torch, SCHEMA=candidate["schema"])
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(row["_source"]), "exec"), namespace)

    class StateOnly:
        def mutable_keys(self):
            return candidate["mutable_keys"]

        def state_dict(self, **kwargs):
            return candidate["state_dict"]

    reference = namespace["candidate_payload"](
        StateOnly(), candidate["state_dict"], row["_config"], row["config_identity"],
        candidate["frozen_hash"], role="candidate_ema_delta", updates=candidate["updates"],
        epoch=candidate.get("epoch"))
    require({k: v for k, v in reference.items() if k != "state_dict"}
            == {k: v for k, v in candidate.items() if k != "state_dict"}, "candidate metadata differs from original builder")
    require(reference["state_dict"].keys() == candidate["state_dict"].keys()
            and all(torch.equal(reference["state_dict"][key], value) for key, value in candidate["state_dict"].items()),
            "candidate tensors differ from original builder")


def capture(row, source, wanted, *, self_test=False):
    """Hold a published inode by descriptor only while cloning its wanted EMA."""
    descriptor = None
    payload = None
    candidate = None
    try:
        descriptor = os.open(source, os.O_RDONLY)
        stat = os.fstat(descriptor)
        payload = torch.load(f"/proc/self/fd/{descriptor}", map_location="cpu", weights_only=False, mmap=True)
        validate_recovery(row, payload)
        updates = payload["updates"]
        if not self_test and updates not in wanted:
            return updates, None, None
        epoch = None if self_test else updates // 600
        require(self_test or updates == epoch * 600, "candidate is not at an epoch boundary")
        candidate = make_candidate(row, payload, epoch)
        source_record = dict(path=str(source), updates=updates, device=stat.st_dev,
                             inode=stat.st_ino, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    finally:
        # Drop the multi-GB recovery mapping before hashing or writing ~0.8 GiB.
        # Keeping its inode past the next rotation would retain unwanted storage.
        del payload
        gc.collect()
        if descriptor is not None:
            os.close(descriptor)
    check_original_payload_semantics(row, candidate)
    return updates, candidate, source_record


def save_candidate(path, candidate):
    """Publish a verified complete file exclusively; never replace a candidate."""
    path = Path(path)
    require(not path.exists(), "candidate already exists: " + str(path))
    tensor_bytes = sum(v.numel() * v.element_size() for v in candidate["state_dict"].values())
    require(shutil.disk_usage(path.parent).free >= tensor_bytes + 64 * 1024**2, "insufficient actual space for candidate write")
    expected = state_hash(candidate["state_dict"])
    temporary = path.with_name(".candidate_" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(candidate, stream)
            stream.flush()
            os.fsync(stream.fileno())
        restored = torch.load(temporary, map_location="cpu", weights_only=False, mmap=True)
        require({k: v for k, v in restored.items() if k != "state_dict"}
                == {k: v for k, v in candidate.items() if k != "state_dict"}, "serialized candidate metadata differs")
        require(state_hash(restored["state_dict"]) == expected, "serialized EMA tensor hash differs")
        del restored
        gc.collect()
        file_sha = sha256(temporary)
        os.link(temporary, path)
        sync_directory(path.parent)
        return dict(path=str(path), sha256=file_sha, ema_state_sha256=expected, bytes=path.stat().st_size)
    finally:
        temporary.unlink(missing_ok=True)


def inventory(row):
    found = {}
    for epoch in row["extra_epochs"]:
        path = row["_directory"] / f"epoch{epoch:03d}_ema_delta.pt"
        receipt = path.with_suffix(".json")
        if not path.exists():
            require(not receipt.exists(), "candidate receipt exists without candidate")
            continue
        require(receipt.exists(), "candidate lacks its supplemental provenance receipt; manual audit required")
        record = json.loads(receipt.read_text())
        require(record.get("manifest_sha256") == row["_manifest_sha256"]
                and record.get("config_identity") == row["config_identity"]
                and record.get("epoch") == epoch and record.get("updates") == epoch * 600
                and record.get("sha256") == sha256(path), "existing supplemental candidate/receipt changed")
        found[epoch] = record
    return found


def record_candidate(row, candidate, source_record):
    epoch = candidate["epoch"]
    path = row["_directory"] / f"epoch{epoch:03d}_ema_delta.pt"
    record = save_candidate(path, candidate)
    record.update(schema=SCHEMA, manifest_sha256=row["_manifest_sha256"],
                  config_file_sha256=row["config_file_sha256"], config_identity=row["config_identity"],
                  model_kind=row["model_kind"], epoch=epoch, updates=epoch * 600,
                  source_recovery=source_record, original_candidate_payload_parity=True,
                  all_ema_tensors_roundtrip_verified=True, created_unix=time.time())
    write_json(path.with_suffix(".json"), record, exclusive=True)
    print(json.dumps(dict(event="extra_candidate_saved", model_kind=row["model_kind"], epoch=epoch, path=str(path))), flush=True)
    return record


def scan(row, found):
    require(sha256(row["config"]) == row["config_file_sha256"], "sealed training config changed")
    require(sha256(Path(row["output_dir"]) / "smoke.json") == row["_smoke_sha256"], "sealed smoke receipt changed")
    pending = sorted(set(row["extra_epochs"]) - set(found))
    if not pending:
        return
    check_process(row)
    highest = -1
    # A previous checkpoint can rescue a daemon restart for Pyramid.
    sources = [Path(row["output_dir"]) / "last.pt"]
    if row["model_kind"] == "pyramid":
        sources.append(Path(row["output_dir"]) / "previous.pt")
    for source in sources:
        if not source.exists():
            continue
        wanted = {epoch * 600 for epoch in pending}
        updates, candidate, provenance = capture(row, source, wanted)
        highest = max(highest, updates)
        if candidate is not None:
            found[candidate["epoch"]] = record_candidate(row, candidate, provenance)
            pending.remove(candidate["epoch"])
            del candidate
            gc.collect()
    if pending and highest > pending[0] * 600:
        raise RuntimeError(f"missed exact {row['model_kind']} epoch {pending[0]}; newest recovery update={highest}")


def report(row, found):
    complete = sorted(found) == row["extra_epochs"]
    result = dict(schema=SCHEMA, status="extra_candidates_completed" if complete else "waiting_for_exact_recovery",
                  manifest_sha256=row["_manifest_sha256"], config_identity=row["config_identity"],
                  model_kind=row["model_kind"], extra_epochs_saved=sorted(found),
                  extra_epochs_pending=sorted(set(row["extra_epochs"]) - set(found)),
                  requested_epochs=row["requested_epochs"], native_epochs=row["native_epochs"],
                  native_candidates_managed_by_original_trainer=True,
                  merged_expected_candidates=[dict(epoch=e, path=str(Path(row["output_dir"]) /
                      ("extra_candidates" if e in row["extra_epochs"] else "candidates") / f"epoch{e:03d}_ema_delta.pt"))
                      for e in row["requested_epochs"]],
                  extra_candidates=[found[e] for e in sorted(found)],
                  evaluation_started=False, time_unix=time.time())
    write_json(row["_directory"] / "progress.json", result)
    completed = row["_directory"] / "completed.json"
    if complete and not completed.exists():
        write_json(completed, result, exclusive=True)
    return complete


def self_test(rows):
    for row in rows:
        updates, candidate, source = capture(row, Path(row["smoke_recovery"]), set(), self_test=True)
        require(updates == 3 and "epoch" not in candidate, "expected the original three-update smoke recovery")
        path = row["_directory"] / (".selftest_" + uuid.uuid4().hex + ".pt")
        try:
            record = save_candidate(path, candidate)
            require(record["ema_state_sha256"] == state_hash(candidate["state_dict"]), "self-test EMA mismatch")
            print(json.dumps(dict(event="self_test_passed", model_kind=row["model_kind"], updates=updates,
                                  original_candidate_payload_parity=True, **record)), flush=True)
        finally:
            del candidate
            gc.collect()
            path.unlink(missing_ok=True)


def run(args):
    path, manifest_sha, rows = prepare_manifest(args.manifest)
    if args.self_test:
        self_test(rows)
        return
    found = {row["model_kind"]: inventory(row) for row in rows}
    try:
        while True:
            require(sha256(path) == manifest_sha, "supplemental manifest changed")
            for row in rows:
                scan(row, found[row["model_kind"]])
            done = [report(row, found[row["model_kind"]]) for row in rows]
            if all(done) or args.once:
                return
            time.sleep(15)
    except Exception as error:
        for row in rows:
            write_json(row["_directory"] / "failed.json", dict(schema=SCHEMA, status="failed", error=str(error),
                       manifest_sha256=manifest_sha, evaluation_started=False, time_unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(CPU_AFFINITY))
    import fcntl
    lock_path = Path(args.manifest).resolve().with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
