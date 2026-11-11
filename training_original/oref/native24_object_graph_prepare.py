"""Prepare only the released OOF120/replay120 graphQS training jobs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

FIXED = {'schema': 'oref_cepr_train120_release_v1', 'seed': 261016, 'versions': ['replay120', 'oof120'], 'arms': ['graphQS'], 'batch_size': 4096, 'cq_ce_steps': 120000, 'cq_soft_steps': 6000, 'cq_ce_checkpoints': [8000, 24000, 60000, 120000], 'cq_soft_checkpoints': [3000, 6000], 'checkpoints': ['CE_8000', 'CE_24000', 'CE_60000', 'CE_120000', 'CE_softIoU_3000', 'CE_softIoU_6000'], 'normalization_samples': 200000, 'sampling': 'Historical uniform valid disagreement groups with sqrt(valid-mass) CE weights', 'soft_start_selection': 'same_pool_C_calibrated_best_CE', 'soft_optimizer_reset': True, 'soft_sampler_reset': False, 'sar_auxiliary_weight': 0.1, 'I_dynamic': True, 'I_max_iterations': 12, 'I_group_cap': 8192, 'I_without_C_start': 'raw', 'I_with_C_start': 'C', 'graph_ground_truth_allowed': False, 'graph_candidate_order': [2, 4, 6, 3, 5, 0, 1], 'official_scores_select_120_models': False}
VERSIONS = FIXED["versions"]
ARMS = FIXED["arms"]
CHECKPOINTS = FIXED["checkpoints"]


def build_config(root, source_config, replay_config):
    train = list(source_config["train_ids"])
    official = list(source_config["official_ids"])
    c = copy.deepcopy(FIXED)
    c.update(output=str(root), source=replay_config["source"],
             controlled=source_config["controlled"], replay_root=replay_config["replay_root"],
             replay_store=replay_config["replay_store"], runtime=source_config["runtime"],
             python=replay_config["python"], resources=source_config["resources"],
             acceleration_dir=str(Path(replay_config["source"]) / "acceleration"),
             graph_store=str(Path(root) / "graphs"), replay_sar_store=str(Path(root) / "replay_sar"),
             cpu_threads=4, gpu_memory_cap_gib=8, calibration_workers=2,
             cq_calibration_min_available_gib=22,
             train120_ids=train, train_ids=train, official30_ids=official,
             report_only_folds={str(f):list(source_config["splits"][str(f)]["validation"]) for f in range(5)},
             pools={v:dict(fit_ids=train, calibration_ids=train, selection_ids=train,
                          official_truth_allowed=False,
                          input_domain="replay" if v == "replay120" else "oof") for v in VERSIONS})
    validate_config(c)
    return c


def validate_config(c):
    for key, value in FIXED.items():
        if c.get(key) != value:
            raise ValueError("Release protocol mismatch: " + key)
    train, official = c["train120_ids"], c["official30_ids"]
    if len(train) != 120 or len(set(train)) != 120 or len(official) != 30 or len(set(official)) != 30:
        raise ValueError("Expected original disjoint 120/30 scene IDs")
    if set(train) & set(official) or c["train_ids"] != train:
        raise ValueError("Training/target identity mismatch")
    if set(c["pools"]) != set(VERSIONS):
        raise ValueError("Only the two released training pools are allowed")
    for version in VERSIONS:
        pool = c["pools"][version]
        for role in ("fit_ids", "calibration_ids", "selection_ids"):
            if pool[role] != train:
                raise ValueError("Only train120 may be used for " + role)
        if pool["official_truth_allowed"] is not False:
            raise ValueError("Target truth is forbidden during fitting")
        if pool["input_domain"] != ("replay" if version == "replay120" else "oof"):
            raise ValueError("Training input provenance mismatch")
    groups = c["report_only_folds"]
    if set(groups) != set(map(str, range(5))) or any(len(groups[str(f)]) != 24 for f in range(5)):
        raise ValueError("Expected five original 24-scene groups")
    ids = [sid for f in range(5) for sid in groups[str(f)]]
    if len(set(ids)) != 120 or set(ids) != set(train):
        raise ValueError("Reporting groups must partition train120")
    return dict(status="passed", training_jobs=2, updates_per_job=126000,
                official_truth_used_for_training=False)


def seal(root, source, replay, graph_settings=None):
    root = Path(root)
    if (root / "config.json").exists() or (root / "preflight.json").exists():
        raise FileExistsError("Do not overwrite a sealed experiment")
    source_config = read(Path(source) / "config.json")
    replay_config = read(Path(replay) / "config.json")
    c = build_config(root, source_config, replay_config)
    if graph_settings:
        c["graph"] = read(graph_settings)
    for name in ("code", "jobs", "graphs", "calibration_slots"):
        (root / name).mkdir(parents=True, exist_ok=True)
    # Bind the code that will actually run, not an external historical checkout.
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.suffix not in (".py", ".cpp"):
            continue
        dst = root / "code" / p.name
        if dst.exists() and sha(dst) != sha(p):
            raise ValueError("Existing experiment source differs: " + str(dst))
        if not dst.exists():
            shutil.copyfile(p, dst)
    write(root / "config.json", c)
    refs = [Path(source) / "config.json", Path(replay) / "config.json",
            Path(c["replay_store"]) / "index.json", Path(source_config["store"]) / "index.json"]
    write(root / "preflight.json", dict(**validate_config(c),
        config_sha256=sha(root / "config.json"),
        code_sha256={p.name:sha(p) for p in sorted((root / "code").iterdir()) if p.suffix in (".py", ".cpp")},
        source_artifacts={str(p):sha(p) for p in refs}, created_unix=time.time()))
    return c


def verify(c):
    validate_config(c)
    root = Path(c["output"])
    receipt = read(root / "preflight.json")
    if sha(root / "config.json") != receipt["config_sha256"]:
        raise ValueError("Sealed config changed")
    for name, digest in receipt["code_sha256"].items():
        if sha(root / "code" / name) != digest or sha(Path(__file__).parent / name) != digest:
            raise ValueError("Sealed or executing source changed: " + name)
    for path, digest in receipt["source_artifacts"].items():
        if sha(path) != digest:
            raise ValueError("Source artifact changed: " + path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--graph-settings")
    args = p.parse_args()
    c = seal(args.root, args.source, args.replay, args.graph_settings)
    print(json.dumps(dict(status="prepared", root=c["output"], training_jobs=2)))
