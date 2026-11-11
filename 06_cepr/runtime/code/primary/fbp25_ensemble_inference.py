"""Authenticated full120 upgrade inference, retaining the native rolling protocol.

The runner owns scene/label provenance and CPU/GPU placement.  This adapter reads
no labels, reconstructs the selected EMA, and writes FP32 full-scene probabilities.
All imports of training/model code are delayed until after the runner selects the
authenticated training source.  SAR-off preserves every learned optical/context
component and calls the same checkpoint's ``forward_optical``.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = "fbp25_third_eval_inference_v1_20261010"
BASE_INPUT_KEYS = ("ms", "sar_db", "sar_valid", "optical_valid")
CONTEXT_INPUT_KEYS = ("ms_context", "context_valid", "local_box_context")
VARIANTS = ("joint", "pyramid", "paired")


def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_candidate(payload, training, spec):
    """Reject smoke/raw/recovery, different schedules and substituted variants."""
    from tools import train_convnext25_full120 as train

    identity = train.validate_config(training, devices=False)
    epoch = spec.get("epoch")
    _require(type(epoch) is int and epoch in training["checkpoint_epochs"],
             "Candidate epoch is not an authenticated training milestone")
    expected = dict(schema=train.SCHEMA, role="candidate_ema_delta", num_classes=25,
                    variant=training["variant"], config_sha256=identity,
                    epoch=epoch, updates=epoch * training["updates_per_epoch"],
                    pretraining_identity=train.pretraining_identity(training),
                    convnext_checkpoint_sha256=training["convnext_checkpoint_sha256"],
                    croma_checkpoint_sha256=training["files"][training["croma_checkpoint"]])
    _require(spec.get("variant") == training["variant"], "Requested variant differs")
    for key, value in expected.items():
        _require(payload.get(key) == value, "Candidate provenance differs: " + key)
    for key in ("config_sha256", "training_identity"):
        if key in spec:
            _require(spec[key] == identity, "Requested training identity differs")
    frozen = payload.get("frozen_hash")
    _require(isinstance(frozen, str) and len(frozen) == 64,
             "Candidate lacks a frozen-state hash")
    if "frozen_hash" in spec:
        _require(spec["frozen_hash"] == frozen, "Requested frozen-state hash differs")
    _require(isinstance(payload.get("state_dict"), dict)
             and isinstance(payload.get("mutable_keys"), list)
             and len(payload["mutable_keys"]) == len(set(payload["mutable_keys"]))
             and set(payload["mutable_keys"]) == set(payload["state_dict"]),
             "Candidate EMA/state inventory differs")
    return identity


def verify_inference_assets(training):
    """Check the reconstruction closure, without rereading training image rasters.

    Scene image/SAR bindings are checked separately by the evaluation runner.
    Source bytes, both source entry points, both pretraining weights and the
    normalization manifest are authenticated here for each fresh worker.
    """
    required = {training[key] for key in
                ("convnext_checkpoint", "convnext_source", "croma_checkpoint", "croma_source")}
    required.add(training["data"]["raw_manifest"])
    required.update(path for path in training["files"] if Path(path).suffix == ".py")
    for path in sorted(required):
        _require(path in training["files"] and sha256(path) == training["files"][path],
                 "Bound inference source/asset changed: " + path)
    metadata = json.loads(Path(training["data"]["raw_manifest"]).read_text(encoding="utf-8"))
    _require("normalization" in metadata, "Training manifest lacks normalization")
    return metadata["normalization"]


def _verify_runtime_sources(training, model, train):
    """A redirected import must actually use the authenticated implementation."""
    import sys

    paths = {str(Path(inspect.getfile(type(model))).resolve()), str(Path(train.__file__).resolve())}
    # Include already imported transitive project modules; a same-name module
    # cached from an older run must never bypass source authentication.
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and name.startswith(("tcmmsline3.", "tools.")):
            path = str(Path(file).resolve())
            relative = name.replace(".", "/") + ".py"
            matches = [bound for bound in training["files"]
                       if str(bound).replace("\\", "/").endswith("/" + relative)]
            if path in training["files"] or matches:
                paths.add(path)
    for path in sorted(paths):
        _require(path in training["files"] and sha256(path) == training["files"][path],
                 "Loaded implementation is outside the authenticated training source: " + path)
    return {path: training["files"][path] for path in sorted(paths)}


def load_candidate(spec, device="cuda"):
    """Return ``(model, provenance)`` for one SHA-bound candidate EMA.

    Required spec fields: path, sha256, epoch, variant, training_config and
    training_config_sha256.  Optional config_sha256/training_identity/frozen_hash
    are additional checks, never substitutes for payload validation.
    """
    import torch
    from tools import train_convnext25_full120 as train

    for key in ("path", "sha256", "training_config", "training_config_sha256"):
        _require(bool(spec.get(key)), "Missing candidate binding: " + key)
    _require(sha256(spec["training_config"]) == spec["training_config_sha256"],
             "Training config bytes changed")
    training = json.loads(Path(spec["training_config"]).read_text(encoding="utf-8"))
    train.validate_config(training, devices=False)
    _require(sha256(spec["path"]) == spec["sha256"], "Candidate checkpoint bytes changed")
    payload = torch.load(spec["path"], map_location="cpu", weights_only=True, mmap=True)
    identity = validate_candidate(payload, training, spec)
    normalization = verify_inference_assets(training)
    # The seed reconstructs any frozen randomly initialized state exactly.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(training["seed"])
        model = train.make_model(training, device="cpu")
    sources = _verify_runtime_sources(training, model, train)
    mutable = tuple(model.mutable_keys())
    _require(set(mutable) == set(payload["mutable_keys"]), "Reconstructed mutable inventory differs")
    frozen = tuple(key for key in model.state_dict() if key not in set(mutable))
    _require(train.state_hash(train.subset(model, frozen)) == payload["frozen_hash"],
             "Candidate frozen reconstruction differs")
    train.apply_delta(model, payload["state_dict"])
    _require(train.state_hash(train.subset(model, frozen)) == payload["frozen_hash"],
             "Applying EMA changed immutable state")
    model.set_stage("joint")
    model.to(device).eval().requires_grad_(False)
    receipt = dict(schema=SCHEMA, variant=training["variant"], epoch=spec["epoch"],
                   updates=payload["updates"], checkpoint_path=spec["path"],
                   checkpoint_sha256=spec["sha256"], training_identity=identity,
                   training_config_sha256=spec["training_config_sha256"],
                   frozen_hash=payload["frozen_hash"], source_sha256=sources,
                   adapter_sha256=sha256(__file__), normalization=normalization,
                   state_sha256=train.state_hash(model.state_dict()),
                   pretraining_identity=payload["pretraining_identity"])
    return model, receipt


def context_batch(image, coords, normalization):
    """Exact training context geometry/normalization, without labels or TTA.

    Training's random flips/spectral augmentation are disabled for evaluation.
    Geometry, uint8 per-band bilinear resize, nearest validity and masking match
    read_context_views before augmentation; no whole-scene thumbnail is needed.
    """
    from PIL import Image
    from tcmmsline3.target_ms75_data import extract_boundless, resize_raw
    from tcmmsline3.target_dino_data import normalize_bgrn

    _require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 4,
             "Context requires the original uint8 BGRN scene")
    _require(bool(coords), "Empty context batch")
    views, masks = [], []
    for top, left in coords:
        _require(min(top, left) >= 0 and top + 512 <= image.shape[0] and left + 512 <= image.shape[1],
                 "Local context crop must lie inside its original scene")
        raw, valid = extract_boundless(image, top=top - 256, left=left - 256, size=1024)
        raw = resize_raw(raw, 512)
        valid = np.asarray(Image.fromarray(valid.astype(np.uint8)).resize(
            (512, 512), Image.Resampling.NEAREST)) > 0
        views.append(normalize_bgrn(raw, normalization) * valid[None])
        masks.append(valid[None])
    return (np.stack(views), np.stack(masks),
            np.tile(np.array([.25, .25, .75, .75], np.float32), (len(coords), 1)))


def batch_inputs(cache, coords, *, variant, image, normalization, device, context_cache=None):
    import torch

    _require(variant in VARIANTS, "Unknown candidate variant")
    arrays = tuple(cache.batch(coords))
    _require(len(arrays) == 4, "Expected native MS/SAR/support cache")
    if variant == "context":
        extra = context_batch(image, coords, normalization) if context_cache is None else tuple(context_cache.batch(coords))
        _require(len(extra) == 3, "Expected context/support/box cache")
        arrays += tuple(extra)
    keys = BASE_INPUT_KEYS + (CONTEXT_INPUT_KEYS if variant == "context" else ())
    device = torch.device(device)
    result = {}
    for key, value in zip(keys, arrays):
        tensor = torch.from_numpy(np.array(value, copy=True, order="C"))
        if device.type == "cuda":
            tensor = tensor.pin_memory()
        result[key] = tensor.to(device, non_blocking=device.type == "cuda")
    expected = {"ms": ((len(coords), 4, 512, 512), torch.float32),
                "sar_db": ((len(coords), 2, 166, 166), torch.float32),
                "sar_valid": ((len(coords), 1, 166, 166), torch.bool),
                "optical_valid": ((len(coords), 1, 512, 512), torch.bool)}
    if variant == "context":
        expected.update(ms_context=((len(coords), 4, 512, 512), torch.float32),
                        context_valid=((len(coords), 1, 512, 512), torch.bool),
                        local_box_context=((len(coords), 4), torch.float32))
    for key, (shape, dtype) in expected.items():
        _require(tuple(result[key].shape) == shape and result[key].dtype == dtype,
                 "Native input shape/dtype differs: " + key)
    return result


def forward_logits(model, inputs, sar_mode):
    """The off arm bypasses SAR, including paired training's auxiliary path."""
    _require(sar_mode in ("on", "off"), "Choose SAR on or off explicitly")
    output = model(**inputs) if sar_mode == "on" else model.forward_optical(**inputs)
    return output["logits"]


def _state_token(model):
    return (tuple((key, value._version, value.data_ptr())
                  for key, value in model.state_dict(keep_vars=True).items()),
            tuple(module.training for module in model.modules()),
            tuple(parameter.requires_grad for parameter in model.parameters()))


def verify_controls(model, inputs):
    """Exact zero/missing/off parity with all context inputs retained."""
    import torch
    from tools.train_ms75_sar_joint import state_hash

    _require(not any(module.training for module in model.modules()), "Controls require eval mode")
    before, token = state_hash(model.state_dict()), _state_token(model)
    absent = dict(inputs, sar_valid=torch.zeros_like(inputs["sar_valid"]),
                  optical_valid=torch.zeros_like(inputs["optical_valid"]))
    device = inputs["ms"].device
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                               enabled=device.type == "cuda", cache_enabled=False):
        reference = forward_logits(model, inputs, "off")
        missing = forward_logits(model, absent, "on")
        zero = model(**inputs, sar_mode="zero")["logits"]
        missing_zero = forward_logits(model, dict(absent, sar_db=torch.zeros_like(inputs["sar_db"])), "on")
        real = forward_logits(model, inputs, "on")
    for name, value in (("missing", missing), ("zero", zero), ("missing_zero", missing_zero)):
        _require(torch.equal(value, reference), name + " differs from same-model optical output")
    _require(bool(torch.isfinite(real).all() and torch.isfinite(reference).all()), "Nonfinite control logits")
    _require(_state_token(model) == token and state_hash(model.state_dict()) == before,
             "Inference controls changed model state")
    difference = (real.float() - reference.float()).abs()
    return dict(missing_equals_off=True, zero_equals_off=True, missing_zero_equals_off=True,
                state_unchanged=True, real_off_max_abs_logit_difference=float(difference.max()),
                real_off_mean_abs_logit_difference=float(difference.mean()))


def predict_loaded(model, image, output, cache, *, variant, normalization, sar_mode,
                   report=None, smoke=False, device="cuda", batch_size=4, context_cache=None):
    """Native FP32 rolling Hann stitch then softmax, writing a CxHxW buffer."""
    import torch
    from tools.evaluate_ms75_rebuild import blend_window, sliding_starts

    _require(sar_mode in ("on", "off"), "Choose SAR on or off explicitly")
    _require(batch_size == 4, "Preserve the historical batch4 evaluation arithmetic")
    _require(variant in VARIANTS, "Unknown candidate variant")
    _require(not any(m.training for m in model.modules())
             and not any(p.requires_grad for p in model.parameters()), "Inference requires a frozen eval model")
    h, w = image.shape[:2]
    _require(image.dtype == np.uint8 and image.shape == (h, w, 4) and min(h, w) >= 512,
             "Expected an in-bounds native BGRN scene")
    _require(output.shape == (25, h, w) and output.dtype == np.float32 and output.flags.writeable,
             "Full output must be writable float32 CxHxW")
    device, token = torch.device(device), _state_token(model)
    ys, xs = sliding_starts(h), sliding_starts(w)
    report = report or (lambda **unused: None)
    checks = {}
    if smoke:
        first = batch_inputs(cache, [(ys[0], xs[0])], variant=variant, image=image,
                             normalization=normalization, device=device, context_cache=context_cache)
        checks = verify_controls(model, first)
        from tcmmsline3.target_dino_data import normalize_bgrn
        expected_ms = normalize_bgrn(image[:512, :512], normalization)
        _require(np.array_equal(first["ms"][0].cpu().numpy(), expected_ms),
                 "Input cache differs from native optical normalization")
        checks["native_ms_input_exact"] = True
        if variant == "context":
            reference = context_batch(image, [(ys[0], xs[0])], normalization)
            for key, value in zip(CONTEXT_INPUT_KEYS, reference):
                _require(np.array_equal(first[key].cpu().numpy(), value),
                         "Context cache differs from native context geometry: " + key)
            checks["native_context_input_exact"] = True
        del first
    weight = torch.from_numpy(blend_window(512)).to(device)
    buf = torch.zeros((25, 512, w), device=device, dtype=torch.float32)
    weights = torch.zeros((512, w), device=device, dtype=torch.float32)
    written, windows = 0, 0
    with torch.inference_mode():
        for yi, top in enumerate(ys):
            for offset in range(0, len(xs), batch_size):
                xx = xs[offset:offset + batch_size]
                coords = [(top, left) for left in xx]
                inputs = batch_inputs(cache, coords, variant=variant, image=image,
                                      normalization=normalization, device=device, context_cache=context_cache)
                with torch.autocast(device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda", cache_enabled=False):
                    logits = forward_logits(model, inputs, sar_mode).float()
                _require(tuple(logits.shape) == (len(xx), 25, 512, 512)
                         and bool(torch.isfinite(logits).all()), "Invalid candidate logits")
                for index, left in enumerate(xx):
                    buf[:, :, left:left + 512].add_(logits[index] * weight[None])
                    weights[:, left:left + 512].add_(weight)
                windows += len(coords)
                del inputs, logits
            rows = ys[yi + 1] - top if yi + 1 < len(ys) else 512
            _require(bool((weights[:rows] > 0).all()), "Uncovered native output")
            for left in range(0, rows, 32):
                right = min(left + 32, rows)
                probability = (buf[:, left:right] / weights[None, left:right]).softmax(0)
                output[:, top + left:top + right] = probability.cpu().numpy()
                written += (right - left) * w
            rest = 512 - rows
            for value in (buf, weights):
                if rest:
                    value[..., :rest, :].copy_(value[..., rows:, :].clone())
                value[..., rest:, :].zero_()
            report(stage="predict", rows=top + rows, total_rows=h, sar_mode=sar_mode)
    _require(written == h * w and _state_token(model) == token,
             "Incomplete output or mutated model state")
    return dict(checks=checks, pixels_written=written, windows=windows, sar_mode=sar_mode,
                precision=("BF16" if device.type == "cuda" else "FP32")
                          + " batch4 forward; native Torch FP32 Hann512/256 logit stitch then softmax",
                state_unchanged=True, missing_sar="same_checkpoint_optical_path",
                context_source_size=1024 if variant == "context" else None,
                context_input_size=512 if variant == "context" else None,
                augmentation="none; same training normalization and deterministic geometry")


def predict_model(spec, image, output, scene, cache, normalization, *, sar_mode=None,
                  report=None, smoke=False, device="cuda", batch_size=4, context_cache=None):
    """Runner API: load one expert, fill full probabilities, return provenance."""
    started = time.monotonic()
    mode = spec.get("sar_mode") if sar_mode is None else sar_mode
    _require(mode in ("on", "off"), "Candidate must explicitly request SAR on/off")
    model, receipt = load_candidate(spec, device=device)
    _require(normalization == receipt["normalization"], "Evaluation normalization differs from training")
    result = predict_loaded(model, image, output, cache, variant=spec["variant"],
                            normalization=normalization, sar_mode=mode, report=report,
                            smoke=smoke, device=device, batch_size=batch_size, context_cache=context_cache)
    if smoke:
        from tools.train_ms75_sar_joint import state_hash
        _require(state_hash(model.state_dict()) == receipt["state_sha256"],
                 "Whole-scene inference changed loaded checkpoint state")
    return dict(receipt, **result, scene=scene, seconds=time.monotonic() - started)


def expert_infer(c, name, image, output, scene, cache, report=None, smoke=False, *, sar_mode=None):
    """Small compatibility entry point for a runner with c['experts'][name]."""
    spec = dict(c["experts"][name])
    return predict_model(spec, image, output, scene, cache, c["normalization"],
                         sar_mode=sar_mode, report=report, smoke=smoke,
                         device=c.get("device", "cuda"))


def predict_dino(spec, image, output, cache, normalization, report=None, smoke=False):
    """Load the authenticated 25-class EMA with its original training builder."""
    import torch
    from tools import train_fbp25_dino as train
    from tcmmsline3.fbp25_dino import build_model
    _require(sha256(spec['training_config']) == spec['training_config_sha256'], 'DINO config changed')
    config = json.loads(Path(spec['training_config']).read_text())
    identity = train.check_binding(config)
    _require(sha256(spec['path']) == spec['sha256'], 'DINO checkpoint changed')
    payload = torch.load(spec['path'], map_location='cpu', weights_only=True, mmap=True)
    for key, value in dict(schema=config['schema'], role='candidate_ema', binding_sha256=identity,
                           epoch=spec['epoch'], updates=spec['epoch']*600, num_classes=25,
                           config=config).items():
        _require(payload.get(key) == value, 'DINO payload differs: '+key)
    reference = json.loads(Path(config['reference_config_file']).read_text())
    _require(reference['data']['normalization']['ms'] == normalization, 'DINO normalization differs')
    model = build_model(config, load_pretrained=False)
    model.load_state_dict(payload['state_dict'], strict=True)
    _require(all(not x.is_floating_point() or bool(torch.isfinite(x).all())
                 for x in model.state_dict().values()), 'Nonfinite DINO state')
    model.requires_grad_(False).eval().cuda()
    del payload
    class Optical(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner=inner
        def forward_optical(self, **inputs):
            return self.inner(inputs['ms'])
    wrapper = Optical(model).requires_grad_(False).eval()
    result = predict_loaded(wrapper, image, output, cache, variant='joint',
                            normalization=normalization, sar_mode='off', report=report, smoke=False)
    if smoke:
        # An independent single-window forward must match the stitched 512 crop.
        # The full multi-window case is checked across both physical GPU pairs.
        result['checks']['classifier_channels']=model.num_classes
    return dict(result, checkpoint_sha256=spec['sha256'], training_identity=identity,
                epoch=spec['epoch'], family=config['variant'], num_classes=25)
