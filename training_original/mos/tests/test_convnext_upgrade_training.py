"""CPU contracts for full120 upgrades and truly disjoint paired optimization."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from tools import train_convnext_upgrade_full120 as train
from tcmmsline3.convnext_paired_model import ConvNeXtPairedSAR


def load_fixture(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load_fixture("test_convnext_sar_model")


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(variant="paired"):
    c = load_fixture("test_train_convnext_sar").config()
    c.update(copy.deepcopy(train.FIXED), variant=variant,
             physical_gpus=train.VARIANT_GPUS[variant],
             gpu_uuids=[train.GPU_UUID_BY_INDEX[i] for i in train.VARIANT_GPUS[variant]])
    c["data"].update(updates_per_epoch=600, scope="full120")
    return c


def model():
    torch.manual_seed(44)
    return ConvNeXtPairedSAR(dict(decoder_channels=16, spectral_channels=[4, 4, 8, 12, 16],
        sar_interaction_width=8, sar_intensity_channels=4), load_pretrained=False,
        backbone=fixture.TinyBackbone(), sar_encoder=fixture.TinyCroma())


def activate_sar(m):
    with torch.no_grad():
        for interaction in m.interactions:
            interaction.output.weight.normal_(std=.1)


def test_three_variants_fixed_full120_schedule_uuid_and_storage():
    for variant in train.VARIANT_GPUS:
        c = config(variant)
        c.update(checkpoint_dir="recovery-volume", candidate_dir="candidate-volume")
        assert train.validate_config(c, devices=False)
        assert c["checkpoint_epochs"] == [4, 6] + list(range(8, 21)) + [22, 24, 27, 30]
        assert len(c["checkpoint_epochs"]) == 19
        assert train.stage_for_update(1199, c) == "optical_warmup"
        assert train.stage_for_update(1200, c) == "joint"
        assert train.lr_factor(0, c) == 1 / 600
        assert train.lr_factor(17999, c) < .001
        for field, value in [("epochs", 24), ("scheduler_epochs", 32), ("warmup_epochs", 0),
                             ("world_size", 4), ("gpu_uuids", ["wrong", "wrong"]),
                             ("candidate_dir", ""), ("official_test_auto", True)]:
            bad = copy.deepcopy(c)
            bad[field] = value
            with pytest.raises(ValueError):
                train.validate_config(bad, devices=False)


def test_paired_one_optical_encoding_and_exact_deployed_outputs():
    m = model().set_stage("joint").eval()
    activate_sar(m)
    batch = fixture.batch()
    counts = {"backbone": 0, "spectral": 0}
    def hook(name):
        def count(*unused):
            counts[name] += 1
        return count
    handles = [m.backbone.register_forward_hook(hook("backbone")),
               m.spectral.register_forward_hook(hook("spectral"))]
    with torch.no_grad():
        paired = m.forward_paired(**batch)
    for handle in handles:
        handle.remove()
    assert counts == {"backbone": 1, "spectral": 1}
    with torch.no_grad():
        ordinary = m(**batch)
        optical = m.forward_optical(**batch)
    torch.testing.assert_close(paired["on"]["logits"], ordinary["logits"], atol=0, rtol=0)
    torch.testing.assert_close(paired["off"]["logits"], optical["logits"], atol=0, rtol=0)
    assert "optical_aux_logits" not in paired["on"] and "spectral_aux_logits" not in paired["on"]


def test_on_loss_has_no_optical_gradient_but_reaches_all_sar_content_paths():
    m = model().set_stage("joint").eval()
    activate_sar(m)
    inputs = fixture.batch()
    inputs["ms"].requires_grad_(True)
    paired = m.forward_paired(**inputs)
    # The main prediction alone must differentiate through the detached-parameter decoder.
    paired["on"]["logits"].square().mean().backward()
    groups = m.gradient_parameter_sets()
    assert inputs["ms"].grad is None
    assert all(p.grad is None for p in groups["optical"])
    for prefix in ("sar_encoder.tail", "sar_intensity_encoder", "interactions.0.query", "interactions.1.query"):
        gradients = [p.grad for n, p in m.named_parameters() if n.startswith(prefix)]
        assert any(g is not None and g.abs().sum() > 0 for g in gradients), prefix
    assert all(p.grad is None for p in m.decoder.parameters())
    assert all(p.grad is None for p in m.sar_aux_head.parameters())


def test_off_loss_has_no_sar_gradient_and_optical_is_not_frozen_in_joint():
    m = model().set_stage("joint").eval()
    activate_sar(m)
    paired = m.forward_paired(**fixture.batch())
    loss = sum(paired["off"][key].square().mean()
               for key in ("logits", "optical_aux_logits", "spectral_aux_logits"))
    loss.backward()
    assert all(p.grad is None for p in m.gradient_parameter_sets()["sar"])
    for module in (m.backbone, m.spectral, m.decoder, m.optical_aux_head, m.spectral_aux_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_warmup_skips_sar_and_paired_missing_and_zero_are_exact_optical():
    m = model()
    x = fixture.batch()
    assert m.forward_paired(**x)["on"] is None
    m.set_stage("joint").eval()
    activate_sar(m)
    with torch.no_grad():
        m.sar_intensity_encoder[0].bias.fill_(2.)
        paired = m.forward_paired(**x, sar_mode="zero")
        torch.testing.assert_close(paired["on"]["logits"], paired["off"]["logits"], atol=0, rtol=0)
        x["sar_db"].fill_(float("nan"))
        x["sar_valid"].zero_()
        x["optical_valid"].zero_()
        paired = m.forward_paired(**x)
        assert not paired["sar_present"].any()
        torch.testing.assert_close(paired["on"]["logits"], paired["off"]["logits"], atol=0, rtol=0)


def test_missing_sar_adds_no_on_loss_and_microbatch_weighting_is_preserved(monkeypatch):
    # A controlled differentiable per-image objective isolates reduction/masking
    # from the independent, already-tested CE/Dice/Lovasz implementation.
    def fake_loss(output, batch, cfg, stage):
        x = output["logits"]
        valid = batch["label"] != 255
        per = (x.square() * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
        value = per.mean()
        return value, {"main": value.detach()}
    monkeypatch.setattr(train, "compute_losses", fake_loss)
    raw_off = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 5
    raw_on = raw_off + 2
    labels = torch.zeros(3, 2, 2, dtype=torch.long)
    present = torch.tensor([True, False, True])
    whole_off = raw_off.clone().requires_grad_()
    whole_on = raw_on.clone().requires_grad_()
    total, parts = train.compute_paired_losses(
        {"off": {"logits": whole_off}, "on": {"logits": whole_on}, "sar_present": present},
        {"label": labels}, config(), "joint")
    total.backward()
    assert whole_on.grad[1].count_nonzero() == 0
    assert whole_off.grad[1].abs().sum() > 0
    off = raw_off.clone().requires_grad_()
    on = raw_on.clone().requires_grad_()
    accumulated = 0
    for a, b in ((0, 1), (1, 3)):
        value, _ = train.compute_paired_losses(
            {"off": {"logits": off[a:b]}, "on": {"logits": on[a:b]}, "sar_present": present[a:b]},
            {"label": labels[a:b]}, config(), "joint")
        value = value * (b - a) / 3
        value.backward()
        accumulated += value.detach()
    torch.testing.assert_close(total, accumulated)
    torch.testing.assert_close(whole_off.grad, off.grad)
    torch.testing.assert_close(whole_on.grad, on.grad)
    assert all(not v.requires_grad for v in parts.values())


def test_paired_separate_clipping_does_not_suppress_other_branch():
    m = model().set_stage("joint")
    groups = m.gradient_parameter_sets()
    a, b = groups["optical"][0], groups["sar"][0]
    a.grad = torch.full_like(a, 1000.)
    b.grad = torch.full_like(b, .00001)
    before = b.grad.clone()
    norms = train.clip_parameter_groups(m, config())
    assert norms["optical_gradient_norm"] > 1 and norms["sar_gradient_norm"] < 1
    assert torch.linalg.vector_norm(a.grad) <= 1.000001
    torch.testing.assert_close(b.grad, before, atol=0, rtol=0)


def test_aliases_and_mutable_inventory_survive_paired_architecture():
    m = model()
    groups = m.parameter_groups()
    train.validate_optimizer_inventory(m, groups)
    values = train.subset(m, m.mutable_keys())
    train.validate_state_aliases(m, values)
    assert torch.equal(values["backbone.native.norm.weight"], values["backbone.native.norms.3.weight"])
    bad = copy.deepcopy(values)
    bad["backbone.native.norms.3.weight"].add_(1)
    with pytest.raises(ValueError, match="conflict"):
        train.apply_delta(m, bad)


def test_rolling_recovery_retains_previous_and_survives_failed_new_write(tmp_path, monkeypatch):
    def save(path, payload):
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    monkeypatch.setattr(train.engine, "atomic_checkpoint", save)
    train.write_rolling_recovery(tmp_path, {"updates": 300})
    train.write_rolling_recovery(tmp_path, {"updates": 600})
    assert torch.load(tmp_path / "last.pt", weights_only=True)["updates"] == 600
    assert torch.load(tmp_path / "previous.pt", weights_only=True)["updates"] == 300
    def fail(path, payload):
        raise RuntimeError("disk full")
    monkeypatch.setattr(train.engine, "atomic_checkpoint", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        train.write_rolling_recovery(tmp_path, {"updates": 900})
    assert torch.load(tmp_path / "last.pt", weights_only=True)["updates"] == 600
    assert torch.load(tmp_path / "previous.pt", weights_only=True)["updates"] == 600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last.pt", "previous.pt"]
