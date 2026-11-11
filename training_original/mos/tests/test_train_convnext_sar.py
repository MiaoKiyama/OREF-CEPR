"""Direction3 trainer contracts without GPU or official pretraining assets."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from tools import train_convnext_sar as trainer


def config():
    value = copy.deepcopy(trainer.FIXED)
    value.update(gpu_uuids=list(trainer.GPU_UUIDS), micro_batch=4,
        num_workers=3, torch_threads=2, save_every_updates=240,
        convnext_checkpoint="convnext.pt", convnext_source="convnext.py",
        convnext_checkpoint_sha256=trainer.CONVNEXT_SHA256, convnext_source_sha256="a"*64,
        croma_checkpoint="croma.pt", croma_source="croma.py", output_dir="output",
        data=dict(raw_manifest="raw.json", views_cache="views.json", boundary_support="boundary.json",
                  seed=1337, updates_per_epoch=480))
    value["files"] = {path:"pin" for path in ("croma.pt","croma.py","raw.json","views.json","boundary.json")}
    value["files"].update({"convnext.pt":trainer.CONVNEXT_SHA256,"convnext.py":"a"*64})
    return value


def test_fixed_protocol_binds_pretraining_physical_gpus_and_data_budget():
    original = config()
    assert trainer.validate_config(original, devices=False)
    for key, value in (("schema","ms75_sar_joint_train96_ddp_v1"),("physical_gpus",[4,5]),
                       ("num_workers",4),("epochs",12),("warmup_epochs",1),
                       ("convnext_checkpoint_sha256","b"*64),("micro_batch",3)):
        bad = copy.deepcopy(original); bad[key] = value
        with pytest.raises(ValueError):
            trainer.validate_config(bad, devices=False)
    for key, value in (("seed",7),("updates_per_epoch",600),("views_cache","other.json")):
        bad = copy.deepcopy(original); bad["data"][key] = value
        with pytest.raises(ValueError):
            trainer.validate_config(bad, devices=False)
    # A baseline/DINO path is deliberately unnecessary for constructing the
    # training configuration; these are never used for its initialization.
    assert "base_checkpoint" not in original and "teachers" not in original


def test_stage_schedule_warms_optical_heads_for_two_epochs_and_fixed_candidates():
    cfg = config()
    assert trainer.stage_for_update(959,cfg) == "optical_warmup"
    assert trainer.stage_for_update(960,cfg) == "joint"
    assert trainer.lr_factor(0,cfg) == 1/480
    assert trainer.lr_factor(479,cfg) == 1
    assert trainer.lr_factor(480,cfg) == 1
    assert 0 < trainer.lr_factor(15359,cfg) < .001
    assert cfg["checkpoint_epochs"] == [4,8,16,24,32] and trainer.SMOKE_UPDATES == 3


def test_main_losses_are_per_image_and_ignore_sar_support_without_erasing_gt():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        generator = torch.Generator().manual_seed(80)
        labels = torch.randint(24,(3,9,11),generator=generator)
        labels[0,:3] = 255; labels[2].fill_(255)
        raw = torch.randn(3,24,9,11,generator=generator)
        batch = dict(label=labels, optical_valid=torch.zeros(3,1,9,11,dtype=torch.bool))
        whole = raw.clone().requires_grad_()
        total, pieces = trainer.compute_losses({"logits":whole}, batch, config(), "optical_warmup")
        total.backward()
        partial = raw.clone().requires_grad_()
        accumulated = 0
        for start, stop in ((0,1),(1,3)):
            term, _ = trainer.compute_losses({"logits":partial[start:stop]},
                {key:value[start:stop] for key,value in batch.items()}, config(), "optical_warmup")
            term = term*(stop-start)/3
            term.backward(); accumulated = accumulated+term.detach()
        torch.testing.assert_close(total,accumulated)
        torch.testing.assert_close(whole.grad,partial.grad)
        assert whole.grad[1].abs().sum() > 0  # no SAR still has full own-GT gradient
        assert whole.grad[2].abs().sum() == 0
        assert pieces["sar_aux"] == 0 and all(not value.requires_grad for value in pieces.values())
    finally:
        torch.set_num_threads(old)


class AliasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.norms = nn.ModuleList([self.norm])
        self.register_buffer("counter",torch.tensor(0))

    def mutable_keys(self):
        return tuple(sorted(self.state_dict()))


def test_delta_and_ema_preserve_shared_official_norm_aliases():
    model = AliasModel()
    values = trainer.subset(model,model.mutable_keys())
    trainer.apply_delta(model,values)
    assert model.norm.weight is model.norms[0].weight
    bad = copy.deepcopy(values); bad["norms.0.weight"].add_(1)
    with pytest.raises(ValueError,match="conflict"):
        trainer.apply_delta(model,bad)
    bad = dict(values); bad.pop("norm.weight")
    with pytest.raises(ValueError,match="omits"):
        trainer.apply_delta(model,bad)
    with torch.no_grad():
        model.norm.weight.add_(1)
    trainer.update_ema(model,model.mutable_keys(),values,.999,0)
    trainer.validate_state_aliases(model,values)
    assert torch.equal(values["norm.weight"],values["norms.0.weight"])
    # The optimizer inventory uses unique Parameter objects, not duplicate
    # state-dict aliases.
    trainer.validate_optimizer_inventory(model,[dict(params=list(model.parameters()))])


def test_recovery_and_candidate_identities_reject_d1_or_another_pretraining():
    cfg, model = config(), AliasModel()
    optimizer = torch.optim.AdamW(model.parameters())
    values = trainer.subset(model,model.mutable_keys())
    payload = trainer.checkpoint_payload(model,model.mutable_keys(),values,optimizer,cfg,
        "identity","frozen",960,[{"rank":0},{"rank":1}])
    trainer.validate_recovery(payload,model,cfg,"identity","frozen")
    assert payload["stage"] == "joint"
    for key,value in (("schema","ms75_sar_joint_train96_ddp_v1"),("role","joint_recovery"),
                      ("stage","optical_warmup"),("pretraining_identity",{}),
                      ("rank_rng",[{"rank":1},{"rank":0}])):
        bad = dict(payload); bad[key] = value
        with pytest.raises(ValueError):
            trainer.validate_recovery(bad,model,cfg,"identity","frozen")
    candidate = trainer.candidate_payload(model,values,cfg,"identity","frozen",
        role="candidate_ema_delta",updates=1920,epoch=4)
    assert candidate["epoch"] == 4 and candidate["updates"] == 1920
    assert candidate["convnext_checkpoint_sha256"] == trainer.CONVNEXT_SHA256
    assert candidate["pretraining_identity"] == trainer.pretraining_identity(cfg)
    assert "base_sha256" not in candidate and "teacher_frozen_hashes" not in candidate


def test_gradient_validation_rejects_frozen_warmup_or_disconnected_joint():
    valid = dict(optical_backbone_decay=0.,optical_new_decay=1.,sar_new_decay=0.,sar_tail_decay=0.)
    trainer.validate_stage_gradients("optical_warmup",valid)
    with pytest.raises(RuntimeError):
        trainer.validate_stage_gradients("optical_warmup",dict(valid,sar_tail_decay=.1))
    with pytest.raises(RuntimeError):
        trainer.validate_stage_gradients("joint",valid)
    trainer.validate_stage_gradients("joint",{key:1. for key in valid})
    # A legal all-missing-SAR global batch must not abort a stage audit. A
    # separate real-SAR main-gradient smoke still certifies the SAR branches.
    trainer.validate_stage_gradients("joint",dict(valid,optical_backbone_decay=1.), require_sar=False)


def miniature():
    path = Path(__file__).with_name("test_convnext_sar_model.py")
    spec = importlib.util.spec_from_file_location("direction3_model_fixture",path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_real_model_three_update_stages_and_main_only_sar_gradient_audit():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        torch.manual_seed(1337)
        fixture = miniature()
        model, batch, cfg = fixture.model(), fixture.batch(), config()
        batch["label"] = torch.zeros((1,512,512),dtype=torch.long)
        batch["index"] = torch.tensor([0])
        objective = trainer.TrainingObjective(model,cfg)
        optimizer = torch.optim.AdamW(model.parameter_groups(),lr=.001)
        trainer.validate_optimizer_inventory(model,optimizer.param_groups)
        for stage in ("optical_warmup","joint","joint"):
            model.set_stage(stage); model.train()
            optimizer.zero_grad(set_to_none=True)
            loss,pieces = objective(batch)
            assert torch.isfinite(loss)
            loss.backward()
            trainer.validate_stage_gradients(stage,trainer.gradient_summary(model,optimizer))
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            assert all(value.ndim == 0 and not value.requires_grad for value in pieces.values())
        assert trainer.no_sar_parity(model,batch) == 0
        class OneCrop:
            availability = {"scene":True}
            blocks = [("scene","scene",-1)]
            def __len__(self): return 1
            def __getitem__(self,index):
                return {key:(value[0] if isinstance(value,torch.Tensor) else value) for key,value in batch.items()}
        rng = torch.get_rng_state().clone()
        before = trainer.state_hash(model.state_dict())
        audit = trainer.main_gradient_audit(model,OneCrop(),cfg,torch.device("cpu"))
        assert audit["only_main_loss"] and not audit["auxiliary_losses_used"]
        assert all(value>0 for value in audit["required"].values())
        assert torch.equal(torch.get_rng_state(),rng) and model.training
        assert trainer.state_hash(model.state_dict()) == before
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        torch.set_num_threads(old)
