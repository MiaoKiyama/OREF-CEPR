"""Offline model contracts; tiny injected encoders avoid real asset/GPU use."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from tcmmsline3.convnext_sar_model import ConvNeXtSAR, OpticalBackbone, fixed_sar_intensity


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dims = [4, 8, 12, 16]
        self.downsample_layers = nn.ModuleList()
        current = 3
        for i, channels in enumerate(self.embed_dims):
            kernel = 4 if i == 0 else 2
            self.downsample_layers.append(nn.Sequential(nn.Conv2d(current, channels, kernel, kernel)))
            current = channels
        self.stages = nn.ModuleList(nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU()) for c in self.embed_dims)
        self.norm = nn.LayerNorm(16)
        self.norms = nn.ModuleList([nn.Identity(),nn.Identity(),nn.Identity(),self.norm])


class TinyCroma(nn.Module):
    channels = (8,8,8,8)
    provenance = {"test_only": True}

    def __init__(self):
        super().__init__()
        self.prefix = nn.Conv2d(2,8,1).requires_grad_(False)
        self.tail = nn.Sequential(nn.Conv2d(8,8,1),nn.GELU(),nn.Dropout(.3),nn.Conv2d(8,8,1))

    def train(self, mode=True):
        super().train(mode)
        self.prefix.eval()
        return self

    def forward(self, db, valid, aux_mode="real"):
        with torch.no_grad():
            x = torch.where(valid, db, torch.zeros_like(db)) if aux_mode == "real" else torch.zeros_like(db)
            x = self.prefix(F.avg_pool2d(F.pad(x,(1,1,1,1)),8))
        observed = F.avg_pool2d(F.pad(valid.float(),(1,1,1,1)),8)>0
        y = self.tail(x) * observed
        return (y,)*4


def model():
    return ConvNeXtSAR(dict(decoder_channels=16,spectral_channels=[4,4,8,12,16],
                           sar_interaction_width=8,sar_intensity_channels=4),
                      load_pretrained=False,backbone=TinyBackbone(),sar_encoder=TinyCroma())


def batch():
    return dict(ms=torch.randn(1,4,512,512),sar_db=torch.randn(1,2,166,166)*4-15,
                sar_valid=torch.ones(1,1,166,166,dtype=torch.bool),
                optical_valid=torch.ones(1,1,512,512,dtype=torch.bool))


class ConvNeXtSARTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(301337)

    def test_rgb_input_reverses_bgr_and_uses_imagenet_stats_without_nir(self):
        m=model()
        raw=torch.tensor([.1,.2,.3,.8]).view(1,4,1,1)
        ms=(raw-m.ms_mean)/m.ms_std
        expected=(raw[:,[2,1,0]]-m.rgb_mean)/m.rgb_std
        torch.testing.assert_close(m.rgb_input(ms),expected)
        changed=ms.clone();changed[:,3]+=17
        torch.testing.assert_close(m.rgb_input(changed),expected)

    def test_native_backbone_scales_and_last_norm_match_reference(self):
        native=TinyBackbone().eval();wrapped=OpticalBackbone(native).eval()
        rgb=torch.randn(1,3,512,512)
        with torch.no_grad():
            features=wrapped(rgb)
            value=rgb
            for down,stage in zip(native.downsample_layers,native.stages):
                value=stage(down(value))
            expected=native.norm(value.flatten(2).transpose(1,2)).transpose(1,2).reshape_as(value)
        self.assertEqual([x.shape[-2:] for x in features],[(128,128),(64,64),(32,32),(16,16)])
        torch.testing.assert_close(features[-1],expected,atol=0,rtol=0)

    def test_checkpointing_preserves_backbone_values_and_gradients(self):
        a=OpticalBackbone(TinyBackbone(),False).train()
        b=copy.deepcopy(a);b.gradient_checkpointing=True
        rgb=torch.randn(1,3,128,128)
        ya=a(rgb);yb=b(rgb)
        for x,y in zip(ya,yb):
            torch.testing.assert_close(x,y,atol=0,rtol=0)
        sum(x.square().mean() for x in ya).backward()
        sum(x.square().mean() for x in yb).backward()
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(na,nb);torch.testing.assert_close(pa.grad,pb.grad,atol=0,rtol=0)

    def test_initial_joint_logits_exactly_match_current_optical_path(self):
        m=model().set_stage("joint").eval();x=batch()
        with torch.no_grad():
            expected=m.forward_optical(**x);actual=m(**x)
        for key in ("logits","optical_aux_logits","spectral_aux_logits"):
            torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
        self.assertEqual(actual["logits"].shape,(1,24,512,512))
        self.assertEqual(actual["sar_aux_logits"].shape,(1,24,21,21))
        self.assertEqual(actual["sar_stride16_delta"].shape[-2:],(32,32))
        self.assertEqual(actual["sar_stride32_delta"].shape[-2:],(16,16))
        self.assertEqual(actual["sar_stride16_delta"].count_nonzero(),0)
        self.assertEqual(actual["sar_stride32_delta"].count_nonzero(),0)

    def test_each_semantic_sar_insertion_can_change_main_prediction(self):
        m=model().set_stage("joint").eval();x=batch()
        with torch.no_grad():
            baseline=m.forward_optical(**x)["logits"]
            for index in (0,1):
                for layer in m.interactions:layer.output.weight.zero_()
                m.interactions[index].output.weight.normal_(std=.1)
                changed=m(**x)["logits"]
                self.assertGreater(float((changed-baseline).abs().max()),1e-5)

    def test_missing_sar_remains_exact_current_optical_after_sar_weights_change(self):
        m=model().set_stage("joint").eval();x=batch()
        x["sar_valid"].zero_();x["optical_valid"].zero_();x["sar_db"].fill_(float("nan"))
        with torch.no_grad():
            for layer in m.interactions:layer.output.weight.normal_(std=.1)
            m.decoder.classifier.bias.add_(.23)
            expected=m.forward_optical(**x)["logits"];actual=m(**x)
        torch.testing.assert_close(actual["logits"],expected,atol=0,rtol=0)
        self.assertFalse(bool(actual["sar_present"].any()))
        self.assertEqual(actual["sar_intensity_feature"].count_nonzero(),0)

    def test_zero_control_disables_croma_and_intensity_even_with_nonzero_bias(self):
        m=model().set_stage("joint").eval();x=batch()
        with torch.no_grad():
            for layer in m.interactions:layer.output.weight.normal_(std=.1)
            m.sar_intensity_encoder[0].bias.fill_(1.)
            expected=m.forward_optical(**x)["logits"];actual=m(**x,sar_mode="zero")
        torch.testing.assert_close(actual["logits"],expected,atol=0,rtol=0)
        self.assertEqual(actual["sar_intensity_feature"].count_nonzero(),0)
        self.assertEqual(actual["sar_content_difference"].count_nonzero(),0)

    def test_fixed_intensity_retains_absolute_db_and_polarization_difference(self):
        valid=torch.ones(1,1,166,166,dtype=torch.bool)
        a=torch.tensor([-25.,-30.]).view(1,2,1,1).expand(1,2,166,166)
        b=torch.tensor([-5.,-10.]).view(1,2,1,1).expand_as(a)
        xa,xb=fixed_sar_intensity(a,valid),fixed_sar_intensity(b,valid)
        torch.testing.assert_close(xa[0,:,20,20],torch.tensor([-.4,-.6,1/6]))
        self.assertGreater(float((xa[:,:2]-xb[:,:2]).abs().max()),.79)
        torch.testing.assert_close(xa[:,2],xb[:,2])
        changed=a.clone();changed[:,1]+=3
        self.assertGreater(float((fixed_sar_intensity(changed,valid)[:,2]-xa[:,2]).abs().max()),.09)

    def test_invalid_db_is_zero_and_cannot_contaminate_intensity(self):
        valid=torch.ones(1,1,166,166,dtype=torch.bool);valid[:,:,:5]=False
        db=torch.full((1,2,166,166),-20.);db[:,:,:5]=float("nan")
        value=fixed_sar_intensity(db,valid)
        self.assertTrue(bool(torch.isfinite(value).all()));self.assertEqual(value[:,:,:5].count_nonzero(),0)
        self.assertEqual(fixed_sar_intensity(db,valid,"zero").count_nonzero(),0)

    def test_warmup_freezes_backbones_and_sar_but_trains_optical_modules(self):
        m=model().train();x=batch()
        self.assertEqual(m.stage,"optical_warmup")
        self.assertFalse(any(p.requires_grad for p in m.backbone.parameters()))
        self.assertFalse(any(p.requires_grad for p in m.sar_encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in m.interactions.parameters()))
        out=m(**x)
        (out["logits"].square().mean()+out["optical_aux_logits"].square().mean()+out["spectral_aux_logits"].square().mean()).backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in m.decoder.parameters()))
        self.assertFalse(any(p.grad is not None for p in m.backbone.parameters()))
        self.assertEqual(out["sar_aux_valid"].count_nonzero(),0)

    def test_eval_executes_sar_even_if_stage_was_warmup(self):
        m=model().eval();x=batch()
        with torch.no_grad():
            m.interactions[0].output.weight.normal_(std=.1)
            expected=m.forward_optical(**x)["logits"];actual=m(**x)["logits"]
        self.assertGreater(float((expected-actual).abs().max()),1e-5)

    def test_joint_unfreezes_all_backbone_and_only_croma_tail(self):
        m=model().set_stage("joint").train();x=batch()
        self.assertTrue(all(p.requires_grad for p in m.backbone.parameters()))
        self.assertFalse(any(p.requires_grad for p in m.sar_encoder.prefix.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.sar_encoder.tail.parameters()))
        with torch.no_grad():
            for layer in m.interactions:layer.output.weight.normal_(std=.05)
        out=m(**x)
        (out["logits"].square().mean()+.1*out["sar_aux_logits"].square().mean()).backward()
        for module in (m.backbone,m.decoder,m.sar_intensity_encoder,m.sar_encoder.tail):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters()))

    def test_optimizer_inventory_is_same_in_both_stages_and_excludes_frozen_croma(self):
        m=model();a=m.parameter_groups();keys=m.mutable_keys();m.set_stage("joint");b=m.parameter_groups()
        self.assertEqual([g["parameter_names"] for g in a],[g["parameter_names"] for g in b])
        self.assertEqual(keys,m.mutable_keys())
        ids=[id(p) for g in a for p in g["params"]]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in m.parameters() if p.requires_grad})
        self.assertEqual({g["group_name"].rsplit("_decay",1)[0].removesuffix("_no") for g in a},
                         {"optical_backbone","optical_new","sar_new","sar_tail"})
        self.assertNotIn("sar_encoder.prefix.weight",keys)

    def test_mutable_delta_retains_official_norm_alias_and_roundtrips(self):
        m=model();keys=m.mutable_keys()
        self.assertIn("backbone.native.norm.weight",keys)
        self.assertIn("backbone.native.norms.3.weight",keys)
        self.assertIn("ms_mean",keys)
        with torch.no_grad():m.backbone.native.norm.weight.add_(1.)
        delta={k:m.state_dict()[k].clone() for k in keys}
        n=model();missing=n.load_state_dict(delta,strict=False).missing_keys
        self.assertTrue(all(k.startswith("sar_encoder.prefix.") for k in missing))
        torch.testing.assert_close(n.backbone.native.norm.weight,m.backbone.native.norm.weight,atol=0,rtol=0)
        self.assertIs(n.backbone.native.norm.weight,n.backbone.native.norms[3].weight)

    def test_zero_pair_uses_matched_stochastic_realizations(self):
        m=model().set_stage("joint").train();x=batch()
        real,difference=m._encode_pair(x["sar_db"],x["sar_valid"],"zero")
        self.assertEqual(difference.count_nonzero(),0)
        self.assertEqual(real.shape,(1,8,21,21))

    def test_accepts_legacy_views_and_filters_metadata_in_forward_batch(self):
        m=model().eval();x=batch()
        with torch.no_grad():
            a=m(**x,ms_short=torch.zeros(1),local_box_global=torch.zeros(1))["logits"]
            b=m.forward_batch({**x,"scene_id":"example","label":torch.zeros(1)})["logits"]
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        with self.assertRaises(ValueError):m(**x,teacher_probability=torch.zeros(1))


if __name__=="__main__":
    unittest.main()
