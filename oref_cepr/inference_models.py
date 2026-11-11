"""Model construction and objective smoke helpers; no training loop or sampler."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .bootstrap import ROOT, LEGACY

def build_model(kind, pretrained, *, initialize=True):
    pretrained = Path(pretrained).resolve()
    if kind in ('dinov2', 'dinov3'):
        from tcmmsline3.native24_oof_dino import build_model as make
        variant = 'v2l' if kind == 'dinov2' else 'v3l'
        name, digest = {
            'v2l': ('dinov2_vitl14_pretrain.pth', 'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'),
            'v3l': ('dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth', '8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035'),
        }[variant]
        config = dict(classes=24, variant=variant, pretrained_file=str(pretrained/name), pretrained_sha256=digest)
        return make(config, load_pretrained=initialize), config
    if kind != 'mos':
        raise ValueError(kind)
    from tcmmsline3.convnext_pyramid_model import ConvNeXtPyramidSAR
    c = json.loads((ROOT/'configs/fbp/mos.json').read_text())
    c.update(convnext_checkpoint=str(pretrained/'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth'),
             croma_checkpoint=str(pretrained/'CROMA_base_0238d814.pt'),
             convnext_source=str(LEGACY/'vendor/dinov3_convnext/convnext.py'),
             croma_source=str(LEGACY/'vendor/CROMA/use_croma.py'))
    return ConvNeXtPyramidSAR(c, load_pretrained=initialize), c

def dino_loss(logits, labels):
    from tcmmsline3.losses import soft_dice_loss, lovasz_softmax_loss
    value = logits.float().sum() * 0
    for i in range(len(labels)):
        target = labels[i:i+1].clone().long()
        target[target == 255] = -1
        if not (target >= 0).any():
            continue
        z = logits[i:i+1].float()
        value = value + (F.cross_entropy(z,target,ignore_index=-1)
                         + .5*soft_dice_loss(z,target,24)
                         + .25*lovasz_softmax_loss(z,target))/len(labels)
    return value
