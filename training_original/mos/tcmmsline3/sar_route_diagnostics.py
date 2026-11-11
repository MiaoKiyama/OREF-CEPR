"""Small conditional probes, not replacement segmentation architectures.

All probes preserve the fixed optical log probabilities outside SAR support.
The decomposition is a train-only ridge predictability diagnostic, not DeCUR.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def shifted_pair(sar, mask):
    """Feature-level same-crop mismatch; identical support for both conditions."""
    wrong = torch.roll(sar, shifts=(11, 13), dims=(-2, -1))
    shared = mask * torch.roll(mask, shifts=(11, 13), dims=(-2, -1))
    return wrong, shared


def sample_delta(delta):
    # Original512 pixel centres 2,6,...,510; these are NOT a resized label map.
    axis = (torch.arange(128, device=delta.device, dtype=torch.float32)*4+2.5)/512*2-1
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    return F.grid_sample(delta.float(), torch.stack((xx, yy), -1)[None].expand(len(delta),-1,-1,-1),
                         mode='bilinear', padding_mode='border', align_corners=False)


class AffinityProbe(nn.Module):
    """SAR may change local attention weights; all values remain optical."""
    def __init__(self, channels=128, width=32):
        super().__init__()
        self.query = nn.Conv2d(channels*2, width, 1)
        self.key = nn.Conv2d(channels*2, width, 1)
        self.value = nn.Conv2d(channels, width, 1)
        self.output = nn.Conv2d(width, 24, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, optical, guide, mask):
        joined = torch.cat((optical, guide), 1)
        q, k, v = self.query(joined), self.key(joined), self.value(optical)
        b, c, h, w = q.shape
        keys = F.unfold(k, 3, padding=1).reshape(b,c,9,h,w)
        values = F.unfold(v, 3, padding=1).reshape(b,c,9,h,w)
        eligible = F.unfold(mask,3,padding=1).reshape(b,9,h,w) > .999
        # Always retain self as a safe softmax fallback; query mask zeros output.
        eligible[:,4] = True
        score = (q[:,:,None]*keys).sum(1)/math.sqrt(c)
        weights = score.masked_fill(~eligible,-1e4).softmax(1)
        aggregate = (values*weights[:,None]).sum(2)
        return self.output(aggregate-v)*mask


class ConditionalProbe(nn.Module):
    """Capacity-matched optical/raw/shared/private conditional readouts."""
    def __init__(self, channels=128, width=64):
        super().__init__()
        self.hidden = nn.Conv2d(channels*3,width,1)
        self.output = nn.Conv2d(width,24,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,optical,first,second,mask):
        return self.output(F.gelu(self.hidden(torch.cat((optical,first,second),1))))*mask


def ridge_fit(optical, sar, mask, ridge=.01):
    """Streaming caller supplies supported train tokens only. No labels/held."""
    if optical.ndim != 2 or sar.shape != optical.shape or mask.shape != (len(optical),):
        raise ValueError('Expected matching NxC and token support')
    x, y = optical[mask].double(), sar[mask].double()
    if len(x) < x.shape[1]+1:
        raise ValueError('Insufficient supported train tokens')
    x = torch.cat((x,torch.ones_like(x[:,:1])),1)
    gram, rhs = x.T@x/len(x), x.T@y/len(x)
    penalty = torch.eye(gram.shape[0],dtype=gram.dtype,device=gram.device)*ridge
    penalty[-1,-1] = 0
    return torch.linalg.solve(gram+penalty,rhs).float()


def common_private(optical,sar,weights):
    common = F.conv2d(optical,weights[:-1].T[:,:,None,None],weights[-1])
    return common, sar-common


def gradient_geometry(main, auxiliary, weight=.1):
    """Same-parameter geometry, with unused entries rejected, not padded."""
    if len(main) != len(auxiliary) or not main or any(x is None for x in (*main,*auxiliary)):
        raise ValueError('Both gradients must refer to exactly the same used parameters')
    if any(a.shape != b.shape for a,b in zip(main,auxiliary)):
        raise ValueError('Gradient parameter shapes differ')
    a2 = sum(x.double().square().sum() for x in main)
    b2 = sum(x.double().square().sum() for x in auxiliary)
    dot = sum((a.double()*b.double()).sum() for a,b in zip(main,auxiliary))
    if not bool(torch.isfinite(a2+b2+dot)):
        raise FloatingPointError('Nonfinite gradient')
    aa, bb, ab = float(a2),float(b2),float(dot)
    return dict(main_norm=math.sqrt(aa),auxiliary_norm=math.sqrt(bb),dot=ab,
        cosine=ab/math.sqrt(aa*bb) if aa>0 and bb>0 else None,
        weighted_norm_ratio=weight*math.sqrt(bb/aa) if aa>0 else None,
        combined_main_projection_ratio=1+weight*ab/aa if aa>0 else None)


def summarize_confusion(cm):
    value = cm.double()
    union = value.sum(0)+value.sum(1)-value.diag()
    iou = value.diag()/union.clamp_min(1)
    return dict(miou_percent=float(iou[union>0].mean()*100) if bool((union>0).any()) else None,
        per_class_iou_percent=[float(iou[i]*100) if union[i]>0 else None for i in range(24)],
        represented_classes=[i for i in range(24) if union[i]>0],valid_pixels=int(cm.sum()),
        confusion_matrix=cm.cpu().tolist())
