import numpy as np
import torch
from torch import nn
WIDTH=313
class SevenNet(nn.Module):
    def __init__(self, kind, mean, scale, mask=7):
        super().__init__()
        self.kind, self.mask = kind, mask
        self.base = base_model(WIDTH)
        self.register_buffer('mean', torch.as_tensor(mean))
        self.register_buffer('scale', torch.as_tensor(scale))
        if kind == 'attention':
            self.token = nn.Sequential(nn.Linear(94, 64), nn.SiLU(), nn.Linear(64, 64))
            self.attention = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0)
            self.norm = nn.LayerNorm(64)
            self.residual = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 24))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        self.sar_effect = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x, aux=False):
        hidden = self.base[:-1](x)
        logits = self.base[-1](hidden)
        effect_feature = hidden
        if self.kind == 'attention':
            raw = x*self.scale+self.mean
            q = x[:, 189:207]*(raw[:, 198:207].sum(1)>0)[:, None]
            s = x[:, 207:214]*(raw[:, 213]>0)[:, None]
            pp = x[:, 214:263].reshape(-1, 7, 7).transpose(1, 2)
            known = x[:, 263:312].reshape(-1, 7, 7).transpose(1, 2)
            rawp = raw[:, 214:263].reshape(-1, 7, 7)
            paired = torch.stack((rawp[:, 5]-rawp[:, 3], rawp[:, 6]-rawp[:, 4]), -1)
            if not self.mask & 4:
                paired = torch.zeros_like(paired)
            common = torch.cat((x[:, 168:189], q, s, x[:, 312:313]), 1)[:, None, :].expand(-1, 7, -1)
            identity = torch.eye(7, device=x.device)[None].expand(len(x), -1, -1)
            token = self.token(torch.cat((x[:, :168].reshape(-1, 7, 24), pp, known, common, paired, identity), 2))
            attended, _ = self.attention(token, token, token, need_weights=False)
            token = self.norm(token+attended)
            logits = logits+self.residual(torch.cat((token.mean(1), token.max(1).values), 1))
            effect_feature = token[:, 2]-token[:, 1]
        if aux:
            return logits, self.sar_effect(effect_feature).squeeze(1)
        return logits

def base_model(width):
    return nn.Sequential(nn.Linear(width,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,24))
