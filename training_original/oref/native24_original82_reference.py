"""Original 82.119 architecture/routing extracted verbatim; only dependency names adapted."""
import numpy as np
import torch
from torch import nn
WIDTH=313
def expected_gains(cm, hist, labels, current):
    n = len(hist)
    rows = np.arange(n)
    oldlabel = labels[rows, current].astype(np.int64)-1
    mass = hist.sum(1)
    diag, union = cm.diagonal(), cm.sum(0)+cm.sum(1)-cm.diagonal()
    tp = hist[rows, oldlabel]
    remove = (diag[oldlabel]-tp)/np.maximum(union[oldlabel]-mass+tp, 1)-diag[oldlabel]/np.maximum(union[oldlabel], 1)
    out = np.zeros(labels.shape, np.float64)
    for action in range(labels.shape[1]):
        new = labels[:, action].astype(np.int64)-1
        tp = hist[rows, new]
        add = (diag[new]+tp)/np.maximum(union[new]+mass-tp, 1)-diag[new]/np.maximum(union[new], 1)
        out[:, action] = (remove+add)/24
        out[new == oldlabel, action] = 0
    return out

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

def fast_calibrate(gains, fixed, counts, labels, eligible):
    """Exact original global threshold search, aggregated by score bins."""
    action=gains.argmax(1).astype(np.uint8);best=gains.max(1)
    values=best[(best>0)&eligible]
    thresholds=np.unique(np.r_[0,np.quantile(values,np.linspace(0,1,21)) if len(values) else [],np.inf])
    bins=np.searchsorted(thresholds,best,side='left');bins[~eligible]=0
    pred=labels[np.arange(len(labels)),action];source=labels[:,2]
    nb=len(thresholds)+1;deltas=np.zeros((nb,25,25),np.int64)
    for y in range(25):
        deltas[:,y,:]=np.bincount(bins*25+pred,weights=counts[:,y],minlength=nb*25).reshape(nb,25).astype(np.int64)
        deltas[:,y,:]-=np.bincount(bins*25+source,weights=counts[:,y],minlength=nb*25).reshape(nb,25).astype(np.int64)
    tails=deltas[::-1].cumsum(0)[::-1]
    base=np.full(len(labels),2,np.uint8);cm=score_route(fixed,counts,labels,base)
    route=base;best_cm=cm;score=miou(cm);chosen=None
    for i,t in enumerate(thresholds):
        candidate=cm+tails[i+1];s=miou(candidate)
        if s>score+1e-12:
            score=s;best_cm=candidate;chosen=float(t)
    if chosen is not None:
        route=base.copy();ok=(best>chosen)&eligible;route[ok]=action[ok]
    assert np.array_equal(best_cm,score_route(fixed,counts,labels,route))
    return route,best_cm,chosen

def base_model(width):
    return nn.Sequential(nn.Linear(width,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,24))

def class_threshold_search(counts, labels, gains, fixed, initial, score, passes=3, quantiles=25):
    """Fit source-class thresholds with exact pooled-CM accounting.

    Class-specific thresholds are calibration parameters, not an I intervention.
    Binned counts make the sweep linear in records, without storing predictions.
    """
    n=len(labels); rows=np.arange(n); source=labels[:,2]
    action=gains.argmax(1).astype(np.uint8); best=gains.max(1)
    new=labels[rows,action]; banks=[]; selected=[]
    initial_cm=fixed.copy()
    for y in range(25):
        initial_cm[y]+=np.bincount(labels[rows,initial],weights=counts[:,y],minlength=25).astype(np.int64)
    cm=initial_cm.copy()
    for cls in range(1,25):
        ix=np.flatnonzero(source==cls); yy=counts[ix]; gg=best[ix]
        vals=gg[gg>0]
        thresholds=np.unique(np.r_[0,np.quantile(vals,np.linspace(0,1,quantiles)) if len(vals) else [],np.inf])
        nb=len(thresholds)+1
        bins=np.searchsorted(thresholds,gg,side='left')
        by_new=np.zeros((nb,25,25),np.int64)
        for truth in range(25):
            by_new[:,truth,:]=np.bincount(bins*25+new[ix],weights=yy[:,truth],minlength=nb*25).reshape(nb,25).astype(np.int64)
        tails=np.cumsum(by_new[::-1],axis=0)[::-1]
        baseline=np.zeros((25,25),np.int64);baseline[:,cls]=yy.sum(0,dtype=np.int64)
        options=[]
        for j in range(len(thresholds)):
            delta=tails[j+1].copy();delta[:,cls]-=delta.sum(1)
            options.append(baseline+delta)
        original=np.zeros((25,25),np.int64)
        for truth in range(25):
            original[truth]=np.bincount(labels[ix,initial[ix]],weights=yy[:,truth],minlength=25).astype(np.int64)
        banks.append((ix,thresholds,np.stack(options),original))
        selected.append(-1)
    history=[]
    for iteration in range(passes):
        changed=0
        for cls,(ix,thresholds,options,original) in enumerate(banks):
            current=original if selected[cls]<0 else options[selected[cls]]
            remainder=cm-current; current_score=score(cm); best_idx=selected[cls]
            for j,option in enumerate(options):
                value=score(remainder+option)
                if value>current_score+1e-12:current_score=value;best_idx=j
            if best_idx!=selected[cls]:
                cm=remainder+options[best_idx];selected[cls]=best_idx;changed+=1
        history.append(dict(iteration=iteration+1,changed_classes=changed,miou_percent=score(cm)*100))
        if not changed:break
    route=initial.copy();policy={}
    for cls,(ix,thresholds,options,original) in enumerate(banks,1):
        j=selected[cls-1]
        if j<0:policy[str(cls)]='keep_initial_global_policy';continue
        threshold=float(thresholds[j]);route[ix]=2
        chosen=ix[best[ix]>threshold];route[chosen]=action[chosen]
        policy[str(cls)]=threshold if np.isfinite(threshold) else 'never_switch'
    return route,cm,dict(source_class_thresholds=policy,history=history)

def cm_from_groups(counts,labels,choice):
    pred=labels[np.arange(len(labels)),choice].astype(np.int64)
    out=np.zeros((25,25),np.int64)
    for y in range(25):out[y]=np.bincount(pred,weights=counts[:,y],minlength=25).astype(np.int64)
    return out

def miou(cm):
    x=np.asarray(cm,np.float64)[1:,1:];d=x.diagonal();return float((d/np.maximum(x.sum(0)+x.sum(1)-d,1)).mean())

def expected_cm(hist,labels,choice):
    pred=labels[np.arange(len(labels)),choice]-1
    return np.stack([np.bincount(pred,weights=hist[:,y],minlength=24) for y in range(24)])

def score_route(fixed,counts,labels,choice):return fixed+cm_from_groups(counts,labels,choice)
