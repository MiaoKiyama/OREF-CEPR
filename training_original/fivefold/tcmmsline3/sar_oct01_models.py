"""October 1 controlled SAR experiments; all production backbones stay frozen.

The native grids describe sampling support, not newly recovered resolution.
The representation penalty is DeCUR/VICReg-inspired, not a reproduction.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .convnext_sar_model import fixed_sar_intensity
from .joint_sar_model import JointCromaSAR
from .ms75_sar_joint_model import MODEL_INPUT_KEYS, JOINT_INPUT_KEYS, align_croma_to_ms


def native_sample(x, size, span, offset=(0., 0.)):
    """SAR166 boundary coordinates -> pooled grid with right/bottom padding."""
    axes = [(torch.arange(n, device=x.device, dtype=torch.float32)+.5)*166/n
            for n in size]
    yy, xx = torch.meshgrid(axes[0]+offset[0], axes[1]+offset[1], indexing='ij')
    grid = torch.stack((2*xx/span-1, 2*yy/span-1), -1)[None].expand(len(x),-1,-1,-1)
    with torch.autocast(x.device.type, enabled=False):
        return F.grid_sample(x.float(), grid, align_corners=False, padding_mode='zeros')


def supported_pool(x, valid, stride):
    pad = (-x.shape[-1]) % stride
    mass = F.avg_pool2d(F.pad(valid.float(), (0,pad,0,pad)), stride, stride)
    value = F.avg_pool2d(F.pad(x*valid, (0,pad,0,pad)), stride, stride)
    return value / mass.clamp_min(1e-6), mass, x.shape[-1]+pad


class NativeSAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3,64,3,padding=1,bias=False),nn.GELU(),
                                     nn.Conv2d(64,64,3,padding=1,bias=False),nn.GELU())

    def forward(self, db, valid, coarse=False):
        x = fixed_sar_intensity(db,valid)
        if coarse:
            # Identical trainable encoder; degrade only its input's spatial scale.
            x, mass, span = supported_pool(x,valid,8)
            x = native_sample(x,(166,166),span)*valid
        feature = self.encoder(x)*valid
        return {s:supported_pool(feature,valid,s) for s in (2,4)}


class LocalRead(nn.Module):
    def __init__(self, channels, croma_channels, neighborhood):
        super().__init__()
        self.neighborhood = neighborhood
        self.query = nn.Conv2d(channels,64,1)
        self.key = nn.Conv2d(64,64,1,bias=False)
        self.value = nn.Conv2d(64,64,1,bias=False)
        self.coarse = nn.Conv2d(croma_channels,64,1,bias=False)
        self.output = nn.Conv2d(64,channels,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, optical, native, content, optical_valid, observed):
        size = optical.shape[-2:]
        q = self.query(optical)
        # 20m grid for readout; 40m grid supplies a second context scale.
        feature,mass,span = native[2]
        value = self.value(feature)
        valid = native_sample(mass,size,span).clamp(0,1)
        if self.neighborhood:
            k = self.key(feature)
            offsets = [(y,x) for y in (-2.,0.,2.) for x in (-2.,0.,2.)]
            keys = torch.stack([native_sample(k,size,span,o) for o in offsets],2)
            values = torch.stack([native_sample(value,size,span,o) for o in offsets],2)
            support = torch.stack([native_sample(mass,size,span,o)[:,0] for o in offsets],1)
            scores = (q.float().unsqueeze(2)*keys).sum(1)/8.
            weights = scores.masked_fill(support<=0,-1e4).softmax(1)*(support>0)
            weights = weights/weights.sum(1,keepdim=True).clamp_min(1e-6)
            read = (values*weights.unsqueeze(1)).sum(2)
        else:
            # Keep Q/K parameters and capacity registry identical across arms.
            read = native_sample(value,size,span)*torch.sigmoid(q.float())
        coarse, coarse_mass, coarse_span = native[4]
        read = read + .5*native_sample(self.value(coarse),size,coarse_span)
        read = read + align_croma_to_ms(self.coarse(content),size)
        support = F.adaptive_avg_pool2d(optical_valid.float(),size)*valid
        support = support*(align_croma_to_ms(observed.float(),size)>0)
        delta = self.output(read.to(optical.dtype))*support.to(optical.dtype)
        return optical+delta


def variance_covariance(x):
    z = x-x.mean(0)
    std = (z.square().mean(0)+1e-4).sqrt()
    variance = F.relu(.25-std).mean()
    cov = z.T@z/max(1,len(z)-1)
    off = cov-torch.diag_embed(torch.diagonal(cov))
    return variance, off.square().sum()/x.shape[1]


def representation_loss(s, augmented, optical, valid):
    """Per-image spatial moments: independent of accumulation/microbatch size.

    Common32 is aligned with optical32. Private32 is invariant within SAR,
    noncollapsed, and penalized for linear cross-correlation with optical.
    This does not assert that private variables are purely unique semantics.
    """
    loss = s.sum()*0 + augmented.sum()*0 + optical.sum()*0
    details = {k:loss.detach().clone() for k in ('common','intra','variance','decorrelation','cells')}
    for i in range(len(s)):
        keep = valid[i,0]>.75
        if int(keep.sum()) < 32:
            continue
        a,b,o = [v[i,:,keep].T.float() for v in (s,augmented,optical)]
        common = .5*(F.mse_loss(a[:,:32],o)+F.mse_loss(b[:,:32],o))
        intra = F.mse_loss(a,b)
        vv,cc = zip(*(variance_covariance(v) for v in (a,b,o)))
        variance = sum(vv)/3
        private = a[:,32:]-a[:,32:].mean(0)
        other = o-o.mean(0)
        private = private/(private.square().mean(0)+1e-4).sqrt()
        other = other/(other.square().mean(0)+1e-4).sqrt()
        decorrelation = (private.T@other/len(a)).square().mean()+.1*sum(cc)/3
        loss = loss+(common+intra+variance+.05*decorrelation)/len(s)
        for k,v in dict(common=common,intra=intra,variance=variance,decorrelation=decorrelation,
                        cells=keep.sum()).items():
            details[k] += v.detach()/len(s)
    return loss,details


class MS75NativeExpert(nn.Module):
    def __init__(self,optical,config,variant):
        super().__init__()
        self.optical = optical.requires_grad_(False).eval()
        self.sar_encoder = JointCromaSAR(config['croma_checkpoint'],config['croma_source']).requires_grad_(False).eval()
        self.native = NativeSAR()
        self.variant = variant
        self.indices = (1,2,3)
        channels = [int(optical.level_fusions[i].refine[0].out_channels) for i in self.indices]
        self.reads = nn.ModuleList(LocalRead(c,self.sar_encoder.channels[-1],variant not in ('coarse_gate','native_gate')) for c in channels)
        self.optical_projection = nn.Conv2d(channels[1],32,1)
        self.context = None
        self.ssl = None
        self._hooks = [optical.level_fusions[i].register_forward_hook(self._hook(j)) for j,i in enumerate(self.indices)]

    def train(self,mode=True):
        super().train(mode)
        self.optical.eval();self.sar_encoder.eval()
        return self

    def mutable_keys(self):
        return tuple(n for n in self.state_dict() if n.startswith(('native.','reads.','optical_projection.')))

    def _hook(self,index):
        def hook(module,args,output):
            if self.context is None:return output
            native,content,valid,observed,augmented = self.context
            value = output[0]
            if index==1 and augmented is not None:
                s,mass,span = native[2]
                a,_,_ = augmented[2]
                s = native_sample(s,value.shape[-2:],span)
                a = native_sample(a,value.shape[-2:],span)
                mask = native_sample(mass,value.shape[-2:],span)*F.adaptive_avg_pool2d(valid.float(),value.shape[-2:])
                # Nonlinear encoder precedes this split; no optical subtraction.
                self.ssl = representation_loss(s,a,self.optical_projection(value.detach()),mask)
            fused = self.reads[index](value,native,content,valid,observed)
            return (fused,*output[1:])
        return hook

    def forward_optical(self,**inputs):
        if self.context is not None:raise RuntimeError('Nested MS75 forward')
        return self.optical(**{k:inputs[k] for k in MODEL_INPUT_KEYS})

    def forward(self,*,sar_db,sar_valid,optical_valid,sar_mode='real',compute_representation=False,**inputs):
        if sar_mode=='zero':return self.forward_optical(**inputs)
        if self.context is not None:raise RuntimeError('Nested MS75 forward')
        with torch.no_grad():
            real = self.sar_encoder(sar_db,sar_valid,'real')[-1]
            reference = self.sar_encoder(sar_db,sar_valid,'zero')[-1]
            content = real-reference
            observed = F.avg_pool2d(F.pad(sar_valid.float(),(1,1,1,1)),8,8)>0
        native = self.native(sar_db,sar_valid,self.variant=='coarse_gate')
        augmented = None
        if compute_representation:
            noisy = sar_db+.5*torch.randn_like(sar_db)*sar_valid
            augmented = self.native(noisy,sar_valid,False)
        self.context = (native,content,optical_valid,observed,augmented)
        self.ssl = None
        try:
            output = self.optical(**{k:inputs[k] for k in MODEL_INPUT_KEYS})
            if compute_representation:
                if self.ssl is None:raise RuntimeError('Representation hook did not run')
                output['representation_loss'],output['representation_stats'] = self.ssl
            return output
        finally:self.context=None;self.ssl=None

    def forward_batch(self,batch,**kw):
        return self(**{k:batch[k] for k in JOINT_INPUT_KEYS},**kw)


class FeatureAdapter(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(channels,64,1),nn.GELU(),nn.Conv2d(64,channels,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,x):return x+self.layers(x)


class RepairExpert(nn.Module):
    """e16 frozen backbone/decoder + three learned semantic feature adapters."""
    def __init__(self,parent):
        super().__init__()
        self.parent=parent.requires_grad_(False).eval()
        self.adapters=nn.ModuleList(FeatureAdapter(c) for c in parent.backbone.channels[1:])
        self.enabled=True
        self._hook=parent.decoder.register_forward_pre_hook(self._adapt)
    def _adapt(self,module,args):
        if not self.enabled:return args
        optical,spectral=args
        return ((optical[0],*(m(x) for m,x in zip(self.adapters,optical[1:]))),spectral)
    def train(self,mode=True):
        super().train(mode);self.parent.eval();return self
    def mutable_keys(self):return tuple(n for n in self.state_dict() if n.startswith('adapters.'))
    def forward(self,**kw):return self.parent(**kw)
    def forward_batch(self,batch,**kw):return self.parent.forward_batch(batch,**kw)
    def forward_optical(self,**kw):return self.parent.forward_optical(**kw)
    @torch.no_grad()
    def reference(self,batch):
        self.enabled=False
        try:return self.parent.forward_batch(batch)['logits']
        finally:self.enabled=True


def class_mean(value,labels,mask):
    terms=[]
    for v,y,m in zip(value,labels,mask):
        sums=torch.zeros(24,device=v.device).scatter_add(0,y[m],v[m])
        counts=torch.bincount(y[m],minlength=24)
        terms.append((sums/counts.clamp_min(1)).sum()/(counts>0).sum().clamp_min(1))
    return torch.stack(terms).mean()


def true_margin(logits,labels):
    safe=labels.clamp(0,23).unsqueeze(1)
    truth=logits.gather(1,safe)[:,0]
    other=logits.scatter(1,safe,float('-inf')).amax(1)
    return truth-other


def dual_preservation(student,parent,ms75,labels):
    """GT selects training regions only; neither teacher is used for routing."""
    known=labels<24
    with torch.no_grad():
        correct=known&(parent.argmax(1)==labels)
        recover=known&~correct&(ms75.argmax(1)==labels)
        old_margin=true_margin(parent.float(),labels).clamp(0,2)
        target_margin=true_margin(ms75.float(),labels).clamp(0,2)
    margin=true_margin(student.float(),labels)
    keep=class_mean(F.relu(old_margin-margin),labels,correct)
    fix=class_mean(F.relu(target_margin-margin),labels,recover)
    stats=dict(keep=keep.detach(),recover=fix.detach(),keep_fraction=correct.float().mean(),recover_fraction=recover.float().mean())
    for name,mask,classes in [('positive',correct,labels),('recover_gt',recover,labels),('false_positive',recover,parent.argmax(1))]:
        counts=torch.bincount(classes[mask],minlength=24).float()/len(labels)
        for i in range(24):stats[f'{name}_{i:02d}']=counts[i]
    return .5*keep+.5*fix,stats
