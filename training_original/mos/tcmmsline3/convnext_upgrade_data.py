"""Full120 upgrade data; retain the original crop plan, add true1024 context.

Only context mode replaces the otherwise unused short view (old1536 support).
Geometry, optical augmentation and SAR flips are generated in one RNG stream.
"""
from __future__ import annotations

import random
import numpy as np
from PIL import Image
import torch

from .convnext_sar_full120_data import Full120Dataset
from .ms75_sar_joint_data import BLOCK_SIZE
from .ms75_rebuild_data import read_cached_scene
from .target_ms75_data import extract_boundless, resize_raw, _flip_views, _augment_views
from .target_dino_data import normalize_bgrn, SPECTRAL_AUGMENTATION
from .light_sar_data import _checked_aux, _load_points, open_sar_scene, apply_flips


def read_context_views(image, label, *, top, left, normalization, global_raw=None):
    """Local512 at native resolution; physical1024 context at identical centre."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Expected original uint8 BGRN scene")
    h,w = image.shape[:2]
    if min(top,left)<0 or top+512>h or left+512>w or label.shape != (h,w):
        raise ValueError("Native local512 must be inside its own original label/image scene")
    raw = image[top:top+512,left:left+512]
    context, valid = extract_boundless(image,top=top-256,left=left-256,size=1024)
    context = resize_raw(context,512)
    valid = np.asarray(Image.fromarray(valid.astype(np.uint8)).resize((512,512),Image.Resampling.NEAREST))>0
    if global_raw is None:
        global_raw=resize_raw(image,512)
    sample=dict(ms=normalize_bgrn(raw,normalization),ms_valid=np.ones((512,512),bool),
        ms_short=normalize_bgrn(context,normalization)*valid[None],ms_short_valid=valid,
        ms_global=normalize_bgrn(global_raw,normalization),ms_global_valid=np.ones((512,512),bool),
        local_box_short=np.array([.25,.25,.75,.75],np.float32),
        local_box_global=np.array([top/h,left/w,(top+512)/h,(left+512)/w],np.float32))
    return sample,np.array(label[top:top+512,left:left+512],dtype=np.int64,copy=True)


class UpgradeFull120Dataset(Full120Dataset):
    def __init__(self,config,epoch=0,start_sample=0):
        self.variant=config.get("variant")
        super().__init__(config,epoch,start_sample)
        if self.variant=="context" and config.get("context_source_size",1024)!=1024:
            raise ValueError("Context protocol is fixed at1024 native source pixels")

    def __getitem__(self,index):
        if self.variant!="context":
            return super().__getitem__(index)
        if not isinstance(index,(int,np.integer)) or not 0<=int(index)<len(self):
            raise IndexError(index)
        absolute=self.start_sample+int(index)
        scene,kind,cls=self.blocks[absolute//BLOCK_SIZE]
        rng=random.Random(self.seed+self.epoch*10000019+absolute*65537)
        image,labels=self._bounded(self._cache,scene,lambda:read_cached_scene(self.metadata,scene))
        h,w=labels.shape
        if min(h,w)<self.n:raise ValueError("Scene smaller than native local crop")
        redraw_count,fallback_used=0,False
        if kind=="scene":
            for attempt in range(16):
                top,left=rng.randrange(h-self.n+1),rng.randrange(w-self.n+1)
                if np.any(labels[top:top+self.n,left:left+self.n]<24):break
                redraw_count+=1
            else:
                row=self.metadata["scenes"][scene]
                reservoirs=self._bounded(self._points,scene,lambda:_load_points(row["points"],row["points_sha256"]))
                eligible=[c for c,points in reservoirs.items() if len(points)]
                if not eligible:raise ValueError("Training scene has no known optical labels")
                points=reservoirs[rng.choice(eligible)]
                cy,cx=map(int,points[rng.randrange(len(points))])
                top,left=int(np.clip(cy-self.n//2,0,h-self.n)),int(np.clip(cx-self.n//2,0,w-self.n))
                fallback_used=True
        else:
            if kind=="boundary":points=self._boundary(scene,labels)
            else:
                row=self.metadata["scenes"][scene]
                points=self._bounded(self._points,scene,lambda:_load_points(row["points"],row["points_sha256"]))[cls]
            if not len(points):raise ValueError("Chosen scene has no target sampling points")
            cy,cx=map(int,points[rng.randrange(len(points))]);jitter=self.n//4
            top=int(np.clip(cy-self.n//2+rng.randint(-jitter,jitter),0,h-self.n))
            left=int(np.clip(cx-self.n//2+rng.randint(-jitter,jitter),0,w-self.n))
        sample,target=read_context_views(image,labels,top=top,left=left,
            normalization=self.metadata["normalization"],global_raw=self.global_views.get(scene,image))
        known_pixels=int(np.count_nonzero(target<24))
        if known_pixels<1:raise ValueError("Crop must retain known optical GT")
        if self.availability[scene]:
            reader=self._bounded(self._aux,scene,lambda:open_sar_scene(self.data_root,scene))
            sar,native,optical=_checked_aux(reader.read_patch(top,left,self.n,modality="sar",out_size=self.sn),self.n,self.sn)
        else:
            sar,native,optical=np.zeros((2,self.sn,self.sn),np.float32),np.zeros((1,self.sn,self.sn),bool),np.zeros((1,self.n,self.n),bool)
        code=int(rng.random()<.5)+2*int(rng.random()<.5) if self.augmentation else 0
        sample=_flip_views(sample,bool(code&1),bool(code&2))
        if self.augmentation:
            sample=_augment_views(sample,rng=rng,augmentation=SPECTRAL_AUGMENTATION,normalization=self.metadata["normalization"])
        sample.update(ms_context=sample["ms_short"],context_valid=sample["ms_short_valid"][None],
                      local_box_context=sample["local_box_short"],sar_db=apply_flips(sar,code),
                      sar_valid=apply_flips(native,code),optical_valid=apply_flips(optical,code),label=apply_flips(target,code))
        result={k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in sample.items()}
        result.update(index=absolute,scene_id=scene,top=top,left=left,sampling_kind=kind,sampling_class=cls,
                      flip_code=code,has_sar=self.availability[scene],known_pixels=known_pixels,
                      redraw_count=redraw_count,fallback_used=fallback_used)
        return result

    def audit(self):
        return dict(super().audit(),variant=self.variant,context_source_size=1024 if self.variant=="context" else None,
                    context_uses_same_augmentation=True,context_changes_local_gt=False)


__all__=["UpgradeFull120Dataset","read_context_views"]
