"""Original FBP objectives, extracted without mathematical changes."""
import torch
import math
from torch.nn import functional as F
def sar_soft_loss(logits, labels, optical_valid, native_observed):
    targets, mass = croma_region_targets(labels, optical_valid, num_classes=logits.shape[1])
    if logits.shape != targets.shape or native_observed.shape != mass.shape:
        raise ValueError("CROMA auxiliary logit/support lattice mismatch")
    eligible = (mass[:, 0] > 0) & native_observed[:, 0].bool()
    pixel = -(targets * logits.float().log_softmax(1)).sum(1)
    per_image = (pixel * eligible).flatten(1).sum(1) / eligible.flatten(1).sum(1).clamp_min(1)
    return per_image.mean(), eligible.float().mean().detach()
def compute_losses(output, batch, config, stage):
    """Own full-domain GT objective, independent of teacher probabilities/SAR masks."""
    from tcmmsline3.losses import soft_dice_loss, lovasz_softmax_loss
    if stage not in ("optical_warmup", "joint"):
        raise ValueError("unknown training stage")
    logits, labels = output["logits"].float(), batch["label"].long()
    if logits.shape != (len(labels),24,*labels.shape[-2:]) or labels.ndim != 3:
        raise ValueError("main logits/GT must share the native24 image geometry")
    if bool(((labels != 255) & ((labels < 0) | (labels >= 24))).any()):
        raise ValueError("invalid main target")
    zero = logits.sum()*0.
    pieces = {name:zero for name in ("main_ce","main_dice","main_lovasz","optical_aux","spectral_aux")}
    for i in range(len(labels)):
        target = labels[i:i+1].clone()
        target[target == 255] = -1
        if not bool((target >= 0).any()):
            continue
        own = logits[i:i+1]
        pieces["main_ce"] = pieces["main_ce"] + F.cross_entropy(own,target,ignore_index=-1)/len(labels)
        pieces["main_dice"] = pieces["main_dice"] + soft_dice_loss(own,target,24)/len(labels)
        pieces["main_lovasz"] = pieces["main_lovasz"] + lovasz_softmax_loss(own,target)/len(labels)
        for name in ("optical_aux", "spectral_aux"):
            if name+"_logits" in output:
                auxiliary = output[name+"_logits"][i:i+1].float()
                if auxiliary.shape[:2] != (1,24):
                    raise ValueError("auxiliary output must predict native24 classes")
                if auxiliary.shape[-2:] != target.shape[-2:]:
                    auxiliary = F.interpolate(auxiliary,size=target.shape[-2:],mode="bilinear",align_corners=False)
                pieces[name] = pieces[name] + F.cross_entropy(auxiliary,target,ignore_index=-1)/len(labels)
    main = pieces["main_ce"] + .5*pieces["main_dice"] + .25*pieces["main_lovasz"]
    auxiliary, support = zero, zero.detach()
    if stage == "joint":
        auxiliary, support = sar_soft_loss(output["sar_aux_logits"],labels,batch["optical_valid"],output["sar_aux_valid"])
    total = (main + config["optical_aux_weight"]*pieces["optical_aux"]
             + config["spectral_aux_weight"]*pieces["spectral_aux"] + config["sar_aux_weight"]*auxiliary)
    return total, {**{name:value.detach() for name,value in pieces.items()},
        "loss":total.detach(),"main":main.detach(),"sar_aux":auxiliary.detach(),"sar_aux_support":support.detach()}

def croma_region_targets(labels, optical_valid, *, num_classes=24, sar_size=166, patch=8):
    """Known optical pixel-centre counts in the padded native CROMA cells.

    This is explicitly discrete pixel-centre quadrature, not continuous area
    integration. No label interpolation, adaptive pooling or purity filtering.
    """
    if labels.ndim != 3 or labels.shape[-2:] != (512, 512):
        raise ValueError("native auxiliary supervision requires512 optical labels")
    if optical_valid.shape != (len(labels), 1, 512, 512) or optical_valid.dtype != torch.bool:
        raise ValueError("optical SAR support must be bool B1x512x512")
    if bool(((labels != 255) & ((labels < 0) | (labels >= num_classes))).any()):
        raise ValueError("auxiliary label IDs must be native classes or255")
    padded = math.ceil(sar_size / patch) * patch
    before = (padded - sar_size) // 2
    n = padded // patch
    coordinates = torch.arange(512, device=labels.device, dtype=torch.float64)
    axis = torch.floor((before + (coordinates + .5) * sar_size / 512) / patch).long()
    cells = (axis[:, None] * n + axis[None]).reshape(1, -1).expand(len(labels), -1)
    known = ((labels != 255) & optical_valid[:, 0]).reshape(len(labels), -1)
    combined = cells * num_classes + labels.reshape(len(labels), -1).clamp(0, num_classes - 1)
    counts = torch.zeros((len(labels), n*n*num_classes), device=labels.device, dtype=torch.float32)
    counts.scatter_add_(1, combined, known.float())
    counts = counts.reshape(len(labels), n, n, num_classes).permute(0, 3, 1, 2)
    mass = counts.sum(1, keepdim=True)
    return counts / mass.clamp_min(1), mass
