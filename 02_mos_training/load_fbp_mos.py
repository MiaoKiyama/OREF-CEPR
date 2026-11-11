"""Load the complete FBP MOS-Net checkpoint without separate backbone weights.

Requires torch, numpy, einops, timm and PyYAML. The checkpoint includes its
audited model-definition source; load only a checkpoint from a trusted source.
This entry loads the network. Scene reading and sliding-window stitching are
separate from network loading.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys
import tempfile

FORMAT = 'fbp_mosnet_full_state_v1'


def source_hash(files):
    h = hashlib.sha256()
    for name, text in sorted(files.items()):
        h.update(name.encode('utf-8'))
        h.update(b'\0')
        h.update(text.encode('utf-8'))
        h.update(b'\0')
    return h.hexdigest()


def load_mos(checkpoint, device='cpu', cache_dir=None):
    """Return (model, metadata); modes share this one model's full state."""
    import torch
    checkpoint = Path(checkpoint).resolve(strict=True)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Expected a complete FBP MOS-Net checkpoint')
    files = payload['model_sources']
    digest = source_hash(files)
    if digest != payload['model_sources_sha256']:
        raise ValueError('Embedded model source checksum mismatch')
    root = Path(cache_dir or Path(tempfile.gettempdir())/'fbp_mosnet_full') / digest[:20]
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Invalid model source path')
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Model source path escapes cache')
        content = text.encode('utf-8')
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError('Cached model source differs: '+name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    existing = sys.modules.get('tcmmsline3')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Load this checkpoint in a fresh Python process, before importing another tcmmsline3 version')
    sys.path.insert(0, str(root))
    from tcmmsline3.convnext_pyramid_model import ConvNeXtPyramidSAR
    config = dict(payload['model_config'])
    config.update(convnext_source=str(root/'vendor/dinov3_convnext/convnext.py'),
                  croma_source=str(root/'vendor/CROMA/use_croma.py'),
                  convnext_checkpoint='__NO_EXTERNAL_WEIGHT_FILE__',
                  croma_checkpoint='__NO_EXTERNAL_WEIGHT_FILE__')
    # The bundled constructors accept load_pretrained=False for BOTH encoders.
    # All learnable/frozen state is then loaded strictly from this one file.
    model = ConvNeXtPyramidSAR(config, load_pretrained=False)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.set_stage('joint').eval().requires_grad_(False)
    model.to(device)
    metadata = {key: value for key, value in payload.items()
                if key not in ('state_dict', 'model_sources')}
    return model, metadata


def predict_logits(model, *, ms, sar_mode='off', sar_db=None, sar_valid=None, optical_valid=None):
    """MS: normalized B,G,R,NIR Bx4x512x512; SAR: VV,VH dB Bx2x166x166."""
    import torch
    if sar_mode not in ('on','off'):
        raise ValueError('sar_mode must be on or off')
    with torch.inference_mode():
        if sar_mode == 'off':
            return model.forward_optical(ms=ms)['logits']
        if any(value is None for value in (sar_db,sar_valid,optical_valid)):
            raise ValueError('SAR-ON requires SAR data and both validity masks')
        return model(ms=ms,sar_db=sar_db,sar_valid=sar_valid,
                     optical_valid=optical_valid,sar_mode='real')['logits']


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path(__file__).with_name('FBP_MOSNet.pth'))
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    model, metadata=load_mos(a.model,a.device)
    print(json.dumps({'status':'loaded','model':str(a.model),'num_classes':model.num_classes,
                      'state_entries':len(model.state_dict()),'sar_modes':['on','off'],
                      'separate_backbone_weights_required':False},indent=2))
