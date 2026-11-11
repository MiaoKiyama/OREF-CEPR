"""Download and SHA256-verify the published FBP checkpoints."""
import argparse
import hashlib
from pathlib import Path

FILES={
 'dinov2':('FBP_DINOv2_L.pth','6acf9f76a81be208fdf83535caffa6e016b29448b134e132e28be4472f9d2342'),
 'dinov3':('FBP_DINOv3_L.pth','bf60f2e8f329404116f5587099040cf763f4c190b48115165ae6385ad3f4ba25'),
 'mos':('FBP_MOSNet.pth','815c1de02e2bb8906cd9fe9b114c52bdfb6952f847eacd97256decac423bae6b'),
 'final':('FBP_QS_OREF_CEPR.pth','8324a07a03f58f472c0c326e1c8f8d5d66da135af356d9b3e5e4c68404d6bdbe'),
}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=[*FILES,'all'],default='final')
    p.add_argument('--output',type=Path,default=Path('weights'));a=p.parse_args()
    from huggingface_hub import hf_hub_download
    for key in (FILES if a.model=='all' else [a.model]):
        name,expected=FILES[key]
        path=Path(hf_hub_download('miaokiyama/OREF-CEPR',name,local_dir=a.output))
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8<<20),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('Checksum mismatch: '+name)
        print('Verified',path)

if __name__=='__main__':main()
