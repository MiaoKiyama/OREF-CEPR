"""Exercise public OOF inference with native-size synthetic input and real models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from oref_cepr.inference_models import build_model
from oref_cepr.data import splits

def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        r=Path(tmp);(r/'data/images').mkdir(parents=True);(r/'data/auxiliary').mkdir()
        fold=splits()['folds'][0];sid=fold['held_ids'][0]
        Image.fromarray(np.random.default_rng(1337).integers(0,256,(512,512,4),dtype=np.uint8)).save(r/'data/images'/(sid+'.tif'))
        (r/'data/auxiliary/availability.csv').write_text('scene_stem,numeric_scene_id,has_pan,has_sar\n'+sid+',1,no,no\n')
        for kind in ['dinov2','dinov3','mos']:
            model,config=build_model(kind,r,initialize=False)
            checkpoint=r/(kind+'.pt')
            torch.save(dict(model=model.state_dict(),signature=dict(fold=0,train_ids=fold['train_ids'])),checkpoint)
            del model
            subprocess.run([sys.executable,str(ROOT/'03_fivefold_training/predict.py'),'--model',kind,'--checkpoint',str(checkpoint),
                            '--fold','0','--scene-limit','1','--data-root',str(r/'data'),'--output',str(r/'pred')],check=True)
            names=['mos_on','mos_off'] if kind=='mos' else [kind]
            for name in names:
                a=np.load(r/'pred'/sid/(name+'.npy'),mmap_mode='r')
                assert a.shape==(24,512,512) and np.isfinite(a).all()
                np.testing.assert_allclose(a.sum(0),1,atol=1e-5)
                del a
            checkpoint.unlink()
            print(json.dumps(dict(model=kind,status='passed',softmax_normalized=True)),flush=True)

if __name__=='__main__':main()
