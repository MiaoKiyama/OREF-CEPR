"""Linux C++ diagnostic graph and group-bank construction on synthetic data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'05_oref_training'))
from prepare_example_bank import prepare_scene

def main():
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp);folder=d/'scene';folder.mkdir();rng=np.random.default_rng(1337)
        for name in ('dinov2','dinov3','mos_off','mos_on'):
            p=rng.random((24,32,32),dtype=np.float32);p/=p.sum(0,keepdims=True);np.save(folder/(name+'.npy'),p)
        np.save(folder/'sar_blocks.npy',np.zeros((1,6),np.float32));(folder/'sar.json').write_text('{"available":true}')
        np.savez(d/'q.npz',q=np.ones((33,18),np.float32))
        Image.fromarray(rng.integers(1,25,(32,32),dtype=np.uint8)).save(d/'labels.tif')
        row=prepare_scene(folder,d/'q.npz',d/'labels.tif',d/'bank/scene')
        manifest=d/'bank/manifest.json';manifest.write_text(json.dumps(dict(source='oof120',scenes=[row])))
        assert row['groups'] > 0
        print(json.dumps(dict(status='passed',groups=row['groups'],cpp_graph=True)))


if __name__=='__main__':main()
