"""Small executable checks; no claim of a new full FBP benchmark."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'05_oref_training'))
import numpy as np
import torch
from PIL import Image
from oref_cepr.bootstrap import LEGACY
from oref_cepr.data import splits,normalize
from oref_cepr.inference_models import dino_loss
from oref_cepr.objectives import compute_losses
from native24_object_graph_model import make_model,loss_value

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class Smoke(unittest.TestCase):
    def test_folds(self):
        s=splits();held=[]
        self.assertEqual(len(s['train_ids']),120);self.assertEqual(len(s['test_ids']),30)
        self.assertFalse(set(s['train_ids'])&set(s['test_ids']))
        for f in s['folds']:
            self.assertEqual(len(f['train_ids']),96);self.assertEqual(len(f['held_ids']),24)
            self.assertFalse(set(f['train_ids'])&set(f['held_ids']))
            self.assertEqual(set(f['train_ids']+f['held_ids']),set(s['train_ids']))
            held+=f['held_ids']
        self.assertEqual(len(set(held)),120)

    def test_objectives_backward(self):
        logits=torch.randn(1,24,32,32,requires_grad=True);labels=torch.randint(0,24,(1,32,32));labels[:,0]=255
        loss=dino_loss(logits,labels);loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        labels=torch.randint(0,24,(1,512,512));labels[:,0]=255
        logits=torch.randn(1,24,512,512,requires_grad=True)
        aux=torch.randn(1,24,21,21,requires_grad=True)
        output=dict(logits=logits,sar_aux_logits=aux,sar_aux_valid=torch.ones(1,1,21,21,dtype=torch.bool))
        batch=dict(label=labels,optical_valid=torch.ones(1,1,512,512,dtype=torch.bool))
        cfg=dict(optical_aux_weight=.2,spectral_aux_weight=.1,sar_aux_weight=.1)
        loss,_=compute_losses(output,batch,cfg,'joint');loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all());self.assertTrue(torch.isfinite(aux.grad).all())

    def test_oref_backward(self):
        torch.manual_seed(261016);n=4
        model=make_model(torch,np.zeros(313,np.float32),np.ones(313,np.float32),'graphQS')
        x=torch.randn(n,313);x[:,198:207]=1;x[:,213]=1
        counts=torch.randint(1,5,(n,24)).float();labels=torch.randint(0,24,(n,7))
        graph=(torch.randn(n,7,12),torch.randn(n,7,3,38),torch.ones(n,7,3))
        batch=(x,counts/counts.sum(1,keepdim=True),counts,torch.ones(n),labels,graph)
        before=model.base[0].weight.detach().clone();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        loss,*_=loss_value(model,batch,'CE',torch.zeros(24,24),n,torch,True)
        loss.backward();optimizer.step();self.assertFalse(torch.equal(before,model.base[0].weight))

    def test_q_views_and_pack(self):
        q=module(ROOT/'04_qwen_observations/generator_reference.py','q_parser')
        record=q.normalize(json.dumps({name:1 for name in q.ATTRIBUTE_NAMES}))
        self.assertEqual(record['values'][9:],[None]*9)
        invalid=q.normalize('{}');self.assertFalse(invalid['description_valid'])
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'images').mkdir()
            Image.fromarray(np.arange(64*80*4,dtype=np.uint8).reshape(64,80,4)).save(d/'images/demo.tif')
            subprocess.run([sys.executable,str(ROOT/'04_qwen_observations/prepare.py'),'--images',str(d/'images'),'--output',str(d/'views')],check=True,capture_output=True)
            rows=json.loads((d/'views/requests.json').read_text());self.assertEqual(len(rows),32)
            (d/'records').mkdir()
            for row in rows:(d/'records'/f'{row["roi"]}.json').write_text(json.dumps(dict(request=row,**record)))
            pack=module(ROOT/'04_qwen_observations/pack.py','q_pack');self.assertEqual(pack.pack(d/'records',d/'q'),1)
            with np.load(d/'q/demo.npz') as z:
                self.assertEqual(z['q'].shape,(33,18));self.assertTrue((z['q'][32]==0).all())

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
