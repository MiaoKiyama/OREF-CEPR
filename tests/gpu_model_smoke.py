"""One synthetic native-size forward/backward; no pretrained score validation."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from oref_cepr.inference_models import build_model,dino_loss
from oref_cepr.objectives import compute_losses

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,choices=['dinov2','dinov3','mos'])
    p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(1337);started=time.time()
    model,config=build_model(a.model,'pretrained',initialize=False);model.to(a.device)
    if a.model=='mos':model.set_stage('joint')
    model.train()
    batch=dict(ms=torch.randn(1,4,512,512,device=a.device),label=torch.randint(0,24,(1,512,512),device=a.device),
               sar_db=torch.randn(1,2,166,166,device=a.device)*5-15,
               sar_valid=torch.ones(1,1,166,166,device=a.device,dtype=torch.bool),
               optical_valid=torch.ones(1,1,512,512,device=a.device,dtype=torch.bool))
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        output=model.forward_batch(batch) if a.model=='mos' else model(ms=batch['ms'])
    loss=compute_losses(output,batch,config,'joint')[0] if a.model=='mos' else dino_loss(output['logits'],batch['label'])
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Missing/nonfinite gradients')
    print(json.dumps(dict(model=a.model,status='passed',shape=list(output['logits'].shape),loss=float(loss.detach()),
                         gradient_tensors=len(gradients),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                         seconds=time.time()-started,pretrained_weights_used=False)),flush=True)

if __name__=='__main__':main()
