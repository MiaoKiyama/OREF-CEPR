"""Optional independent scorer. Labels are read only here, after inference."""
import os
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def metrics(cm):
 c=cm[1:,1:].astype(np.float64);d=c.diagonal();union=c.sum(0)+c.sum(1)-d
 iou=d/np.maximum(union,1)
 return dict(miou_percent=float(100*iou.mean()),oa_percent=float(100*d.sum()/c.sum()),valid_pixels=int(c.sum()),class_iou_percent=(100*iou).tolist())
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,help='Completed infer.py output');p.add_argument('--labels-root',required=True);a=p.parse_args()
 out=Path(a.output).resolve();assert read(out/'completed.json')['status']=='completed'
 contract=read(B/'evaluation_contract.json');cm=np.zeros((2,25,25),np.int64);dino_cm=np.zeros((25,25),np.int64);counts=np.zeros(3,np.int64);rows=[];start=time.time()
 for row in contract['scenes']:
  sid=row['id'];label=Path(a.labels_root)/row['label_file'];assert sha(label)==row['target_sha256'],sid
  with Image.open(label) as im:truth=np.asarray(im)
  receipt=read(out/'fusion/receipts/two_oref'/(sid+'.json'))
  with np.load(out/'fusion/predictions/two_oref'/(sid+'.npz'),allow_pickle=False) as z:prediction=z['prediction']
  assert prediction.shape==truth.shape==tuple(row['shape']);assert prediction.dtype==np.uint8 and prediction.min()>=1 and prediction.max()<=24
  digest=hashlib.sha256(np.ascontiguousarray(prediction).view(np.uint8)).hexdigest()
  assert digest==row['prediction_sha256']==receipt['array_sha256'],('Sealed prediction mismatch',sid)
  assert receipt['stage_array_sha256']==row['stage_array_sha256'],('Intermediate prediction mismatch',sid)
  meta=read(out/'candidates'/sid/'completed.json');scene_counts=np.zeros(3,np.int64);scene_cm=np.zeros_like(cm)
  for ch in meta['chunks']:
   path=Path(ch['path']);path=path if path.is_absolute() else out/path
   with np.load(path,allow_pickle=False) as z:labels=z['labels'];base=labels[6];dino=labels[0]
   sl=slice(ch['start'],ch['start']+ch['count']);y=truth.ravel()[sl];f=prediction.ravel()[sl];valid=y>0
   for k,v in enumerate([base,f]):scene_cm[k]+=np.bincount(y.astype(np.int64)*25+v,minlength=625).reshape(25,25)
   dino_cm+=np.bincount(y.astype(np.int64)*25+dino,minlength=625).reshape(25,25)
   scene_counts+=np.array([np.count_nonzero(valid&(base!=y)&(f==y)),np.count_nonzero(valid&(base==y)&(f!=y)),np.count_nonzero(valid&(base!=f))])
  cm+=scene_cm;counts+=scene_counts;rows.append(dict(scene=sid,prediction_exact=True,corrected=int(scene_counts[0]),harmed=int(scene_counts[1]),changed=int(scene_counts[2])))
 final=metrics(cm[1]);baseline=metrics(cm[0])
 for k in ['miou_percent','oa_percent']:assert abs(final[k]-contract['expected'][k])<1e-10,(k,final[k])
 assert final['valid_pixels']==contract['valid_pixels']
 original=metrics(dino_cm)
 result=dict(status='passed',scenes=len(rows),final=final,baseline=baseline,baseline_name='Tri-Avg-ON',original_baseline=original,original_baseline_name='DINOv2-L',delta_miou_pp=final['miou_percent']-baseline['miou_percent'],delta_original_miou_pp=final['miou_percent']-original['miou_percent'],corrected=int(counts[0]),harmed=int(counts[1]),changed=int(counts[2]),truth_used_only_in_evaluation=True,seconds=time.time()-start,rows=rows)
 (out/'evaluation.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'evaluation_confusions.npz',baseline=cm[0],final=cm[1],dino= dino_cm)
 print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
