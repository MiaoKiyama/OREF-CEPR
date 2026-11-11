"""Frozen FBP83 backend: observable candidate features -> neural posteriors -> C/E/P/R.

This entry accepts observable caches. Use infer.py for the raw-image pipeline.
"""
import argparse,json,os,subprocess,sys,time
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
from pathlib import Path

def arguments():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
 p.add_argument('--stage',choices=['all','posteriors','groups','spatial'],default='all')
 p.add_argument('--gpu',default='7');p.add_argument('--workers',type=int,default=2)
 p.add_argument('--lane',type=int,default=0)
 return p.parse_args()

def initialize(a):
 os.environ.update(FBP83_OUTPUT=str(Path(a.output).resolve()),CUDA_VISIBLE_DEVICES=a.gpu,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
 sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
 from common import PACKAGE,ROOT,read,write,sha
 ROOT.mkdir(parents=True,exist_ok=True)
 manifest=Path(a.inputs).resolve();cfg=read(PACKAGE/'rules.json');inputs=read(manifest)
 assert inputs['candidate_raw_order']==['D2','D3','dual','pyramid_off','triple_off','pyramid_on','triple_on']
 assert inputs['candidate_oref_order']==['dual','triple_off','triple_on','pyramid_off','pyramid_on','D2','D3']
 cfg['scenes']=inputs['scenes'];cursor=0
 for row in cfg['scenes']:
  assert row['start']==cursor and row['end']>cursor;cursor=row['end']
  for k in ['keys','observable','graph']:
   row[k]=str((manifest.parent/row[k]).resolve())
  row['q']=str((PACKAGE/row['q']).resolve())
  for ch in row['chunks']:ch['path']=str((manifest.parent/ch['path']).resolve())
 identity=dict(inputs_sha256=sha(manifest),rules_sha256=sha(PACKAGE/'rules.json'),scene_ids=[r['scene'] for r in cfg['scenes']])
 p=ROOT/'run_identity.json'
 if p.exists():assert read(p)==identity,'Output belongs to a different input manifest/rule set'
 else:write(p,identity)
 return cfg

def observation_guard(cfg):
 """Deny every non-observable NPZ member before NumPy can decode it."""
 import numpy as np
 from common import write,ROOT
 allowed={}
 for r in cfg['scenes']:
  allowed[r['keys']]={'keys','numeric','labels'}
  allowed[r['observable']]={'mass','fixed_prediction_counts'}
  allowed[r['q']]={'q'}
  allowed[str(Path(r['graph'])/'graph.npz')]={'geometry','labels','expert','area','contact','overlap','continuation','primary','mass','extra'}
  for c in r['chunks']:allowed[c['path']]={'labels','top1_fp32','region_raw','region_x','region_features'}
 original=np.lib.npyio.NpzFile.__getitem__;audit={'members':{},'denied':[]}
 def get(z,key):
  path=str(Path(z.zip.filename).resolve())
  if path in allowed:
   if key not in allowed[path]:audit['denied'].append(path+'::'+key);raise PermissionError('Non-observable member: '+key)
   token=path+'::'+key;audit['members'][token]=audit['members'].get(token,0)+1
  elif not Path(path).is_relative_to(ROOT):
   audit['denied'].append(path+'::'+key);raise PermissionError('Unregistered observable NPZ: '+path)
  return original(z,key)
 np.lib.npyio.NpzFile.__getitem__=get
 try:
  with np.load(cfg['scenes'][0]['chunks'][0]['path']) as z:z['gt']
 except PermissionError:
  assert len(audit['denied'])==1;audit['negative_control']=audit['denied'].pop()
 else:raise AssertionError('Truth-member negative control did not block')
 return audit

def posteriors(cfg):
 import gc,numpy as np,torch
 from common import PACKAGE,ROOT,load,read,write,sha,progress
 from features import canonical,SceneGraph,normalize_graph
 from native24_object_graph_model import make_model
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 # Preserve the original FP32 model-forward precision and scene/batch boundaries.
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.use_deterministic_algorithms(True)
 audit=observation_guard(cfg);n=cfg['scenes'][-1]['end'];beg=time.time()
 assets={x['path']:x for x in read(PACKAGE/'assets.json')['files']}
 states=[];models=[]
 for name in ['oref_oof8k.pt','oref_replay8k.pt']:
  p=PACKAGE/'weights'/name;assert sha(p)==assets['weights/'+name]['sha256']
  state=torch.load(p,map_location='cpu',weights_only=False)
  assert state['arm']=='graphQS' and state['checkpoint']=='CE_8000'
  model=make_model(torch,state['mean'],state['scale'],state['arm'],state['graph_meta']).to('cuda')
  model.load_state_dict(state['state'],strict=True);model.requires_grad_(False).eval()
  states.append(state);models.append(model)
 target=ROOT/'posteriors';target.mkdir(exist_ok=True)
 assert not (target/'completed.json').exists(),'Posteriors already complete; choose a new output for rerun'
 arrays=[np.lib.format.open_memmap(target/(name+'.partial.npy'),mode='w+',dtype=np.float32,shape=(n,24)) for name in ['oof','replay']]
 graph_cfg={'graph_paths':{r['scene']:r['graph'] for r in cfg['scenes']}}
 with torch.inference_mode():
  for i,row in enumerate(cfg['scenes']):
   d=load(row['keys'],'keys','numeric','labels');q=load(row['q'],'q')['q']
   assert d['numeric'].shape==(row['end']-row['start'],126)
   numeric=np.concatenate([d['numeric'],q[((d['keys']//25**7)%33).astype(int)]],1)
   raw=canonical(numeric,d['labels']);graph=SceneGraph(graph_cfg,row['scene'],'official')
   assert graph.n==len(raw);del d,numeric;gc.collect()
   for lo in range(0,len(raw),4096):
    hi=min(lo+4096,len(raw));sl=slice(row['start']+lo,row['start']+hi)
    g={k:np.ascontiguousarray(v) for k,v in graph.get(np.arange(lo,hi,dtype=np.int64)).items()}
    for j,(state,model) in enumerate(zip(states,models)):
     x=(raw[lo:hi].copy()-state['mean'])/state['scale']
     gv=tuple(torch.as_tensor(v,device='cuda') for v in normalize_graph(g,state['graph_norm']))
     arrays[j][sl]=model(torch.as_tensor(x,device='cuda'),graph=gv).softmax(1).cpu().numpy()
   progress('neural posteriors',completed_scenes=i+1,total_scenes=len(cfg['scenes']),scene=row['scene'],seconds=time.time()-beg)
   del raw,graph;gc.collect()
 for name,arr in zip(['oof','replay'],arrays):
  assert np.isfinite(arr).all();np.testing.assert_allclose(arr.sum(1),1,rtol=1e-5,atol=1e-6);arr.flush()
 del arr,arrays
 for name in ['oof','replay']:(target/(name+'.partial.npy')).replace(target/(name+'.npy'))
 assert not audit['denied'];write(ROOT/'audit/posteriors.json',audit)
 write(target/'completed.json',dict(seconds=time.time()-beg,groups=n,networks=2,network_forward=True,truth_read=False,sha256={k:sha(target/(k+'.npy')) for k in ['oof','replay']}))

def groups(cfg):
 import numpy as np
 from common import ROOT,read,write,progress,sha,arrsha
 from group_cases import pack_inputs
 from frozen_groups_fast import calibrated,expansion,enable_parallel_kernels
 enable_parallel_kernels()
 audit=observation_guard(cfg);receipt=read(ROOT/'posteriors/completed.json')
 for k in ['oof','replay']:assert sha(ROOT/'posteriors'/(k+'.npy'))==receipt['sha256'][k]
 beg=time.time();pack=pack_inputs(cfg);labs=pack['labels'];p1=np.load(ROOT/'posteriors/oof.npy',mmap_mode='r');p2=np.load(ROOT/'posteriors/replay.npy',mmap_mode='r')
 base=calibrated(cfg,pack,p1,p2);expanded,final=expansion(cfg,pack,p1,base)
 dest=ROOT/'groups';dest.mkdir(exist_ok=True);np.savez_compressed(dest/'two_oref.npz',base=base,expanded=expanded,combined=final,scene_ids=np.array([r['scene'] for r in cfg['scenes']]))
 assert not audit['denied'];write(ROOT/'audit/groups.json',audit)
 write(dest/'completed.json',dict(seconds=time.time()-beg,truth_read=False,route_cache_used=False,sha256=sha(dest/'two_oref.npz'),arrays={k:arrsha(v) for k,v in [('base',base),('expanded',expanded),('combined',final)]}))

def spatial(cfg,a):
 from common import ROOT,read,load,write,sha
 from fast_spatial import verify_kernel,fast_neighborhood
 from vendor import multi
 from reference_spatial import scene
 verify_kernel();multi.neighborhood=fast_neighborhood;audit=observation_guard(cfg)
 path=ROOT/'groups/two_oref.npz';assert sha(path)==read(ROOT/'groups/completed.json')['sha256']
 z=load(path,'combined','scene_ids');assert z['scene_ids'].tolist()==[r['scene'] for r in cfg['scenes']]
 for row in cfg['scenes'][a.lane::a.workers]:
  receipt=ROOT/'receipts/two_oref'/(row['scene']+'.json')
  if receipt.exists():assert sha(read(receipt)['path'])==read(receipt)['sha256'];continue
  scene(row,z['combined'][row['start']:row['end']],cfg)
 assert not audit['denied'];write(ROOT/('audit/spatial_%d.json'%a.lane),audit)

def main():
 a=arguments();cfg=initialize(a)
 if a.stage=='all':
  from common import ROOT,write
  common=[sys.executable,'-u',__file__,'--inputs',str(Path(a.inputs).resolve()),'--output',str(ROOT),'--gpu',a.gpu,'--workers',str(a.workers)]
  beg=time.time()
  for stage in ['posteriors','groups']:
   receipt=ROOT/stage/'completed.json'
   if receipt.exists():
    from common import read,sha
    record=read(receipt)
    if stage=='posteriors':
     for name,digest in record['sha256'].items():assert sha(ROOT/stage/(name+'.npy'))==digest
    else:assert sha(ROOT/stage/'two_oref.npz')==record['sha256']
   else:subprocess.run(common+['--stage',stage],check=True)
  jobs=[subprocess.Popen(common+['--stage','spatial','--lane',str(i)]) for i in range(a.workers)]
  codes=[p.wait() for p in jobs];assert codes==[0]*a.workers,codes
  write(ROOT/'completed.json',dict(status='completed',seconds=time.time()-beg,scenes=len(cfg['scenes']),network_forward=True,route_cache_used=False,truth_read=False))
 elif a.stage=='posteriors':posteriors(cfg)
 elif a.stage=='groups':groups(cfg)
 else:spatial(cfg,a)

if __name__=='__main__':main()
