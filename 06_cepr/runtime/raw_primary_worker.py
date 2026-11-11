"""Frozen primary experts. Inputs are raw MS/SAR only; output is caller-owned RAM."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',NUMPY_MADVISE_HUGEPAGE='0')
import argparse,copy,gc,importlib,inspect,json,sys,time
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'code'))
from common import read,write,sha
_DINO_MODELS={}
_MOS=None
_ACTIVE={}

def load_image(job):
 from PIL import Image
 p=Path(job['image']);assert sha(p)==job['image_sha256']
 with Image.open(p) as im:a=np.ascontiguousarray(np.asarray(im)[...,::-1])
 assert a.shape==tuple(job['shape'])+(4,) and a.dtype==np.uint8
 a.flags.writeable=False;return a

def output(job,index):return np.memmap(job['fields'][index],dtype=np.float32,mode='r+',shape=(24,*job['shape']))

def guard(job):
 root=Path(job['data_root']).resolve();image=Path(job['image']).resolve();aux=root/'auxiliary';seen=set()
 installed=bool(_ACTIVE);_ACTIVE.update(root=root,image=image,aux=aux,seen=seen)
 if installed:return seen
 def hook(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.path.abspath(os.fsdecode(args[0])))
  if p.is_relative_to(_ACTIVE['root']):
   if p!=_ACTIVE['image'] and not p.is_relative_to(_ACTIVE['aux']):raise PermissionError('Inference cannot open non-image/auxiliary dataset file: '+str(p))
   _ACTIVE['seen'].add(str(p))
 sys.addaudithook(hook);return seen

def dino(job,kind):
 cfg=read(B/'source_snapshot/primary_source_config.json');spec=copy.deepcopy(cfg['optical_models'][kind]);name='d2_e17.pt' if kind=='v2l_e17' else 'd3_e26.pt'
 spec['checkpoint']=str(B/'weights'/name);spec['source_root']=str(B/'source_snapshot/dino')
 spec['runtime_source_files']={str(B/'source_snapshot/native'/Path(p).relative_to(cfg['native_runtime'])):v for p,v in spec['runtime_source_files'].items()}
 sys.path.insert(0,str(B/'source_snapshot/native'))
 from tools import ms75_rebuild_fusion_inference as inference
 # Preload the authenticated scheduling adapter before native source binding.
 from tools import ms75_hr_autofusion_fast_inference
 import torch
 torch.set_num_threads(4)
 image=load_image(job);arr=output(job,0 if kind=='v2l_e17' else 1)
 def progress(**v):write(Path(job['output'])/(kind+'_progress.json'),dict(time_unix=time.time(),**v))
 original_builder=inference._modern_model;original_bind=inference._bind_sources
 reused=kind in _DINO_MODELS;tokens=[]
 def builder(s,ids):
  if kind not in _DINO_MODELS:_DINO_MODELS[kind]=original_builder(s,ids)
  m=_DINO_MODELS[kind];tokens.append([(k,id(v),v._version) for k,v in m.state_dict(keep_vars=True).items()]);return m
 inference._modern_model=builder
 if reused:inference._bind_sources=lambda s:Path(s['source_root'])
 try:result=inference.infer_array(spec,image,arr,expected_test_ids=cfg['scene_ids'],report=progress)
 finally:inference._modern_model=original_builder;inference._bind_sources=original_bind
 assert tokens==[[(k,id(v),v._version) for k,v in _DINO_MODELS[kind].state_dict(keep_vars=True).items()]],'Frozen DINO state changed'
 result['frozen_model_reused']=reused
 arr.flush();return result

class Inputs:
 def __init__(self,job,image,normalization):
  from tools.evaluate_ms75_rebuild import sliding_starts
  from tcmmsline3.light_sar_data import open_sar_scene
  self.job,self.image,self.normalization=job,image,normalization
  self.reader=open_sar_scene(job['data_root'],job['scene'])
  assert self.reader.available['sar']==job['has_sar']
  self.coords=[(y,x) for y in sliding_starts(image.shape[0]) for x in sliding_starts(image.shape[1])]
  self.indices={xy:i for i,xy in enumerate(self.coords)}
  self.summaries=np.zeros((len(self.coords),6),np.float32);self.summarized=np.zeros(len(self.coords),bool)
  self.aux={}
 def batch(self,coords):
  from tcmmsline3.target_dino_data import normalize_bgrn
  from tcmmsline3.light_sar_data import _checked_aux
  samples=[]
  for y,x in coords:
   i=self.indices[(y,x)]
   if i not in self.aux:
    if self.reader.available['sar']:v=_checked_aux(self.reader.read_patch(y,x,512,modality='sar',out_size=166),512,166)
    else:v=(np.zeros((2,166,166),np.float32),np.zeros((1,166,166),bool),np.zeros((1,512,512),bool))
    self.aux[i]=v
    raw,valid,_=v;m=valid[0];s=self.summaries[i];s[0]=m.mean()
    if m.any():
     selected=raw[:,m];s[1:3]=selected.mean(1);s[3:5]=selected.std(1);s[5]=(selected[0]-selected[1]).mean()
    self.summarized[i]=True
   ms=normalize_bgrn(self.image[y:y+512,x:x+512],self.normalization)
   samples.append((ms,*self.aux[i]))
  return tuple(np.stack([s[i] for s in samples]) for i in range(4))
 def save_sar(self):
  assert self.summarized.all()
  h,w=self.job['shape'];coords=np.asarray(self.coords)
  yy,xx=np.meshgrid(np.arange((h+63)//64)*64+32,np.arange((w+63)//64)*64+32,indexing='ij')
  centers=np.stack((yy.ravel(),xx.ravel()),1)
  nearest=np.argmin(((centers[:,None,:]-(coords[None,:,:]+256))**2).sum(2),1)
  np.save(Path(self.job['output'])/'sar_blocks.npy',self.summaries[nearest],allow_pickle=False)
  self.reader.close()

def load_mos():
 sys.path.insert(0,str(B/'source_snapshot/mos'))
 sys.path.insert(0,str(B/'code/primary'))
 import torch
 torch.set_num_threads(4)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 from tools import train_convnext_upgrade_full120 as train
 from tcmmsline3.convnext_pyramid_model import ConvNeXtPyramidSAR
 cfg=read(B/'source_snapshot/mos_training_config.json');spec=read(B/'source_snapshot/primary_source_config.json')['experts']['pyramid_e009']
 path=B/'weights/mos_e009_ema_delta.pt';assert sha(path)==spec['sha256']
 payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 assert payload['pretraining_identity']==train.pretraining_identity(cfg)
 for k,v in dict(role='candidate_ema_delta',epoch=9,updates=5400,config_sha256=spec['training_identity'],frozen_hash=spec['frozen_hash'],variant='pyramid').items():assert payload[k]==v,k
 relocated=copy.deepcopy(cfg)
 for key,value in [('convnext_checkpoint',B/'weights/convnext_large.pth'),('croma_checkpoint',B/'weights/croma_base.pt'),('convnext_source',B/'source_snapshot/mos/vendor/dinov3_convnext/convnext.py'),('croma_source',B/'source_snapshot/mos/vendor/CROMA/use_croma.py')]:relocated[key]=str(value)
 with torch.random.fork_rng(devices=[]):torch.manual_seed(cfg['seed']);model=ConvNeXtPyramidSAR(relocated,load_pretrained=True)
 mutable=tuple(model.mutable_keys());assert set(mutable)==set(payload['mutable_keys'])==set(payload['state_dict'])
 frozen=tuple(k for k in model.state_dict() if k not in set(mutable))
 assert train.state_hash(train.subset(model,frozen))==spec['frozen_hash'],'Frozen reconstruction changed'
 train.apply_delta(model,payload['state_dict']);model.set_stage('joint');model.cuda().eval().requires_grad_(False);del payload;gc.collect()
 import fbp25_ensemble_inference as old
 src=inspect.getsource(old.predict_loaded)
 for before,after in [('(25, h, w)','(24, h, w)'),('(25, 512, w)','(24, 512, w)'),('(len(xx), 25, 512, 512)','(len(xx), 24, 512, 512)')]:assert src.count(before)==1;src=src.replace(before,after)
 ns=dict(vars(old));exec(compile(src,'<frozen-native24-predictor>','exec'),ns);predict=ns['predict_loaded']
 norm=read(B/'source_snapshot/primary_source_config.json')['normalization']
 return model,predict,norm,spec

def mos(job):
 global _MOS
 reused=_MOS is not None
 if _MOS is None:_MOS=load_mos()
 model,predict,norm,spec=_MOS
 image=load_image(job);cache=Inputs(job,image,norm);results={}
 for i,mode in enumerate(('off','on') if job['has_sar'] else ('off',)):
  out=output(job,i+2)
  def progress(**v):write(Path(job['output'])/'mos_progress.json',dict(time_unix=time.time(),**v))
  results[mode]=predict(model,image,out,cache,variant='pyramid',normalization=norm,sar_mode=mode,report=progress);out.flush()
 if not job['has_sar']:
  dst=output(job,3);dst[:]=output(job,2);dst.flush()
 cache.save_sar();return dict(frozen_hash=spec['frozen_hash'],results=results,frozen_model_reused=reused)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--expert',required=True,choices=['v2l_e17','v3l_e26','mos']);args=p.parse_args()
 paths=(line.strip() for line in sys.stdin) if args.job=='-' else [args.job]
 initialized=False
 for job_path in paths:
  if not job_path:continue
  job=read(job_path);os.environ['CUDA_VISIBLE_DEVICES']=job['gpu'];opened=guard(job);start=time.time()
  if not initialized:
   import torch
   torch.set_num_interop_threads(1);initialized=True
  result=mos(job) if args.expert=='mos' else dino(job,args.expert)
  write(Path(job['output'])/(args.expert+'_completed.json'),dict(seconds=time.time()-start,result=result,raw_data_files=sorted(opened),labels_read=False,attempt_id=job.get('attempt_id')))
  gc.collect();torch.cuda.empty_cache()
