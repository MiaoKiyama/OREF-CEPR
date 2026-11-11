"""One-command frozen official30 inference from MS images and registered SAR."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',NUMPY_MADVISE_HUGEPAGE='0')
import argparse,ctypes,gc,json,signal,subprocess,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'code'))
from common import read,write,sha
WORKERS={}

def start_expert(expert,job):
 if expert not in WORKERS:
  log=(Path(job['output']).parents[1]/('worker_'+expert+'.log')).open('ab',buffering=0)
  process=subprocess.Popen([sys.executable,'-u',str(B/'raw_primary_worker.py'),'--job','-','--expert',expert],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
  log.close();WORKERS[expert]=process
 process=WORKERS[expert];assert process.poll() is None,('primary worker exited',expert,process.returncode)
 path=Path(job['output'])/(expert+'_job.json');write(path,job)
 process.stdin.write(str(path)+'\n');process.stdin.flush()
 return process,Path(job['output'])/(expert+'_completed.json')

def run_expert(expert,job):
 process,receipt=start_expert(expert,job)
 while True:
  if receipt.exists() and read(receipt).get('attempt_id')==job['attempt_id']:return
  if process.poll() is not None:raise RuntimeError('Primary worker failed: '+expert+'; see worker log')
  time.sleep(1)

def run_parallel(job,gpus):
 pending={}
 for expert,gpu in zip(['v2l_e17','v3l_e26','mos'],gpus):pending[expert]=start_expert(expert,dict(job,gpu=gpu))
 while pending:
  for expert,(process,receipt) in list(pending.items()):
   if receipt.exists() and read(receipt).get('attempt_id')==job['attempt_id']:del pending[expert];continue
   if process.poll() is not None:raise RuntimeError('Primary worker failed: '+expert+'; see worker log')
  write(Path(job['output']).parents[1]/'progress.json',dict(stage='primary_parallel',scene=job['scene'],pending_experts=list(pending),expert_gpus=gpus,time_unix=time.time()))
  if pending:time.sleep(5)

def close_workers():
 for p in WORKERS.values():
  if p.poll() is None:p.stdin.close()
 for p in WORKERS.values():
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:p.terminate();p.wait()
 WORKERS.clear()

def available_memory():return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024

def memory_file(name):
 if hasattr(os,'memfd_create'):return os.memfd_create(name,flags=0)
 # This server's Python was built against an older libc; the Linux kernel
 # still supplies the anonymous RAM-file syscall used by current Python.
 import platform
 assert platform.machine()=='x86_64','memfd fallback requires Linux x86_64'
 libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
 fd=libc.syscall(319,ctypes.c_char_p(name.encode()),ctypes.c_uint(0))
 if fd<0:raise OSError(ctypes.get_errno(),'memfd_create')
 return int(fd)

def one_scene(info,a):
 from raw_features import compact,graph
 from verify_inputs import verify_aux
 verify_aux(a.data_root,info['id'])
 output=Path(a.output).resolve()/'candidates'/info['id'];output.mkdir(parents=True,exist_ok=True)
 completed=output/'completed.json'
 image=Path(a.data_root).resolve()/'images'/(info['id']+'.tif')
 assert sha(image)==info['image_sha256'],'Frozen Q sidecar is valid only for the exact bound image: '+str(image)
 if completed.exists():return read(completed)
 job=dict(scene=info['id'],image=str(image),image_sha256=info['image_sha256'],shape=info['shape'],has_sar=info['has_sar'],data_root=str(Path(a.data_root).resolve()),output=str(output),gpu=a.gpu,attempt_id=uuid.uuid4().hex)
 checkpoint=output/'compact_completed.json'
 if checkpoint.exists():row=read(checkpoint)
 else:
  size=24*int(np.prod(info['shape']))*4;required=4*size+10*2**30
  while available_memory()<required:
   write(Path(a.output)/'progress.json',dict(stage='waiting_for_memory',required_bytes=required,available_bytes=available_memory(),scene=info['id'],time_unix=time.time()));time.sleep(15)
  fds=[];fields=[];beg=time.time()
  try:
   for i in range(4):
    fd=memory_file('fbp83_%d'%i);os.ftruncate(fd,size);fds.append(fd)
    path='/proc/%d/fd/%d'%(os.getpid(),fd);fields.append(np.memmap(path,dtype=np.float32,mode='r+',shape=(24,*info['shape'])))
   job['fields']=['/proc/%d/fd/%d'%(os.getpid(),fd) for fd in fds];write(output/'job.json',job)
   if a.expert_gpus:run_parallel(job,a.expert_gpus)
   else:
    for expert in ['v2l_e17','v3l_e26','mos']:
     write(Path(a.output)/'progress.json',dict(stage='primary_'+expert,scene=info['id'],seconds=time.time()-beg,time_unix=time.time()))
     run_expert(expert,job)
   row=compact(job,fields)
  finally:
   for x in fields:x._mmap.close()
   for fd in fds:os.close(fd)
   gc.collect()
 row=graph(row);row['image_sha256']=info['image_sha256'];row['features_sha256']=sha(row['keys']);row['observable_sha256']=sha(row['observable'])
 write(completed,row);return row

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',default='0');p.add_argument('--workers',type=int,default=2)
 p.add_argument('--expert-gpus',default='',help='Optional three GPUs for simultaneous DINOv2,DINOv3,MOS; --gpu still controls compaction and OREF')
 p.add_argument('--prepare-only',action='store_true');p.add_argument('--scene-limit',type=int,default=0,help='Diagnostic prefix; final inference always requires all 30 scenes')
 a=p.parse_args()
 if a.workers<1:p.error('--workers must be at least 1')
 if a.scene_limit<0 or a.scene_limit>30:p.error('--scene-limit must be between 0 and 30')
 if a.scene_limit and not a.prepare_only:p.error('--scene-limit requires --prepare-only')
 a.expert_gpus=a.expert_gpus.split(',') if a.expert_gpus else []
 if a.expert_gpus and (len(a.expert_gpus)!=3 or len(set(a.expert_gpus))!=3 or not all(x.isdigit() for x in a.expert_gpus)):p.error('--expert-gpus requires three distinct integer GPU IDs')
 a.output=str(Path(a.output).resolve());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 assert not out.is_relative_to(B),'Output must be outside the immutable package'
 os.environ['FBP83_OUTPUT']=str(out);os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 infos=read(B/'official30_images.json')['scenes'];assert len(infos)==30
 identity=dict(data_root=str(Path(a.data_root).resolve()),images_sha256=sha(B/'official30_images.json'),rules_sha256=sha(B/'rules.json'),assets_sha256=sha(B/'assets.json'))
 if (out/'input_identity.json').exists():assert read(out/'input_identity.json')==identity
 else:write(out/'input_identity.json',identity)
 rows=[];cursor=0
 for i,info in enumerate(infos[:a.scene_limit] if a.scene_limit else infos):
  row=one_scene(info,a);row.update(start=cursor,end=cursor+row['groups']);cursor=row['end'];rows.append(row)
  write(out/'progress.json',dict(stage='primary_complete',scenes=i+1,total_scenes=len(infos),time_unix=time.time()))
 manifest=dict(schema='fbp83_observable_inputs_v1',candidate_raw_order=['D2','D3','dual','pyramid_off','triple_off','pyramid_on','triple_on'],candidate_oref_order=['dual','triple_off','triple_on','pyramid_off','pyramid_on','D2','D3'],scenes=rows)
 write(out/'observables.json',manifest);close_workers()
 if a.prepare_only:return
 assert len(rows)==30,'Final frozen policy requires the complete official30 cohort; scene-limit is preparation only'
 subprocess.run([sys.executable,'-u',str(B/'infer_cached.py'),'--inputs',str(out/'observables.json'),'--output',str(out/'fusion'),'--gpu',a.gpu,'--workers',str(a.workers)],check=True)
 write(out/'completed.json',dict(status='completed',scenes=30,predictions=str(out/'fusion/predictions/two_oref'),raw_primary_forward=True,truth_used_in_inference=False,Q='frozen image-only sidecars verified against image SHA256',weights_changed=False,rules_changed=False,time_unix=time.time()))

if __name__=='__main__':
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(SystemExit('Inference stopped')))
 try:main()
 except Exception:
  print(traceback.format_exc(),file=sys.stderr,flush=True);raise
 finally:close_workers()
