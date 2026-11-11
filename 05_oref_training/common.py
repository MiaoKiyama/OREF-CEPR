import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',NUMPY_MADVISE_HUGEPAGE='0')
import hashlib,json,sys,time,zipfile
from pathlib import Path
import numpy as np
PACKAGE=Path(__file__).resolve().parent.parent
ROOT=Path(os.environ.get('FBP83_OUTPUT',str(PACKAGE.parent/'run'))).resolve()
def read(p):return json.loads(Path(p).read_text())

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def progress(stage,**kw):write(ROOT/'progress.json',dict(stage=stage,time_unix=time.time(),**kw))

def load(p,*members):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in members}
