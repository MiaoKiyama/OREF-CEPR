"""Verify the sealed package and report the runtime; no dataset or labels needed."""
import argparse,ctypes,hashlib,importlib.metadata,json,platform,shutil,sys
from pathlib import Path
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime-only',action='store_true');a=p.parse_args()
 if not a.runtime_only:
  manifest=json.loads((B/'MANIFEST.sha256.json').read_text());size=0
  for row in manifest['files']:
   path=(B/row['path']).resolve();assert path.is_relative_to(B),row['path']
   assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
   size+=row['bytes']
  print(json.dumps(dict(integrity='passed',files=len(manifest['files']),bytes=size)),flush=True)
 import numpy,scipy,torch,PIL,rasterio,timm,einops
 expected=json.loads((B/'environment/runtime.json').read_text())
 report=dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,scipy=scipy.__version__,Pillow=PIL.__version__,rasterio=rasterio.__version__,timm=timm.__version__,einops=einops.__version__,cuda_available=torch.cuda.is_available())
 print(json.dumps(report,indent=2));assert sys.platform=='linux','Validated runtime is Linux x86_64'
 assert torch.cuda.is_available(),'A CUDA GPU is required'
 assert torch.__version__.split('+')[0]==expected['torch'] and torch.version.cuda==expected['cuda'],'Use the recorded PyTorch/CUDA versions for exact reproduction'
 versions=json.loads((B/'environment/core_versions.json').read_text())
 for name,version in versions.items():assert importlib.metadata.version(name)==version,('Dependency version mismatch',name,version,importlib.metadata.version(name))
 assert shutil.which('g++'),'g++ with OpenMP support is required for the object-graph kernel'
 for name in ['gains','components','neighborhood']:ctypes.CDLL(str(B/'code'/(name+'.so')))
 print('Package/runtime checks passed. No labels have been opened.')
if __name__=='__main__':main()
