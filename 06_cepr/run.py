"""Run the frozen FBP C/E/P/R backend using the verified final bundle runtime."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--runtime-dir',required=True,type=Path)
    p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['all','posteriors','groups','spatial'],default='all')
    p.add_argument('--gpu',default='0');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'07_final_inference/infer_fbp.py'),'--model',a.model,
                    '--unpack-only','--runtime-dir',str(a.runtime_dir)],check=True)
    subprocess.run([sys.executable,str(a.runtime_dir/'infer_cached.py'),'--inputs',a.inputs,'--output',a.output,
                    '--stage',a.stage,'--gpu',a.gpu,'--workers',str(a.workers)],check=True)

if __name__=='__main__':main()
