"""Compile the CEPR native sources and check expected-confusion arithmetic."""
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'06_cepr/runtime/code'

def main():
    names=['gains','components','context_refine','context_refresh','neighborhood']
    with tempfile.TemporaryDirectory() as tmp:
        output=Path(tmp)
        for name in names:
            subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-fPIC','-shared',str(ROOT/(name+'.cpp')),'-o',str(output/(name+'.so'))],check=True)
        lib=ctypes.CDLL(str(output/'gains.so'));ptr=ctypes.c_void_p
        lib.expected_cpu.argtypes=[ptr,ptr,ptr,ctypes.c_int64,ptr,ctypes.c_int]
        rng=np.random.default_rng(1337);n=37
        hist=rng.random((n,24));labels=rng.integers(1,25,(n,7),dtype=np.uint8);route=rng.integers(0,7,n,dtype=np.uint8)
        actual=np.empty((24,24),np.float64)
        lib.expected_cpu(hist.ctypes.data,labels.ctypes.data,route.ctypes.data,n,actual.ctypes.data,2)
        target=labels[np.arange(n),route]-1
        expected=np.stack([np.bincount(target,weights=hist[:,y],minlength=24) for y in range(24)])
        np.testing.assert_array_equal(actual,expected)
        print(json.dumps(dict(status='passed',compiled=names,expected_confusion_bitwise_equal=True)))

if __name__=='__main__':main()
