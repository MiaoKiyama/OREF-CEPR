from common import *
import ctypes
from vendor import multi as m
lib=ctypes.CDLL(str(PACKAGE/"code/neighborhood.so"))
lib.alternatives.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*2
original_neighborhood=m.neighborhood
def fast_neighborhood(pred,conf,current,eligible):
 arrays=[np.ascontiguousarray(v,dtype=d) for v,d in zip((pred,conf,current,eligible),(np.uint8,np.float32,np.uint8,bool))]
 outputs=[np.empty(pred.shape,np.uint8) for _ in range(4)]
 lib.alternatives(*[v.ctypes.data for v in arrays+outputs],*pred.shape)
 return outputs[:2],outputs[2:]

def verify_kernel():
 rng=np.random.default_rng(261016)
 for shape in [(17,31),(269,275)]:
  p=rng.integers(1,25,size=shape,dtype=np.uint8);conf=rng.random(shape,dtype=np.float32);cur=rng.integers(1,25,size=shape,dtype=np.uint8);elig=rng.random(shape)>.3
  for left,right in zip(original_neighborhood(p,conf,cur,elig),fast_neighborhood(p,conf,cur,elig)):
   for x,y in zip(left,right):np.testing.assert_array_equal(x,y)
 write(ROOT/'kernel_verified.json',dict(exact=True,shapes=[[17,31],[269,275]],source_sha256=sha(PACKAGE/'code/neighborhood.cpp')))
