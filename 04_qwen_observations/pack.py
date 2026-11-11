"""Pack 32 regions into the 33x18 Q value/knownness sidecar."""
import argparse
import json
from pathlib import Path
import numpy as np

def pack(records,output):
    scenes={}
    for path in sorted(Path(records).glob('*.json')):
        row=json.loads(path.read_text());request=row['request'];sid=request['scene'];roi=request['roi']
        if not 0<=roi<32:raise ValueError('ROI must be 0..31')
        scenes.setdefault(sid,{})
        if roi in scenes[sid]:raise ValueError('Duplicate ROI')
        scenes[sid][roi]=row
    if not scenes:raise ValueError('No generation records')
    Path(output).mkdir(parents=True,exist_ok=True)
    for sid,rows in scenes.items():
        if set(rows)!=set(range(32)):raise ValueError(f'{sid}: expected all 32 observations')
        hashes={r['request']['image_sha256'] for r in rows.values()}
        if len(hashes)!=1:raise ValueError('Conflicting source image identities')
        q=np.zeros((33,18),np.float32)
        for roi,row in rows.items():
            for j,v in enumerate(row['values'][:9]):
                if v is not None:
                    if type(v)!=int or v not in (0,1):raise ValueError('Invalid observation value')
                    q[roi,j]=v;q[roi,9+j]=1
        np.savez_compressed(Path(output)/(sid+'.npz'),q=q,image_sha256=np.array(next(iter(hashes))))
    return len(scenes)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(dict(scenes=pack(a.records,a.output))))
