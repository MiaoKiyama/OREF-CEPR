from common import *
import gc,resource
from scipy.ndimage import distance_transform_edt
from vendor import consensus as c,multi as m,region as r,refine as t,refresh as w,apply as a

def regenerated_pixels(obs,policy):
 pred=obs['current'];dest=pred.copy();best=np.full(pred.shape,-1,np.float32)
 for center,neighbor in [(np.s_[1:,:],np.s_[:-1,:]),(np.s_[:,1:],np.s_[:,:-1]),(np.s_[:,:-1],np.s_[:,1:]),(np.s_[:-1,:],np.s_[1:,:])]:
  use=(pred[center]!=pred[neighbor])&(obs['mean'][neighbor]>best[center]);best[center][use]=obs['mean'][neighbor][use];dest[center][use]=pred[neighbor][use]
 boundary=best>=0;del best;out=pred.copy();old=np.zeros_like(pred);oldbin=old.copy()
 if boundary.any():
  distance,near=distance_transform_edt(~boundary,return_indices=True);anchor=pred[tuple(near)];dst=dest[tuple(near)];del near,dest
  common=obs['agree']&(anchor==pred)&(dst!=pred);del anchor,boundary
  for stage,low,edges in [('consensus',-1,np.array([0,1,2,4,8,16,32,64])),('radius',64,np.array([96,128,192,256,384,512])),('radius1024',512,np.array([640,768,896,1024]))]:
   use=common&(distance>low)&(distance<=edges[-1]);db=np.searchsorted(edges,distance[use],side='left').astype(np.uint32)
   code=pred[use].astype(np.uint32)|(dst[use].astype(np.uint32)<<5)|(db<<10)|(obs['minbin'][use].astype(np.uint32)<<14)|(obs['meanbin'][use].astype(np.uint32)<<18)
   if stage=='consensus':
    values=a.consensus(code,policy[stage]['rules']);hi=obs['minbin'][use]>=8;values[hi]=a.band(code,policy['highconf']['rules'],8)[hi]
   else:values=a.band(code,policy[stage]['rules'])
   out[use]=values;old[use]=dst[use];oldbin[use]=np.searchsorted(m.OLD_EDGES,edges[db],side='left').astype(np.uint8)
  del distance,dst,common,db,code,use
 candidates,distances=m.neighborhood(pred,obs['mean'],out,obs['agree'])
 sig=m.encode(dict(incumbent=out,raw=obs['base'],old=old,oldbin=oldbin,minbin=obs['minbin'],meanbin=obs['meanbin'],candidate1=candidates[0],candidate2=candidates[1],distance1=distances[0],distance2=distances[1]))
 del candidates,distances,old,oldbin,obs;gc.collect();flat=out.ravel();s=sig.ravel()
 for lo in range(0,len(flat),1000000):
  sl=slice(lo,min(lo+1000000,len(flat)));flat[sl]=a.multi(a.decode(s[sl]),policy['multicandidate'])
 return sig,out

def region_inputs(ch):
 with np.load(ch['path'],allow_pickle=False) as z:
  if 'region_raw' in z.files:
   raw=np.ascontiguousarray(z['region_raw'].T).view(np.float32).reshape(3,ch['count'])
   return z['region_x'],z['region_features'],raw
  prob=np.ascontiguousarray(z['top1_fp32'].T).view(np.float32).reshape(7,ch['count'])
 return r.pixel_features(prob)

def confidence_region(sig,pred,row,bundle):
 lab,reg=r.components(pred);n=len(reg['source']);lf=lab.ravel();sums=np.zeros((3,n),np.float64)
 for ch in row['chunks']:
  sl=slice(ch['start'],ch['start']+ch['count']);_,features,raw=region_inputs(ch)
  for j in range(3):sums[j]+=np.bincount(lf[sl],weights=raw[j],minlength=n)
 avg=sums/np.maximum(reg['area'][None],1)
 reg['features']=np.concatenate([reg.pop('geometry'),np.stack([r.bins(avg[0],r.CONF),r.bins(avg[1],r.RANGE),r.bins(avg[2],r.CONF)])])
 del sums,avg,raw,features;gc.collect()
 accept={name:a.accept(reg['source'],reg['destination'],np.zeros(n,np.uint8),reg['area_bin'],reg['features'],p) for name,p in bundle['branches'].items() if p['kind']=='region'}
 sf=sig.ravel();pf=pred.ravel();out=pf.copy();f=bundle['fusion'];cf=bundle['confidence_branch'];rg=bundle['region_branch'];choices=np.asarray(f['choice'])
 for ch in row['chunks']:
  sl=slice(ch['start'],ch['start']+ch['count']);src=pf[sl];dst,tag,_=r.pixel_proposal(sf[sl],src)
  x,features,_=region_inputs(ch)
  outputs={'incumbent':src};masks={}
  for name,p in bundle['branches'].items():
   if p['kind']=='pixel':use=a.accept(src,dst,tag,x,features,p);d=dst
   else:use=accept[name][lf[sl]];d=reg['destination'][lf[sl]]
   outputs[name]=np.where(use,d,src).astype(np.uint8);masks[name]=use
  outputs['confidence_priority']=np.where(masks[cf],outputs[cf],outputs[rg]);outputs['region_priority']=np.where(masks[rg],outputs[rg],outputs[cf]);selected=choices[src]
  for j,name in enumerate(f['names']):use=selected==j;out[sl][use]=outputs[name][use]
 return out.reshape(pred.shape)

def scene(row,part,cfg,control=False):
 sid=row['scene'];label='control' if control else 'two_oref';started=time.time();policy=cfg['spatial']
 def stamp(stage):write(ROOT/'workers'/(label+'_'+str(os.getpid())+'.json'),dict(scene=sid,stage=stage,time_unix=time.time()))
 stamp('loading optical/SAR candidate cache');keys=load(row['keys'],'keys')['keys']
 folder=Path(row['chunks'][0]['path']).parents[1];manifest={'shape':row['shape'],'chunks':[dict(file=Path(ch['path']).name,start=ch['start'],count=ch['count']) for ch in row['chunks']]}
 obs=c.observable_scene(folder,manifest,keys,part);initial_hash=arrsha(obs['current']);stamp('rebuilding pixel neighborhoods')
 sig,pixel=regenerated_pixels(obs,policy);del obs;gc.collect();hashes={'multicandidate':arrsha(pixel)}
 stamp('rebuilding confidence and connected regions');pred=confidence_region(sig,pixel,row,policy['confregion']);hashes['confregion']=arrsha(pred)
 stamp('rebuilding region context');lab,reg=t.observables(sig,pixel,pred);del pred;gc.collect()
 values=a.route(reg,policy['regionrefine'],a.REGION_STAGES);refined=values[lab];hashes['regionrefine']=arrsha(refined)
 extra=a.guard_features(reg,values);values=a.route(extra,policy['regionguard'],a.REGION_STAGES,True);guarded=values[lab];hashes['regionguard']=arrsha(guarded)
 del lab,reg,extra,values;gc.collect();stamp('rebuilding refreshed connected regions')
 lab,reg=w.observables(sig,refined,guarded);del sig,refined,guarded;gc.collect()
 values=a.route(reg,policy['regionrefresh'],a.REFRESH_STAGES);pred=values[lab];final_hash=arrsha(pred)
 if control:
  old=cfg['historical_stage_hashes'][sid]
  assert hashes==old['stage_array_sha256'],(hashes,old['stage_array_sha256'])
  assert final_hash==old['array_sha256'],(final_hash,old['array_sha256'])
 dest=ROOT/'predictions'/label/(sid+'.npz');a.save(dest,prediction=pred,pixel=pixel)
 record={'scene':sid,'path':str(dest),'sha256':sha(dest),'array_sha256':final_hash,'stage_array_sha256':hashes,'group_filled_sha256':initial_hash,'seconds':time.time()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'all_spatial_features_rebuilt':True,'historical_exact':control,'truth_read':False}
 write(ROOT/'receipts'/label/(sid+'.json'),record);stamp('scene sealed');return record
