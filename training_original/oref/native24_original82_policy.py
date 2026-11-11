"""Original gain calibration and I, with every label-dependent choice fitted on training."""
import numpy as np
from native24_original82_reference import expected_cm,expected_gains,cm_from_groups,score_route,miou,fast_calibrate,class_threshold_search

def fit_prior(pack):
    cnt,labs=pack['counts'],pack['labels'];anchor=labs[:,2];valid=cnt[:,1:].sum(1,dtype=np.float64)
    total=pack['mass'].astype(np.float64);num=np.bincount(anchor,weights=valid,minlength=25)
    den=np.bincount(anchor,weights=total,minlength=25);global_rate=valid.sum()/max(total.sum(),1)
    rates=np.divide(num,den,out=np.full(25,global_rate),where=den>0)
    fixed=pack['fixed'];obs=pack['fixed_obs'];base=pack['baselines'][2]
    fallback=np.divide(base[1:],base.sum(0)[None,:],out=np.zeros((24,25),float),where=base.sum(0)[None,:]>0)
    prior=np.divide(fixed[1:],obs[None,:],out=fallback.copy(),where=obs[None,:]>0)
    # A never-observed prediction class falls back to the training true-class distribution.
    empty=base.sum(0)==0;prior[:,empty]=base[1:].sum(1)[:,None]/max(base.sum(),1)
    return dict(valid_rate=rates.tolist(),agreement_conditional=prior.tolist(),reference_pixels=int(pack['pixels']))

def expected_inputs(p,labels,mass,fixed_obs,pixels,prior):
    # Freeze gain units to the training population, independent of target set size.
    scale=prior['reference_pixels']/pixels
    m=mass.astype(np.float64)*np.asarray(prior['valid_rate'])[labels[:,2]]*scale
    hist=p.astype(np.float64)*m[:,None]
    fixed=np.asarray(prior['agreement_conditional'])[:,1:]*fixed_obs[None,1:]*scale
    return hist,fixed

def fit_calibration(p,pack,prior):
    labs=pack['labels'];hist,fixed=expected_inputs(p,labs,pack['mass'],pack['fixed_obs'],pack['pixels'],prior)
    base=np.full(len(labs),2,np.uint8);expected=fixed+expected_cm(hist,labs,base)
    gains=expected_gains(expected,hist,labs,base)
    route,cm,threshold=fast_calibrate(gains,pack['fixed'],pack['counts'],labs,pack['mass']>0)
    route,cm,policy=class_threshold_search(pack['counts'],labs,gains,pack['fixed'],route,miou)
    assert np.array_equal(cm,score_route(pack['fixed'],pack['counts'],labs,route))
    global_threshold=threshold if threshold is None or np.isfinite(threshold) else 'never_switch'
    result=dict(global_threshold=global_threshold,**policy,training_class_miou=100*miou(cm),prior=prior)
    replay=apply_thresholds(gains,labs,result);assert np.array_equal(route,replay)
    return result,route

def apply_thresholds(gains,labels,policy):
    route=np.full(len(labels),2,np.uint8);action=gains.argmax(1).astype(np.uint8);gain=gains.max(1)
    for cls in range(1,25):
        t=policy['source_class_thresholds'][str(cls)]
        if t=='keep_initial_global_policy':t=policy['global_threshold']
        if t is None or t=='never_switch':continue
        ix=(labels[:,2]==cls)&(gain>float(t));route[ix]=action[ix]
    return route

def iterate_I(hist,labels,fixed,initial,dynamic,max_iterations=12,observer=None):
    """Same 8192-group cap/backtracking and frozen/dynamic state semantics as original."""
    route=initial.copy();expected=fixed+expected_cm(hist,labels,route);initial_expected=expected.copy();trace=[]
    mass=hist.sum(1)
    def em(x):
        dg=x.diagonal();return float((dg/np.maximum(x.sum(0)+x.sum(1)-dg,1)).mean())
    for iteration in range(max_iterations):
        gains=expected_gains(expected if dynamic else initial_expected,hist,labels,route)
        action=gains.argmax(1).astype(np.uint8);gain=gains.max(1);pool=np.flatnonzero((gain>1e-14)&(mass>0))
        if not len(pool):break
        cap=min(len(pool),8192);ix=pool[np.argpartition(-gain[pool],cap-1)[:cap]]
        old=route[ix].copy();new=action[ix]
        delta=expected_cm(hist[ix],labels[ix],new)-expected_cm(hist[ix],labels[ix],old)
        while em(expected+delta)<em(expected)-1e-12 and len(ix)>1:
            ix=ix[np.argsort(-gain[ix])[:len(ix)//2]];old=route[ix].copy();new=action[ix]
            delta=expected_cm(hist[ix],labels[ix],new)-expected_cm(hist[ix],labels[ix],old)
        if em(expected+delta)<em(expected)-1e-12:break
        expected+=delta;route[ix]=new
        row=dict(iteration=iteration+1,groups=len(ix),predicted_miou=100*em(expected));trace.append(row)
        if observer is not None:observer(iteration+1,ix,old,new,row)
    return route,trace

def fit_I(p,pack,policy,initial):
    hist,fixed=expected_inputs(p,pack['labels'],pack['mass'],pack['fixed_obs'],pack['pixels'],policy['prior'])
    output={};traces={}
    for dynamic,name in [(False,'I_frozen'),(True,'I_dynamic')]:
        cm=score_route(pack['fixed'],pack['counts'],pack['labels'],initial)
        best=[0,100*miou(cm)]
        def observe(step,ix,old,new,row):
            cm[:]+=cm_from_groups(pack['counts'][ix],pack['labels'][ix],new)-cm_from_groups(pack['counts'][ix],pack['labels'][ix],old)
            s=100*miou(cm);row['training_miou']=s
            if s>best[1]+1e-10:best[:]=[step,s]
        _,trace=iterate_I(hist,pack['labels'],fixed,initial,dynamic,12,observe)
        output[name]=dict(iterations=best[0],training_miou=best[1],dynamic=dynamic);traces[name]=trace
    return dict(policy,I=output,I_training_trace=traces)

def deploy(p,labels,mass,fixed_obs,pixels,policy):
    """No counts, truth masks, evaluation confusions, or target-side selection inputs."""
    hist,fixed=expected_inputs(p,labels,mass,fixed_obs,pixels,policy['prior'])
    base=np.full(len(labels),2,np.uint8);gains=expected_gains(fixed+expected_cm(hist,labels,base),hist,labels,base)
    route=apply_thresholds(gains,labels,policy);result={'class':route};traces={}
    del gains
    for name in ('I_frozen','I_dynamic'):
        v=policy['I'][name]
        result[name],traces[name]=iterate_I(hist,labels,fixed,route,v['dynamic'],v['iterations'])
    return result,traces
