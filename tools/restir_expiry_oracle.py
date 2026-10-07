#!/usr/bin/env python3
"""CPU-only independent history-expiry counterexample; no renderer/compiler.

By default enumerate rational probabilities exactly. --iid additionally runs
131072 independently seeded paths with the production DI caps8/64/16 and
32 warmup+256 measured frames. It writes only a NEW chosen JSON path.
"""
from fractions import Fraction as F
from collections import defaultdict
import argparse,json
from pathlib import Path

def exact(mode):
    states={(0,0,F(0)):F(1)}; rows=[]
    for frame in range(7):
        new=defaultdict(F)
        for (M,age,mean),prob in states.items():
            for fresh in (F(1),F(3)):
                m=min(M,2) if age<1 else 0
                total=fresh+m*mean; newmean=total/(1+m)
                if mode=='selected_age':
                    oldp=m*mean/total
                    new[(1+m,age+1,newmean)]+=prob*oldp/2
                    new[(1+m,0,newmean)]+=prob*(1-oldp)/2
                else:new[(1+m,age+1 if m else 0,newmean)]+=prob/2
        states={k:v for k,v in new.items() if v}
        assert sum(states.values())==1
        expectation=sum(mean*prob for (M,age,mean),prob in states.items())
        rows.append({'frame':frame,'expectation_fraction':str(expectation),'expectation':float(expectation),'states':len(states)})
    return rows

def iid():
    import numpy as np
    n=131072;frames=288;warmup=32;result={}
    for mode in ('selected_age','chain_age','no_expiry'):
        rng=np.random.default_rng(2938495)
        mean=np.zeros(n);age=np.zeros(n,dtype=np.int32);M=np.zeros(n,dtype=np.int32);accumulated=np.zeros(n);dropped=0
        for frame in range(frames):
            fresh=rng.binomial(8,.125,n).astype(float)+1e-6
            u=rng.random(n)
            valid=(M>0)&((age<16) if mode!='no_expiry' else True)
            dropped+=int(((M>0)&~valid).sum())
            m=np.where(valid,np.minimum(M,64),0)
            total=8*fresh+m*mean
            choose=u*total<m*mean
            mean=total/(8+m)
            age=np.where(choose,age+1,0) if mode!='chain_age' else np.where(valid,age+1,0)
            M=8+m
            if frame>=warmup:accumulated+=mean/(frames-warmup)
        result[mode]={'mean':float(accumulated.mean()),'independent_path_se':float(accumulated.std(ddof=1)/np.sqrt(n)),
                      'expected':1.000001,'resets':dropped,'paths':n,'frames':frames,'warmup':warmup,'seed':2938495}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--iid',action='store_true');p.add_argument('--out',type=Path)
    a=p.parse_args();r={mode:exact(mode) for mode in ('selected_age','chain_age')}
    assert r['selected_age'][4]['expectation_fraction']=='182666318/91265265'
    assert all(row['expectation_fraction']=='2' for row in r['chain_age'])
    if a.iid:r['iid']=iid()
    if a.out:
        with a.out.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
