"""Offline local weighted linear regression; explicit bounded gap filling."""
import numpy as np

def smooth(joints,confidence,timestamps_ns,window_seconds=.1,max_gap_seconds=0.):
    x=np.asarray(joints,float); c=np.asarray(confidence,float); t=np.asarray(timestamps_ns,dtype=np.float64)*1e-9
    if x.shape[:2]!=c.shape or len(t)!=len(x) or np.any(np.diff(t)<=0): raise ValueError('Invalid temporal arrays')
    y=x.copy(); source=np.where(np.isfinite(x).all(-1),1,0).astype(np.uint8)
    for j in range(x.shape[1]):
        good=np.isfinite(x[:,j]).all(1)&(c[:,j]>0)
        ids=np.flatnonzero(good)
        for i,ti in enumerate(t):
            if not good[i]:
                before=ids[t[ids]<ti]; after=ids[t[ids]>ti]
                if not len(before) or not len(after) or t[after[0]]-t[before[-1]]>max_gap_seconds: continue
            take=good&(np.abs(t-ti)<=window_seconds)
            if take.sum()<3:continue
            dt=t[take]-ti; w=np.sqrt(c[take,j]*np.exp(-.5*(dt/max(window_seconds/2,1e-8))**2))
            A=np.c_[np.ones(len(dt)),dt]
            coef=np.linalg.lstsq(A*w[:,None],x[take,j]*w[:,None],rcond=None)[0]
            y[i,j]=coef[0]; source[i,j]=3
    return y,source
