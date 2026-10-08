import numpy as np


def fuse_metric(a,b,weight_b=.5,agreement_m=.05,fill_missing=True):
    """Fuse already co-framed metric predictions; never align or rescale.

    a is the preferred source on disagreement. The fixed blend weight must be
    selected on development data; detector confidence scores are not comparable.
    Returns source IDs: 0 missing, 1 a, 2 b fallback, 3 agreeing blend.
    """
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.shape[-1]!=3:raise ValueError('Pose shapes differ')
    if not 0<=weight_b<=1 or agreement_m<0:raise ValueError('Invalid fusion settings')
    va,vb=np.isfinite(a).all(-1),np.isfinite(b).all(-1)
    out=a.copy();source=np.where(va,1,0).astype(np.uint8)
    if fill_missing:
        take=~va&vb;out[take]=b[take];source[take]=2
    agree=va&vb&(np.linalg.norm(a-b,axis=-1)<=agreement_m)
    out[agree]=(1-weight_b)*a[agree]+weight_b*b[agree];source[agree]=3
    return out,source
