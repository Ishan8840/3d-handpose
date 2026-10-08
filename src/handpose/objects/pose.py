"""Multistart stereo convex-silhouette fitting, independent of reference poses.

Uses mesh support directions, a convex silhouette approximation. Visible masks
under occlusion are biased shape observations; retain fit residual and hypothesis
spread rather than treating optimizer convergence as pose certainty.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def support(uv,directions):
    return (uv@directions.T).max(0)


def estimate_pose(vertices,masks,K,T_right_from_left,previous=None,starts=8):
    contours=[]
    for mask in masks:
        y,x=np.where(mask)
        if len(x)<20:return None
        contours.append(cv2.convexHull(np.c_[x,y].astype(np.float32)).reshape(-1,2))
    directions=np.c_[np.cos(np.arange(16)*np.pi/8),np.sin(np.arange(16)*np.pi/8)]
    targets=np.array([support(p,directions) for p in contours])
    centers=np.array([p.mean(0) for p in contours])
    P=[K@np.eye(4)[:3],K@T_right_from_left[:3]]
    h=cv2.triangulatePoints(P[0],P[1],centers[:1].T,centers[1:].T)[:,0]
    center=h[:3]/h[3]
    extent=np.ptp(vertices,axis=0).max()
    size=np.mean([np.ptp(p,axis=0).max() for p in contours])
    if not np.isfinite(center).all() or not .08<center[2]<2:
        z=np.clip(K[0,0]*extent/max(size,1),.1,1.5)
        center=np.linalg.inv(K)@np.r_[centers[0],1]*z
    offset=np.mean(vertices,axis=0); centered=vertices-offset
    def residual(x):
        xyz=centered@Rotation.from_rotvec(x[:3]).as_matrix().T+x[3:]
        result=[]
        for v,T in enumerate([np.eye(4),T_right_from_left]):
            p=xyz@T[:3,:3].T+T[:3,3]; uv=p@K.T
            uv=uv[:,:2]/np.maximum(uv[:,2:],.01)
            result.extend((support(uv,directions)-targets[v])/2.)
        return np.array(result)
    initial=[np.r_[r,center] for r in Rotation.random(starts,random_state=7).as_rotvec()]
    initial[0][:3]=0
    if previous is not None:
        r=previous[:3,:3];initial.insert(0,np.r_[Rotation.from_matrix(r).as_rotvec(),previous[:3,3]+r@offset])
    fits=[]
    for x in initial:
        x[3:]=np.clip(x[3:],[-2,-2,.08],[2,2,2])
        fit=least_squares(residual,x,bounds=([-np.inf]*3+[-2,-2,.08],[np.inf]*3+[2,2,2]),
                          loss='soft_l1',max_nfev=45,diff_step=1e-4)
        fits.append((np.mean(np.minimum(residual(fit.x)**2,25)),fit.x))
    fits.sort(key=lambda p:p[0]);score,x=fits[0]
    R=Rotation.from_rotvec(x[:3]).as_matrix(); T=np.eye(4);T[:3,:3]=R;T[:3,3]=x[3:]-R@offset
    plausible=[z for c,z in fits if c<score+1]
    translations=np.array([z[3:]-Rotation.from_rotvec(z[:3]).as_matrix()@offset for z in plausible])
    spread=float(np.max(np.linalg.norm(translations-T[:3,3],axis=1)))
    return dict(T=T,score=float(score),translation_hypothesis_spread_m=spread,
                accepted=bool(score<9 and spread<.08),orientation_observable=False)
