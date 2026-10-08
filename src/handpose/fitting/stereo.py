"""Robust metric stereo refinement with optional aligned multiview prior.

Inputs share the original left optical frame. Priors are gated in metric space;
no root or scale alignment is performed. No ground truth enters the objective.
"""
import numpy as np
from scipy.optimize import least_squares
from handpose.models.base import EDGES
from handpose.geometry.camera import transform


def refine_frame(initial,uv_left,uv_right,confidence,left,right,prior=None,prior_confidence=None,anatomy_weight=0.,prior_weight=0.,bone_lengths=None):
    x=np.asarray(initial,float).copy();valid=np.isfinite(x).all(-1);ids=np.flatnonzero(valid)
    if len(ids)<3:return x
    c=np.clip(np.nan_to_num(confidence),0,1)
    prior_good=np.zeros(21,bool)
    if prior is not None:
        prior=np.asarray(prior,float)
        # Gate failed identity/frame hypotheses instead of aligning them to measurements.
        prior_good=valid&np.isfinite(prior).all(-1)&(np.linalg.norm(prior-x,axis=-1)<.05)
        pc=np.clip(np.nan_to_num(prior_confidence if prior_confidence is not None else np.ones(21)),0,1)
    edges=[(i,j) for i,j in EDGES if valid[i] and valid[j] and bone_lengths is not None and np.isfinite(bone_lengths.get((i,j),np.nan))]
    mask_l=valid&np.isfinite(uv_left).all(-1);mask_r=valid&np.isfinite(uv_right).all(-1)
    def residual(flat):
        points=x.copy();points[ids]=flat.reshape(-1,3)
        terms=[((points[ids]-x[ids])/.01*np.sqrt(c[ids,None])).ravel()]
        for camera,uv,mask in ((left,uv_left,mask_l),(right,uv_right,mask_r)):
            terms.append(((camera.project(points[mask])-uv[mask])/2*np.sqrt(c[mask,None])).ravel())
        if prior_weight and prior_good.any():terms.append(((points[prior_good]-prior[prior_good])/.02*np.sqrt(prior_weight*pc[prior_good,None])).ravel())
        if anatomy_weight and edges:terms.append(np.array([(np.linalg.norm(points[i]-points[j])-bone_lengths[(i,j)])/.005*np.sqrt(anatomy_weight) for i,j in edges]))
        return np.concatenate(terms)
    result=least_squares(residual,x[ids].ravel(),loss='soft_l1',max_nfev=30)
    candidate=x.copy();candidate[ids]=result.x.reshape(-1,3)
    # Reject optimizer results violating either camera's cheirality.
    good=(transform(candidate[ids],left.T_camera_from_left)[:,2]>0)&(transform(candidate[ids],right.T_camera_from_left)[:,2]>0)
    if result.success and good.all():return candidate
    return x


def refine_sequence(joints,uv_left,uv_right,confidence,cameras,anatomy_weight=1.):
    lengths={}
    for edge in EDGES:
        values=np.linalg.norm(joints[:,edge[0]]-joints[:,edge[1]],axis=-1)
        lengths[edge]=float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan
    return np.array([refine_frame(x,l,r,c,*pair,anatomy_weight=anatomy_weight,bone_lengths=lengths) for x,l,r,c,pair in zip(joints,uv_left,uv_right,confidence,cameras)])
