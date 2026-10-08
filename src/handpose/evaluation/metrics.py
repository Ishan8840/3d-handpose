"""All absolute errors use unaligned metric camera coordinates."""
import numpy as np
from handpose.models.base import FINGERTIPS


def similarity_align(pred, target):
    x, y = pred - pred.mean(0), target - target.mean(0)
    u, s, vt = np.linalg.svd(x.T @ y)
    correction = np.ones(3)
    correction[-1] = np.linalg.det(u @ vt)
    R = (u * correction) @ vt
    scale = np.sum(s * correction) / max(np.sum(x*x), 1e-20)
    return scale*x @ R + target.mean(0)


def evaluate(pred, gt, validity=None, gt_validity=None, timestamps_ns=None, missing_penalty_mm=100.):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (21,3):
        raise ValueError('Expected matching (frames,21,3) arrays')
    eligible = np.isfinite(gt).all(-1)
    if gt_validity is not None:
        eligible &= np.asarray(gt_validity, bool)
    observed = np.isfinite(pred).all(-1)
    if validity is not None:
        observed &= np.asarray(validity, bool)
    valid = observed & eligible
    error = np.linalg.norm(pred-gt, axis=-1)*1000
    error[~valid] = np.nan
    def mean(x):
        a = np.asarray(x)
        return float(np.mean(a[np.isfinite(a)])) if np.isfinite(a).any() else None
    e = error[valid]
    penalty = np.where(valid, np.minimum(error,missing_penalty_mm), missing_penalty_mm)
    rel_valid = valid & valid[:, :1]
    relative = np.linalg.norm((pred-pred[:, :1])-(gt-gt[:, :1]), axis=-1)*1000
    pa = []
    for p,g,v in zip(pred,gt,valid):
        if v.sum() >= 3 and np.linalg.matrix_rank(p[v]-p[v].mean(0)) >= 2:
            pa.extend(np.linalg.norm(similarity_align(p[v],g[v])-g[v],axis=-1)*1000)
    result = {
        'absolute_mpjpe_mm':mean(e), 'wrist_relative_mpjpe_mm':mean(relative[rel_valid]),
        'pa_mpjpe_mm':mean(pa), 'wrist_mm':mean(error[:,0]), 'fingertips_mm':mean(error[:,FINGERTIPS]),
        'joint_coverage':float(valid.sum()/eligible.sum()) if eligible.any() else None,
        'frame_coverage_any':float(valid.any(1)[eligible.any(1)].mean()) if eligible.any() else None,
        'frame_coverage_all_eligible':float((valid.sum(1)==eligible.sum(1))[eligible.any(1)].mean()) if eligible.any() else None,
        'eligible_joints':int(eligible.sum()), 'observed_joints':int(valid.sum()),
        'per_joint_mm':[mean(error[:,j]) for j in range(21)],
        'median_mm':float(np.median(e)) if len(e) else None,
        'p90_mm':float(np.percentile(e,90)) if len(e) else None,
        'p95_mm':float(np.percentile(e,95)) if len(e) else None,
        'capped_error_with_missing_penalty_mm':mean(penalty[eligible]), 'missing_penalty_mm':missing_penalty_mm,
        'depth_mae_mm':mean(np.abs(pred[...,2]-gt[...,2])[valid]*1000),
        'lateral_error_mm':mean(np.linalg.norm(pred[...,:2]-gt[...,:2],axis=-1)[valid]*1000),
        'tracking_discontinuities':int(np.sum(observed[1:] != observed[:-1])),
    }
    if timestamps_ns is not None:
        t=np.asarray(timestamps_ns,dtype=np.float64)*1e-9
        if len(t)!=len(pred) or np.any(np.diff(t)<=0):
            raise ValueError('Timestamps must strictly increase and match frames')
        if len(t)>=3:
            dt=np.diff(t)
            acceleration=lambda x: 2*np.diff(np.diff(x,axis=0)/dt[:,None,None],axis=0)/(dt[1:]+dt[:-1])[:,None,None]
            v=valid[:-2]&valid[1:-1]&valid[2:]
            result['acceleration_error_m_s2']=mean(np.linalg.norm(acceleration(pred)-acceleration(gt),axis=-1)[v])
    result['by_depth']={f'{lo}-{hi}m':mean(error[(gt[...,2]>=lo)&(gt[...,2]<hi)&valid]) for lo,hi in [(0,.3),(.3,.5),(.5,.8),(.8,2.)]}
    return result


def sequence_bootstrap(values, sequence_ids, seed=0, samples=2000):
    """Resample sequences, preserving frame weighting within sampled clusters."""
    values=np.asarray(values,float); ids=np.asarray(sequence_ids)
    unique=np.unique(ids); rng=np.random.default_rng(seed)
    if len(unique)<2:
        return None
    estimates=[np.nanmean(np.concatenate([values[ids==s] for s in rng.choice(unique,len(unique),replace=True)])) for _ in range(samples)]
    return np.percentile(estimates,[2.5,97.5]).tolist()
