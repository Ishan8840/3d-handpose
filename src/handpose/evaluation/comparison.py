import numpy as np
from .metrics import sequence_bootstrap

def paired_comparison(a,b,gt,sequence_ids,penalty_mm=100):
    """Paired per-frame capped losses; CI clusters by sequence, not joints."""
    eligible=np.isfinite(gt).all(-1)
    def loss(p):
        e=np.linalg.norm(p-gt,axis=-1)*1000
        e=np.where(np.isfinite(p).all(-1),np.minimum(e,penalty_mm),penalty_mm)
        return np.divide(np.where(eligible,e,0).sum(1),eligible.sum(1),out=np.full(len(p),np.nan),where=eligible.sum(1)>0)
    delta=loss(a)-loss(b)
    return {'mean_paired_delta_mm':float(np.nanmean(delta)),'sequence_bootstrap_95ci':sequence_bootstrap(delta,sequence_ids),'negative_favors':'a'}

def error_coverage(pred,gt,confidence,thresholds=(0,.25,.5,.75,.9,.95)):
    from .metrics import evaluate
    return [{'threshold':t,**{k:v for k,v in evaluate(pred,gt,np.asarray(confidence)>=t).items() if k in ('absolute_mpjpe_mm','joint_coverage','capped_error_with_missing_penalty_mm')}} for t in thresholds]
