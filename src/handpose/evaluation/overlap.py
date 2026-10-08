"""Paired failure overlap. Oracle scores are diagnostic upper bounds only."""
import numpy as np
from scipy.stats import spearmanr


def compare_errors(a, b, gt, threshold_mm=20, cap_mm=100):
    a, b, gt = (np.asarray(x, float) for x in (a, b, gt))
    if a.shape != b.shape or a.shape != gt.shape or a.ndim != 3:
        raise ValueError('Expected matching frame/joint/xyz arrays')
    eligible = np.isfinite(gt).all(-1)
    va, vb = np.isfinite(a).all(-1) & eligible, np.isfinite(b).all(-1) & eligible
    ea, eb = np.linalg.norm(a-gt, axis=-1)*1000, np.linalg.norm(b-gt, axis=-1)*1000
    common = va & vb
    fa, fb = eligible & (~va | (ea > threshold_mm)), eligible & (~vb | (eb > threshold_mm))
    rate = lambda mask: float(mask.sum()/eligible.sum()) if eligible.any() else None
    mean = lambda x: float(np.mean(x)) if len(x) else None
    def correlation(x, y):
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return None
        return float(spearmanr(x, y).statistic)
    da, db = (a-gt)[common], (b-gt)[common]
    norm = np.linalg.norm(da, axis=-1)*np.linalg.norm(db, axis=-1)
    nonzero = norm > 1e-12
    cosine = (da[nonzero]*db[nonzero]).sum(-1)/norm[nonzero]
    la, lb = np.where(va, np.minimum(ea, cap_mm), cap_mm), np.where(vb, np.minimum(eb, cap_mm), cap_mm)
    # Averages only on common observations. Never interpret the oracle as inference.
    avg_error = np.linalg.norm((a+b)/2-gt, axis=-1)*1000
    result = dict(threshold_mm=threshold_mm, eligible_joint_slots=int(eligible.sum()),
        common_observed_slots=int(common.sum()), common_coverage=rate(common),
        both_missing_rate=rate(eligible & ~va & ~vb),
        only_a_observed_rate=rate(va & ~vb), only_b_observed_rate=rate(vb & ~va),
        both_failure_rate=rate(fa & fb),
        failure_jaccard=float((fa & fb).sum()/(fa | fb).sum()) if (fa | fb).any() else None,
        a_failure_b_success_rate=rate(fa & ~fb), b_failure_a_success_rate=rate(fb & ~fa),
        error_spearman_common=correlation(ea[common], eb[common]),
        error_vector_cosine_common=mean(cosine),
        a_common_mpjpe_mm=mean(ea[common]), b_common_mpjpe_mm=mean(eb[common]),
        unweighted_average_common_mpjpe_mm=mean(avg_error[common]),
        a_capped_mm=mean(la[eligible]), b_capped_mm=mean(lb[eligible]),
        oracle_joint_selection_capped_mm=mean(np.minimum(la, lb)[eligible]),
        oracle_uses_ground_truth=True)
    result['per_joint'] = []
    for j in range(gt.shape[1]):
        mask = common[:, j]
        result['per_joint'].append(dict(joint=j, common_slots=int(mask.sum()),
            error_spearman=correlation(ea[mask,j], eb[mask,j]),
            a_mpjpe_mm=mean(ea[mask,j]), b_mpjpe_mm=mean(eb[mask,j])))
    return result
