"""ACE appendix A.1 scoring, independent of inference and network dependencies.

Unreleased evaluator ambiguities are explicit: bbox_scale=1.1 means 10% total
size dilation; best-overlap selection precedes side rejection. Do not claim
bit-for-bit parity with the authors' evaluator until these are confirmed.
"""
from dataclasses import dataclass
import numpy as np
from .metrics import similarity_align


@dataclass
class Hand:
    side: str
    joints: np.ndarray  # (21,3), camera meters, including translation
    vertices: np.ndarray  # camera meters
    translation: np.ndarray  # MANO parameter, NOT anatomical wrist
    rotation: np.ndarray
    anchors: np.ndarray  # model's own (21,2) pixel outputs
    existence: float = 1.


def project(points, K):
    p = np.asarray(points) @ np.asarray(K).T
    with np.errstate(divide='ignore', invalid='ignore'):
        return p[:, :2] / p[:, 2:]


def inside(hand, K, size):
    uv = project(hand.joints, K)
    return (np.isfinite(uv).all(1) & (hand.joints[:,2] > .01)
            & (uv[:,0] >= 0) & (uv[:,0] < size[0])
            & (uv[:,1] >= 0) & (uv[:,1] < size[1]))


def box(hand, K, scale=1.):
    uv = project(hand.vertices, K)
    uv = uv[np.isfinite(uv).all(1)]
    if not len(uv): return np.zeros(4)
    lo, hi = uv.min(0), uv.max(0)
    center, half = (lo+hi)/2, (hi-lo)*scale/2
    return np.r_[center-half, center+half]


def iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union = np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-intersection
    return float(intersection/union) if union > 0 else 0.


def match(predictions, targets, K, size, bbox_scale=1.1, same_side_first=False):
    gt = [g for g in targets if inside(g,K,size).any()]
    pred = [p for p in predictions if p.existence > .5 and inside(p,K,size).any()]
    winners = {}
    for pi,p in enumerate(pred):
        overlaps = [iou(box(p,K),box(g,K,bbox_scale))
                    if not same_side_first or p.side == g.side else 0. for g in gt]
        if not overlaps: continue
        gi = int(np.argmax(overlaps)); overlap = overlaps[gi]
        if overlap <= 0 or p.side != gt[gi].side: continue
        if gi not in winners or overlap > winners[gi][1]: winners[gi] = (pi,overlap)
    pairs = [(pred[pi],gt[gi]) for gi,(pi,_) in winners.items()]
    missing = [g for gi,g in enumerate(gt) if gi not in winners]
    used = {pi for pi,_ in winners.values()}
    extra = [p for pi,p in enumerate(pred) if pi not in used]
    return pairs, missing, extra


def pose_errors(p,g):
    relp, relg = p.joints-p.joints[:1], g.joints-g.joints[:1]
    cosine = np.clip((np.trace(p.rotation.T@g.rotation)-1)/2,-1,1)
    return {'MPJPE-p_mm': float(np.linalg.norm(relp-relg,axis=1).mean()*1000),
            'PA-p_mm': float(np.linalg.norm(similarity_align(p.joints,g.joints)-g.joints,axis=1).mean()*1000),
            'GO-p_deg': float(np.degrees(np.arccos(cosine))),
            'CT-p_m': float(np.linalg.norm(p.translation-g.translation))}


def score(frames, canonical, bbox_scale=1.1, same_side_first=False):
    """Frames: (predictions, GT, K, (width,height)); canonical keyed by side.

    Canonical poses must come from mean-shape MANO, identity orientation,
    zero articulation, zero translation. GT never enters any model call.
    """
    errors = []; matched_errors=[]; epe = []; matched_epe=[]
    counts = {s:dict(TP=0,FP=0,FN=0) for s in ('left','right')}
    correct = 0; nframes = 0
    for pred,gt,K,size in frames:
        pairs, missing, extra = match(pred,gt,K,size,bbox_scale,same_side_first)
        nframes += 1; correct += not (missing or extra)
        for p,g in pairs:
            counts[g.side]['TP'] += 1
            error=pose_errors(p,g); errors.append(error); matched_errors.append(error)
            visible = inside(g,K,size)
            pixel_errors=np.linalg.norm(p.anchors[visible]-project(g.joints,K)[visible],axis=1)
            epe.extend(pixel_errors); matched_epe.extend(pixel_errors)
        for g in missing:
            counts[g.side]['FN'] += 1; errors.append(pose_errors(canonical[g.side],g))
            epe.extend([float(np.hypot(*size))]*int(inside(g,K,size).sum()))
        for p in extra: counts[p.side]['FP'] += 1
    def detection(c):
        tp,fp,fn = (c[k] for k in ('TP','FP','FN'))
        return dict(**c, recall=tp/(tp+fn) if tp+fn else None,
                    precision=tp/(tp+fp) if tp+fp else None,
                    F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    totals = {k:sum(c[k] for c in counts.values()) for k in ('TP','FP','FN')}
    result = detection(totals)
    result.update({k:float(np.mean([e[k] for e in errors])) if errors else None
                   for k in ('MPJPE-p_mm','PA-p_mm','GO-p_deg','CT-p_m')})
    result.update({'EPE2D-p_px':float(np.mean(epe)) if epe else None,
                   'FAcc':correct/nframes if nframes else None, 'frames':nframes,
                   'per_side':{s:detection(c) for s,c in counts.items()},
                   'bbox_scale':bbox_scale, 'same_side_first':same_side_first,
                   'author_evaluator_parity_verified':False})
    result['matched_only']={k.replace('-p',''):float(np.mean([e[k] for e in matched_errors])) if matched_errors else None
                           for k in ('MPJPE-p_mm','PA-p_mm','GO-p_deg','CT-p_m')}
    result['matched_only']['EPE2D_px']=float(np.mean(matched_epe)) if matched_epe else None
    return result
