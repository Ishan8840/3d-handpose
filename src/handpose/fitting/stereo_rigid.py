"""Stereo global alignment with fixed articulation; no reference-label inputs."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from handpose.geometry.camera import transform
from handpose.triangulation.weighted import triangulate

PALM = np.array([0, 5, 9, 13, 17])


@dataclass(frozen=True)
class RigidConfig:
    max_epipolar_px: float = 5.
    max_reprojection_px: float = 8.
    min_anchors: int = 3
    anchor_outlier_m: float = .025
    reprojection_scale_px: float = 4.
    anchor_sigma_m: float = .03
    rotation_sigma_deg: float = 30.
    min_depth_m: float = .08
    max_depth_m: float = 2.
    fit_palm_only: bool = False


def geometric_median(points, weights):
    center=np.average(points,axis=0,weights=weights)
    for _ in range(100):
        w=weights/np.maximum(np.linalg.norm(points-center,axis=1),1e-6)
        nxt=np.average(points,axis=0,weights=w)
        if np.linalg.norm(nxt-center)<1e-7: return nxt
        center=nxt
    return center


def fit_stereo_rigid(pose, uv_left, uv_right, confidence, cameras, config=RigidConfig()):
    """Return wrist-anchored and SE(3)-aligned poses in the left optical frame.

    pose: final ACE MANO21, not auxiliary head. Camera frames must agree.
    Confidence has shape (21,2); absent observations have zero confidence.
    Both outputs preserve every pairwise joint distance exactly. Failed fits
    stay NaN; a failed SE(3) solve is never silently replaced by monocular ACE.
    """
    missing=np.full((21,3),np.nan)
    if any(c.model!='opencv' or np.any(c.distortion) for c in cameras):
        raise ValueError('Rigid reprojection requires undistorted pinhole observations')
    result=dict(anchored=missing.copy(),se3=missing.copy(),rotation=np.full((3,3),np.nan),
                stereo_valid=np.zeros(21,bool),anchor_inliers=np.zeros(21,bool),status='missing_pose')
    pose=np.asarray(pose,float)
    if pose.shape!=(21,3) or not np.isfinite(pose).all(): return result
    confidence=np.asarray(confidence,float)
    xyz,valid=triangulate(*cameras,uv_left,uv_right,confidence,
                         max_epipolar_px=config.max_epipolar_px,
                         max_reprojection_px=config.max_reprojection_px)
    valid &= (xyz[:,2]>=config.min_depth_m)&(xyz[:,2]<=config.max_depth_m)
    result['stereo_valid']=valid
    ids=PALM[valid[PALM]]
    result['status']='insufficient_palm_anchors'
    if len(ids)<config.min_anchors: return result
    offsets=pose-pose[:1]
    # First-order stereo depth uncertainty grows approximately as z^2.
    weights=np.minimum(confidence[ids,0],confidence[ids,1])/np.maximum(xyz[ids,2],.08)**4
    candidates=xyz[ids]-offsets[ids]
    wrist=geometric_median(candidates,weights)
    distances=np.linalg.norm(candidates-wrist,axis=1)
    keep=distances<=config.anchor_outlier_m
    if keep.sum()<config.min_anchors: return result
    ids=ids[keep]; weights=weights[keep]; candidates=candidates[keep]
    wrist=geometric_median(candidates,weights)
    anchored=offsets+wrist
    if any(np.min(transform(anchored,c.T_camera_from_left)[:,2])<=config.min_depth_m
           or np.max(transform(anchored,c.T_camera_from_left)[:,2])>config.max_depth_m for c in cameras):
        result['status']='cheirality_failure'; return result
    result['anchored']=anchored; result['anchor_inliers'][ids]=True
    fit_ids=ids if config.fit_palm_only else np.flatnonzero(valid)
    pixels=[np.asarray(uv_left)[fit_ids],np.asarray(uv_right)[fit_ids]]
    rot_sigma=np.radians(config.rotation_sigma_deg)
    def residual(params):
        pts=offsets@Rotation.from_rotvec(params[:3]).as_matrix().T+params[3:]
        errors=[]
        for view,cam in enumerate(cameras):
            # Smooth projection during optimization; final cheirality is checked.
            local=transform(pts[fit_ids],cam.T_camera_from_left)
            projected=(local/local[:,2:3].clip(.01))@cam.K.T
            errors.extend(((projected[:,:2]-pixels[view])*np.sqrt(confidence[fit_ids,view,None])).ravel())
            errors.extend(np.minimum(local[:,2]-.01,0)*10000)
        errors.extend((params[3:]-wrist)/config.anchor_sigma_m)
        errors.extend(params[:3]/rot_sigma)
        return np.asarray(errors)
    fit=least_squares(residual,np.r_[np.zeros(3),wrist],loss='soft_l1',
                      f_scale=config.reprojection_scale_px,max_nfev=100)
    R=Rotation.from_rotvec(fit.x[:3]).as_matrix()
    refined=offsets@R.T+fit.x[3:]
    result['status']='se3_rejected'
    if (fit.success and np.isfinite(refined).all() and
        np.linalg.norm(fit.x[:3])<=np.pi/3 and
        all(np.min(transform(refined,c.T_camera_from_left)[:,2])>=config.min_depth_m
            and np.max(transform(refined,c.T_camera_from_left)[:,2])<=config.max_depth_m for c in cameras)):
        result.update(se3=refined,rotation=R,status='ok')
    result['nfev']=fit.nfev
    return result
