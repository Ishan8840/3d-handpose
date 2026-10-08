"""GT-free rigid reconstruction from synchronized official ACE exports.

All 2D exports, final MANO joints and calibration must describe the same
processed camera frame. Final poses NPZ requires only joints_3d (F,21,3).
"""
import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
import numpy as np
from handpose.fitting.stereo_rigid import RigidConfig, fit_stereo_rigid
from handpose.geometry.camera import load_stereo
from handpose.models.base import JOINT_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['left-predictions','right-predictions','final-poses','calibration','timestamps','output']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=['anchored','se3'],default='anchored')
    p.add_argument('--config',help='JSON RigidConfig fields; default uses development-selected config')
    a=p.parse_args(); cams=load_stereo(a.calibration)
    config=RigidConfig(**json.load(open(a.config))) if a.config else RigidConfig(anchor_outlier_m=.04)
    data=[]
    for path in [a.left_predictions,a.right_predictions]:
        with open(path,'rb') as f:data.append(pickle.load(f))
    poses=np.load(a.final_poses)['joints_3d']; times=np.load(a.timestamps)
    stamps=times['left']; n=len(poses)
    if any(len(e['claim'])!=n for e in data) or len(stamps)!=n or len(times['right'])!=n:
        raise ValueError('All synchronized inputs must have identical frame counts')
    if np.any(np.diff(stamps)<=0) or np.any(np.diff(times['right'])<=0) or np.any(np.abs(stamps-times['right'])>1_000_000):
        raise ValueError('Invalid synchronization timestamps')
    xyz=[]; evidence=[]; inliers=[]; rotations=[]; confidences=[]; statuses=[]
    for i,pose in enumerate(poses):
        pixels=[]; confidence=[]
        for view,e in enumerate(data):
            intr=e['intrinsics']; size=np.array([intr['image_width'],intr['image_height']])
            expected=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
            if not np.allclose(expected,cams[view].K): raise ValueError('Export/calibration intrinsics differ')
            uv=e['joints_2d'][i,1]*size
            c=np.full(21,e['exists_2d'][i,1] if e['claim'][i,1] else 0.)
            c[~(np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<size).all(1))]=0
            pixels.append(uv); confidence.append(c)
        confidence=np.stack(confidence,axis=1)
        r=fit_stereo_rigid(pose,*pixels,confidence,cams,config)
        x=r[a.mode]; xyz.append(x); evidence.append(r['stereo_valid']); inliers.append(r['anchor_inliers'])
        rotations.append(np.eye(3) if a.mode=='anchored' and np.isfinite(x).all() else r['rotation'])
        statuses.append(r['status']); confidences.append(np.where(np.isfinite(x).all(1),np.minimum(*confidence.T),0))
    xyz=np.array(xyz); valid=np.isfinite(xyz).all(-1); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    arrays=dict(joints_3d=xyz,validity=valid,confidence=confidences,timestamps=stamps,
        frame_ids=np.arange(n),rotation_delta=rotations,stereo_evidence=evidence,anchor_inliers=inliers,
        observation_type=np.where(valid,5,0).astype(np.uint8),statuses=statuses)
    np.savez_compressed(out/'predictions.npz',**arrays)
    for name in ['joints_3d','validity','confidence','timestamps']:np.save(out/(name+'.npy'),arrays[name])
    (out/'metadata.json').write_text(json.dumps(dict(mode=a.mode,config=asdict(config),
        coordinate_frame='provided calibration left camera optical; exactly the final-poses input frame',
        units='meters',hand_side='right',joint_order=JOINT_NAMES,
        articulation='fixed; rotation_delta is relative to input pose, not absolute wrist orientation',
        ground_truth_used=False,observation_types={'0':'missing','5':'fixed-articulation model inferred'},
        confidence_semantics='uncalibrated input hand existence scores; support is separately stored'),indent=2))


if __name__=='__main__':main()
