import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from handpose.geometry.camera import load_stereo
from handpose.models.registry import create
from handpose.models.base import JOINT_NAMES, EDGES
from .pipeline import reconstruct


def save_predictions(predictions, output, metadata):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    xyz=np.stack([p.joints_3d_camera for p in predictions]); valid=np.stack([p.joint_validity for p in predictions])
    wrist=np.full((len(predictions),4,4),np.nan)
    wrist[:,:3,3]=xyz[:,0]; wrist[:,:3,:3]=np.stack([p.wrist_rotation for p in predictions]); wrist[:,3]=[0,0,0,1]
    arrays=dict(joints_3d=xyz,timestamps=np.array([p.timestamp_ns for p in predictions],np.int64),confidence=np.stack([p.joint_confidence for p in predictions]),validity=valid,wrist_poses=wrist,frame_ids=np.arange(len(predictions)),joints_2d_left=np.stack([p.joints_2d_left for p in predictions]),joints_2d_right=np.stack([p.joints_2d_right for p in predictions]),observation_type=np.where(valid,1,0).astype(np.uint8))
    for name in ('joints_3d','timestamps','confidence','validity','wrist_poses'):
        np.save(out/f'{name}.npy',arrays[name])
    np.savez_compressed(out/'predictions.npz',**arrays)
    metadata.update(units='meters',coordinate_frame='left_camera_optical',axes='+x right,+y down,+z forward',joint_order=JOINT_NAMES,observation_types={'0':'missing','1':'stereo_observed','2':'single_view_inferred','3':'temporal'},wrist_rotation='unavailable: NaN',confidence_semantics='hand-level handedness score, gated by stereo validity; not calibrated joint probability')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser()
    for name in ('left','right','calibration','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--model',default='mediapipe'); p.add_argument('--timestamps',help='NPZ with left/right int64 nanoseconds, one per decoded frame')
    p.add_argument('--sync-tolerance-ns',type=int,default=1000000)
    a=p.parse_args(); left,right=load_stereo(a.calibration)
    caps=[cv2.VideoCapture(a.left),cv2.VideoCapture(a.right)]
    if not all(c.isOpened() for c in caps): raise ValueError('Cannot open stereo videos')
    fps=[c.get(cv2.CAP_PROP_FPS) for c in caps]
    if min(fps)<=0 or abs(fps[0]-fps[1])>1e-4: raise ValueError('Videos must have matching valid FPS')
    times=np.load(a.timestamps) if a.timestamps else None
    if times is not None:
        if len(times['left'])!=len(times['right']) or np.any(np.diff(times['left'])<=0) or np.any(np.diff(times['right'])<=0): raise ValueError('Invalid timestamps')
        if np.any(np.abs(times['left']-times['right'])>a.sync_tolerance_ns): raise ValueError('Stereo timestamp mismatch')
    model=create(a.model); predictions=[]; writer=None; start=time.perf_counter()
    Path(a.output).mkdir(parents=True,exist_ok=True)
    try:
        while True:
            frames=[c.read() for c in caps]
            if frames[0][0]!=frames[1][0]: raise ValueError('Stereo frame counts differ')
            if not frames[0][0]: break
            i=len(predictions)
            timestamp=int(times['left'][i]) if times is not None else round(i/fps[0]*1e9)
            images=[f[1] for f in frames]
            pred=reconstruct(left,right,*[model.predict(im) for im in images],timestamp)
            predictions.append(pred)
            vis=images[0].copy()
            for j,k in EDGES:
                if pred.joint_validity[j] and pred.joint_validity[k]: cv2.line(vis,tuple(np.rint(pred.joints_2d_left[j]).astype(int)),tuple(np.rint(pred.joints_2d_left[k]).astype(int)),(0,255,0),2)
            if writer is None:
                writer=cv2.VideoWriter(str(Path(a.output)/'visualization.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps[0],(vis.shape[1],vis.shape[0]))
                if not writer.isOpened(): raise RuntimeError('Video encoder unavailable')
            writer.write(vis)
        if not predictions: raise ValueError('No decoded frames')
        if times is not None and len(times['left'])!=len(predictions): raise ValueError('Timestamp count differs from decoded frames')
        save_predictions(predictions,a.output,dict(model=a.model,frames=len(predictions),elapsed_seconds=time.perf_counter()-start,calibration_sha256=hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest(),timestamp_source='provided' if times is not None else 'nominal FPS; assumes externally synchronized videos'))
    finally:
        for c in caps: c.release()
        if writer: writer.release()
        model.close()
