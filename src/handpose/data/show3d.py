"""SHOW3D v3 reader; world is moving back-rig coordinates, not static world."""
import json
from pathlib import Path
import cv2
import numpy as np
from handpose.geometry.camera import Camera, transform
from .hot3d import TO_21


def iter_scene(path,annotations,max_frames=None):
    root=Path(path)
    calib=[json.loads((root/f'camera_calibration/headset{i}.json').read_text()) for i in range(2)]
    info=json.loads((root/'metadata/frame_info.json').read_text()); hands=json.load(open(annotations))
    caps=[cv2.VideoCapture(str(root/f'headset{i}.mp4')) for i in range(2)]
    if not all(c.isOpened() for c in caps): raise ValueError('Missing SHOW3D stereo videos')
    try:
        for record in info[:max_frames]:
            i=record['index']; key=str(i); frames=[c.read() for c in caps]
            if not all(x[0] for x in frames): raise ValueError('Video shorter than frame metadata')
            entries=[c['T_WorldFromCamera_by_index'][key] for c in calib]
            usable=all(e['is_pose_valid'] and e['T_WorldFromCamera'] is not None for e in entries)
            if not usable: # preserve time/frame count; no geometry from invalid calibration
                yield dict(frame_id=key,timestamp_ns=round(record['timestamp']*1e9),left=frames[0][1],right=frames[1][1],cameras=None,ground_truth=np.full((21,3),np.nan),visibility=None,info=record,T_world_from_left=np.full((4,4),np.nan))
                continue
            Ts=[np.asarray(e['T_WorldFromCamera'],float).copy() for e in entries]
            for T in Ts: T[:3,3]*=.001 # native rig translations and landmarks both millimeters
            cams=[]
            for c,T in zip(calib,Ts):
                if c['DistortionModel']!='PinholePlane': raise ValueError('Unexpected SHOW3D projection model')
                K=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1.]])
                cams.append(Camera(K,np.linalg.inv(T)@Ts[0]))
            hand=hands.get(key,{}).get('hand_poses',{}).get('1',{})
            gt=np.full((21,3),np.nan)
            if hand.get('confidence',0)>.1 and hand.get('landmarks_3d_mm') is not None:
                landmarks=np.asarray(hand['landmarks_3d_mm'],float)*.001
                for dst,src in TO_21.items(): gt[dst]=landmarks[src]
                gt=transform(gt,np.linalg.inv(Ts[0])); gt[0]=np.nan
                # Verify metric scale/transform against published projected annotations.
                reference=hand.get('landmarks_2d')
                if reference and reference.get('headset0'):
                    original=np.array([p if p is not None else [np.nan,np.nan] for p in reference['headset0']])
                    projected=cams[0].project(transform(landmarks,np.linalg.inv(Ts[0])))
                    good=np.isfinite(original).all(1)&np.isfinite(projected).all(1)
                    if good.any() and np.max(np.linalg.norm(projected[good]-original[good],axis=1))>1.:
                        raise ValueError('SHOW3D GT reprojection audit failed (>1 pixel)')
            missing=record.get('missing_cameras',[])
            yield dict(frame_id=key,timestamp_ns=round(record['timestamp']*1e9),left=frames[0][1],right=frames[1][1],cameras=None if any(s in missing for s in ('headset0','headset1')) else cams,ground_truth=gt,visibility=None,info=record,T_world_from_left=Ts[0])
    finally:
        for c in caps:c.release()
