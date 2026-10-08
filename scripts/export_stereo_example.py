import argparse,json
from pathlib import Path
import cv2
import numpy as np
from handpose.data.hot3d import iter_clip
p=argparse.ArgumentParser();p.add_argument('--clip',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=30);p.add_argument('--upright',action='store_true');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);writers=[];timestamps=[]
for s in iter_clip(a.clip,a.frames):
    if not writers:
        for side in ('left','right'):writers.append(cv2.VideoWriter(str(out/f'{side}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,640)))
        calibration={'units':'meters','transform_convention':'T_camera_from_left'}
        for name,cam in zip(('left','right'),s['cameras']):calibration[name]={'K':cam.K.tolist(),'T_camera_from_left':cam.T_camera_from_left.tolist(),'distortion':cam.distortion.tolist(),'model':cam.model}
        if a.upright:
            R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
            H=np.eye(4);H[:3,:3]=R
            for name in ('left','right'):
                calibration[name]['T_camera_from_left']=(H@np.array(calibration[name]['T_camera_from_left'])@np.linalg.inv(H)).tolist()
        (out/'calibration.json').write_text(json.dumps(calibration,indent=2))
        (out/'cam.json').write_text(json.dumps({'image_width':640,'image_height':640,'frames':[{'intrinsics':{'fx':320,'fy':320,'cx':319.5,'cy':319.5}}]}))
    for w,side in zip(writers,('left','right')):w.write(cv2.rotate(s[side],cv2.ROTATE_90_CLOCKWISE) if a.upright else s[side])
    timestamps.append(s['timestamp_ns'])
for w in writers:w.release()
np.savez(out/'timestamps.npz',left=np.array(timestamps,np.int64),right=np.array(timestamps,np.int64))
