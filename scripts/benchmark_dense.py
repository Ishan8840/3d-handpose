"""Dense surface-depth ablation; do not mistake a surface point for joint center."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from handpose.data.hot3d import iter_clip
from handpose.geometry.rectification import rectify,rotate_clockwise_camera
from handpose.models.foundationstereo_adapter import FoundationStereoAdapter
from handpose.models.mediapipe_adapter import MediaPipeAdapter
from handpose.inference.pipeline import reconstruct
from handpose.evaluation.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument('--clip',default='data/hot3d/clip-000000.tar');p.add_argument('--frames',type=int,default=30);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);model=FoundationStereoAdapter();det=MediaPipeAdapter();gt=[];stereo=[];dense=[];hybrid=[];start=time.perf_counter()
for i,s in enumerate(iter_clip(a.clip,a.frames)):
    original_l,original_r=s['cameras']
    l,Rup=rotate_clockwise_camera(original_l,640,640);r,_=rotate_clockwise_camera(original_r,640,640)
    Rl,Rr,Pl,Pr,Q,maps=rectify(l,r,(640,640))
    if abs(Pr[0,3])<1e-6:raise ValueError('FoundationStereo requires horizontal epipolar geometry')
    images=[cv2.remap(cv2.rotate(s[k],cv2.ROTATE_90_CLOCKWISE),*m,cv2.INTER_LINEAR) for k,m in zip(('left','right'),maps)]
    disp=model.predict(*images);np.save(out/f'disparity_{i:04d}.npy',disp)
    pred=reconstruct(original_l,original_r,det.predict(s['left']),det.predict(s['right']),s['timestamp_ns'])
    xyz=np.full((21,3),np.nan);fused=pred.joints_3d_camera.copy()
    if np.isfinite(pred.joints_2d_left).all():
        upright_uv=np.c_[639-pred.joints_2d_left[:,1],pred.joints_2d_left[:,0]]
        uv=cv2.undistortPoints(upright_uv.reshape(-1,1,2),l.K,l.distortion,R=Rl,P=Pl).reshape(-1,2)
        for j,(x,y) in enumerate(uv):
            ix,iy=int(round(x)),int(round(y))
            if not (2<=ix<638 and 2<=iy<638):continue
            patch=disp[iy-2:iy+3,ix-2:ix+3];d=float(np.median(patch))
            if not np.isfinite(d) or d<=0 or np.std(patch)>2:continue
            point=Q@np.array([x,y,d,1.]);point=point[:3]/point[3]
            xyz[j]=Rup.T@Rl.T@point
            if pred.joint_validity[j] and np.linalg.norm(xyz[j]-fused[j])<.03:fused[j]=.8*fused[j]+.2*xyz[j]
    gt.append(s['ground_truth']);stereo.append(pred.joints_3d_camera);dense.append(xyz);hybrid.append(fused)
result={name:evaluate(x,gt) for name,x in [('stereo',stereo),('surface_depth_at_landmarks',dense),('conservative_depth_fusion',hybrid)]};result['elapsed_seconds']=time.perf_counter()-start
(out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));np.savez_compressed(out/'predictions.npz',stereo=stereo,dense=dense,hybrid=hybrid,ground_truth=gt)
print(json.dumps(result),flush=True);det.close()
