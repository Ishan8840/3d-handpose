"""Calibrated WiLoR/HaMeR video inference, optionally fitted with MANO."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from handpose.geometry.camera import load_stereo
from handpose.models.wilor_adapter import MeshRegressor
from handpose.models.base import EDGES
from handpose.inference.pipeline import reconstruct
from handpose.inference.cli import save_predictions

def main(argv=None):
    p=argparse.ArgumentParser()
    for key in ('left','right','calibration','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--model',choices=['wilor','hamer'],default='wilor');p.add_argument('--checkpoint');p.add_argument('--rotate-cw',action='store_true');p.add_argument('--timestamps');p.add_argument('--mano-fit',choices=['none','adam','analytic_lm'],default='none')
    p.add_argument('--temporal-seconds',type=float,default=0.)
    a=p.parse_args(argv);cameras=load_stereo(a.calibration)
    if any(np.any(c.distortion) for c in cameras):raise ValueError('Undistort both videos and provide corresponding pinhole calibration first')
    caps=[cv2.VideoCapture(path) for path in (a.left,a.right)]
    if not all(c.isOpened() for c in caps):raise ValueError('Cannot open stereo videos')
    fps=[c.get(cv2.CAP_PROP_FPS) for c in caps]
    if min(fps)<=0 or abs(fps[0]-fps[1])>1e-4:raise ValueError('Mismatched FPS')
    times=np.load(a.timestamps) if a.timestamps else None
    if times is not None:
        if len(times['left'])!=len(times['right']) or any(np.any(np.diff(times[k])<=0) for k in ('left','right')) or np.any(abs(times['left']-times['right'])>1_000_000):raise ValueError('Invalid stereo timestamps')
    model=MeshRegressor(a.model,checkpoint=a.checkpoint);predictions=[];params=[];out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    try:
        while True:
            frames=[c.read() for c in caps]
            if frames[0][0]!=frames[1][0]:raise ValueError('Mismatched video lengths')
            if not frames[0][0]:break
            i=len(predictions)
            if times is not None and i>=len(times['left']):raise ValueError('Too few timestamps')
            t=int(times['left'][i]) if times is not None else round(i/fps[0]*1e9)
            details=[]
            for _,image in frames:
                h=image.shape[0];ds=model.predict(cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE) if a.rotate_cw else image)
                if a.rotate_cw:
                    for d in ds:
                        uv=d['observation'].pixels;d['observation'].pixels=np.stack([uv[:,1],h-1-uv[:,0]],-1)
                details.append(ds)
            predictions.append(reconstruct(*cameras,*[[d['observation'] for d in ds] for ds in details],t))
            params.append([{k:v.tolist() for k,v in d['mano'].items()} for d in details[0]])
        if not predictions:raise ValueError('No frames decoded')
        if times is not None and len(times['left'])!=len(predictions):raise ValueError('Too many timestamps')
    finally:
        for cap in caps:cap.release()
    initial=np.array([q.joint_validity for q in predictions]);diagnostics={}
    if a.mano_fit!='none':
        from handpose.fitting.wilor_mano import fit_mano_sequence
        obs={'joints_3d':np.array([q.joints_3d_camera for q in predictions]),'validity':initial,'joints_2d_left':np.array([q.joints_2d_left for q in predictions]),'joints_2d_right':np.array([q.joints_2d_right for q in predictions])}
        rotation=np.array([[0,1,0],[-1,0,0],[0,0,1]]) if a.rotate_cw else np.eye(3)
        xyz,diagnostics=fit_mano_sequence(obs,params,[cameras]*len(predictions),rotation,a.mano_fit)
        for q,x in zip(predictions,xyz):q.joints_3d_camera=x;q.joint_validity=np.isfinite(x).all(-1)
    provenance=np.where(np.array([q.joint_validity for q in predictions]),1 if a.mano_fit=='none' else 5,0).astype(np.uint8)
    before_temporal=provenance.copy()
    if a.temporal_seconds:
        from handpose.temporal.smoothing import smooth
        xyz,provenance=smooth(np.array([q.joints_3d_camera for q in predictions]),np.array([q.joint_validity for q in predictions],float),np.array([q.timestamp_ns for q in predictions]),a.temporal_seconds,.1)
        for q,x in zip(predictions,xyz):q.joints_3d_camera=x;q.joint_validity=np.isfinite(x).all(-1)
    save_predictions(predictions,out,{'model':a.model,'mano_fit':a.mano_fit,'temporal_seconds':a.temporal_seconds,'checkpoint':model.checkpoint,'checkpoint_load':model.load_status,'frames':len(predictions),'elapsed_seconds':time.perf_counter()-start,'fit_diagnostics':diagnostics,'input_rotation_clockwise':a.rotate_cw,'observation_code':1 if a.mano_fit=='none' else 5,'timestamp_source':'provided' if times is not None else 'nominal FPS; assumes externally synchronized videos'})
    arrays=dict(np.load(out/'predictions.npz'));arrays['original_observation_type']=initial.astype(np.uint8);arrays['observation_type']=provenance;arrays['pre_temporal_observation_type']=before_temporal;np.savez_compressed(out/'predictions.npz',**arrays)
    (out/'mano_parameters.json').write_text(json.dumps({'description':'predicted initializers, not fitted output parameters','frames':params}))
    cap=cv2.VideoCapture(a.left);writer=None
    try:
        for q in predictions:
            ok,image=cap.read()
            if not ok:raise ValueError('Video decode changed')
            uv=cameras[0].project(q.joints_3d_camera)
            for j,k in EDGES:
                if q.joint_validity[[j,k]].all() and np.isfinite(uv[[j,k]]).all() and np.abs(uv[[j,k]]).max()<1e6:cv2.line(image,tuple(np.rint(uv[j]).astype(int)),tuple(np.rint(uv[k]).astype(int)),(0,255,0),2)
            if writer is None:
                writer=cv2.VideoWriter(str(out/'visualization.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps[0],(image.shape[1],image.shape[0]))
                if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
            writer.write(image)
    finally:
        cap.release()
        if writer:writer.release()
