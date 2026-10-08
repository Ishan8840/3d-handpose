"""ACE predicted-observation SHOW3D evaluation with per-frame calibrated geometry."""
import json,subprocess,time
from pathlib import Path
import cv2
import numpy as np
from handpose.data.show3d import iter_scene
from handpose.models.ace_adapter import ACEExport
from handpose.models.base import Prediction
from handpose.inference.pipeline import reconstruct
from handpose.inference.cli import save_predictions
from handpose.fitting.stereo import refine_sequence
from handpose.temporal.smoothing import smooth
from handpose.evaluation.metrics import evaluate
base=Path.cwd()
for entry in json.load(open('configs/datasets/show3d.json')):
    if entry['split']!='development':continue
    name=Path(entry['path']).stem;out=base/'outputs/ace-show3d-dev'/name;out.mkdir(parents=True,exist_ok=True)
    samples=list(iter_scene(entry['path'],entry['annotations'],entry['max_frames']))
    valid=next(s for s in samples if s['cameras'] is not None);exports=[];start=time.perf_counter()
    for v,side in enumerate(('left','right')):
        video=out/f'{side}.mp4';h,w=samples[0][side].shape[:2]
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
        if not writer.isOpened():raise RuntimeError('Video encoder failed')
        for s in samples:writer.write(s[side])
        writer.release();K=valid['cameras'][v].K
        for s in samples:
            if s['cameras'] is not None:np.testing.assert_allclose(K,s['cameras'][v].K)
        cam=out/f'{side}.json';cam.write_text(json.dumps({'image_width':w,'image_height':h,'frames':[{'intrinsics':{'fx':K[0,0],'fy':K[1,1],'cx':K[0,2],'cy':K[1,2]}}]}))
        export=out/side/f'{side}.pkl'
        if not export.exists():
            with (out/f'{side}.log').open('w') as log:
                subprocess.run([str(base/'.venv-ace/bin/python'),'infer_video.py','--video',str(video),'--camera',str(cam),'--opt','options/ace_ego_hand_k.yml','--ckpt','checkpoints/ace_ego_hand_k.pt','--out',str(out/side),'--encode_w','640'],cwd=base/'third_party/ace',stdout=log,stderr=subprocess.STDOUT,check=True)
        exports.append(ACEExport(export,(w,h)))
    predictions=[]
    for i,s in enumerate(samples):
        if s['cameras'] is None:
            q=Prediction(s['timestamp_ns'],np.full((21,3),np.nan),np.full((21,2),np.nan),np.full((21,2),np.nan),np.zeros(21),np.zeros(21,bool),np.full((3,3),np.nan))
        else:q=reconstruct(*s['cameras'],exports[0].observations(i),exports[1].observations(i),s['timestamp_ns'])
        predictions.append(q)
    xyz=np.array([q.joints_3d_camera for q in predictions]);left=np.array([q.joints_2d_left for q in predictions]);right=np.array([q.joints_2d_right for q in predictions]);confidence=np.array([q.joint_confidence for q in predictions]);timestamps=np.array([q.timestamp_ns for q in predictions])
    original=np.isfinite(xyz).all(-1);cameras=[s['cameras'] or valid['cameras'] for s in samples]
    # Missing-camera frames remain excluded; substituted camera never has finite observations.
    xyz=refine_sequence(xyz,left,right,confidence,cameras,1.)
    xyz,provenance=smooth(xyz,confidence,timestamps,.1,.1)
    for i,(q,x) in enumerate(zip(predictions,xyz)):
        if samples[i]['cameras'] is None:x[:]=np.nan;provenance[i]=0
        q.joints_3d_camera=x;q.joint_validity=np.isfinite(x).all(-1)
    save_predictions(predictions,out,{'model':'ACE frozen spatial temporal','anatomy_weight':1.,'temporal_seconds':.1,'seconds':time.perf_counter()-start,'crop_source':'predicted','camera_geometry':'per-frame SHOW3D calibration'})
    arrays=dict(np.load(out/'predictions.npz'));arrays.update(observation_type=provenance,original_observation_type=original.astype(np.uint8));np.savez_compressed(out/'predictions.npz',**arrays)
    gt=np.array([s['ground_truth'] for s in samples]);np.savez_compressed(out/'ground_truth.npz',joints_3d=gt,timestamps=timestamps,frame_ids=np.arange(len(gt)),T_world_from_left=[s['T_world_from_left'] for s in samples])
    metrics=evaluate(xyz,gt,timestamps_ns=timestamps);(out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False));print(name,metrics['absolute_mpjpe_mm'],metrics['joint_coverage'],flush=True)
