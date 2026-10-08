"""Compare official mesh-model projected 2D stereo and calibrated monocular lift."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from handpose.data.hot3d import iter_clip
from handpose.data.show3d import iter_scene
from handpose.models.wilor_adapter import MeshRegressor
from handpose.models.base import Prediction
from handpose.inference.pipeline import reconstruct
from handpose.inference.cli import save_predictions
from handpose.evaluation.metrics import evaluate
from handpose.geometry.translation import translation_from_rays
from handpose.geometry.rectification import rotate_clockwise_camera

p=argparse.ArgumentParser();p.add_argument('--model',choices=['wilor','hamer','egoforce','omni'],default='wilor');p.add_argument('--manifest',default='configs/datasets/hot3d.json');p.add_argument('--split',default='smoke');p.add_argument('--output',required=True);p.add_argument('--max-frames',type=int)
p.add_argument('--checkpoint');p.add_argument('--forearm',action='store_true')
a=p.parse_args();base=Path.cwd()
if a.model=='egoforce':
    from handpose.models.egoforce_adapter import EgoForce
    model=EgoForce(base,forearm=a.forearm)
elif a.model=='omni':
    from handpose.models.omnihands_adapter import OmniHands
    model=OmniHands(base)
else:
    model=MeshRegressor(a.model,base,a.checkpoint)
entries=[e for e in json.loads(Path(a.manifest).read_text()) if e['split']==a.split]
for e in entries:
    count=min(e['max_frames'],a.max_frames or e['max_frames'])
    samples=iter_scene(e['path'],e['annotations'],count) if 'annotations' in e else iter_clip(e['path'],count)
    preds={'stereo':[],'mono':[]};gt=[];world=[];durations=[];params=[]
    for i,s in enumerate(samples):
        start=time.perf_counter();observations=[];details=[]
        if s['cameras'] is None:
            missing=Prediction(s['timestamp_ns'],np.full((21,3),np.nan),np.full((21,2),np.nan),np.full((21,2),np.nan),np.zeros(21),np.zeros(21,bool),np.full((3,3),np.nan))
            for values in preds.values():values.append(missing)
            params.append([]);gt.append(s['ground_truth']);world.append(s['T_world_from_left']);durations.append(time.perf_counter()-start)
            continue
        if a.model=='omni':
            multi_details=model.predict_views([cv2.rotate(s[key],cv2.ROTATE_90_CLOCKWISE) if 'annotations' not in e else s[key] for key in ('left','right')])
        for view,key in enumerate(('left','right')):
            image=s[key];h=image.shape[0]
            rotated='annotations' not in e
            input_image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE) if rotated else image
            cam=rotate_clockwise_camera(s['cameras'][view],image.shape[1],h)[0] if rotated else s['cameras'][view]
            if a.model=='omni':detections=multi_details[view]
            else:detections=model.predict(input_image,cam) if a.model=='egoforce' else model.predict(input_image)
            if rotated:
                for d in detections:
                    uv=d['observation'].pixels.copy();d['observation'].pixels=np.stack([uv[:,1],h-1-uv[:,0]],-1)
                    xyz_key='joints_camera' if a.model=='egoforce' else 'joints_relative'
                    xyz=d[xyz_key];d[xyz_key]=np.stack([xyz[:,1],-xyz[:,0],xyz[:,2]],-1)
            observations.append([d['observation'] for d in detections]);details.append(detections)
        stereo=reconstruct(*s['cameras'],*observations,s['timestamp_ns'])
        mono=reconstruct(*s['cameras'],[],[],s['timestamp_ns'])
        if details[0]:
            d=max(details[0],key=lambda d:d['observation'].confidence.mean())
            uv=d['observation'].pixels
            if a.model=='egoforce':xyz=d['joints_camera'].copy()
            else:
                xyz=d['joints_relative'];xyz=xyz+translation_from_rays(xyz,uv,s['cameras'][0])
            valid=np.isfinite(xyz).all(-1)&(xyz[:,2]>0);xyz[~valid]=np.nan
            mono=Prediction(s['timestamp_ns'],xyz,uv,np.full((21,2),np.nan),d['observation'].confidence*valid,valid,np.full((3,3),np.nan)).validate()
        preds['stereo'].append(stereo);preds['mono'].append(mono)
        params.append([{k:v.tolist() for k,v in d['mano'].items()} for d in details[0]])
        gt.append(s['ground_truth']);world.append(s['T_world_from_left']);durations.append(time.perf_counter()-start)
        if i%30==0:print(a.model,e['sequence'],i,int(stereo.joint_validity.sum()),flush=True)
    for kind,ps in preds.items():
        out=Path(a.output)/kind/Path(e['path']).stem;out.mkdir(parents=True,exist_ok=True)
        metadata=dict(model=a.model,checkpoint=model.checkpoint,split=a.split,detector='WiLoR YOLO conf0.3 right class1',observation_code=1 if kind=='stereo' else (4 if a.model=='omni' else 2),
                      crop_source='predicted',forearm_detector=a.forearm,landmarks='native learned 2D head' if a.model=='egoforce' else 'projected from reconstructed mesh, not independent heatmaps',
                      translation='official ray-space solver' if a.model=='egoforce' else 'least squares known-intrinsic rays and metric oriented MANO joints',
                      input_views=2 if a.model=='omni' or kind=='stereo' else 1,
                      checkpoint_load=model.load_status,seconds_per_frame_both_methods=float(np.mean(durations)))
        save_predictions(ps,out,metadata)
        np.savez_compressed(out/'ground_truth.npz',joints_3d=gt,validity=np.isfinite(gt).all(-1),timestamps=[x.timestamp_ns for x in ps],frame_ids=np.arange(len(ps)),T_world_from_left=world)
        metrics=evaluate(np.array([x.joints_3d_camera for x in ps]),np.asarray(gt),np.array([x.joint_validity for x in ps]),timestamps_ns=np.array([x.timestamp_ns for x in ps]))
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False));(out/'mano_parameters.json').write_text(json.dumps(params))
        print(kind,metrics['absolute_mpjpe_mm'],metrics['joint_coverage'],flush=True)
