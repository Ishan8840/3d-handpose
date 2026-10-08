import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from handpose.data.hot3d import iter_clip
from handpose.data.show3d import iter_scene
from handpose.models.registry import create
from handpose.models.base import EDGES
from handpose.inference.pipeline import reconstruct
from handpose.inference.cli import save_predictions
from handpose.evaluation.metrics import evaluate


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',default='data/hot3d/manifest.json'); p.add_argument('--split',choices=['smoke','development','held_out'],default='smoke'); p.add_argument('--model',default='mediapipe'); p.add_argument('--output',required=True); p.add_argument('--seed-model',default='mediapipe')
    a=p.parse_args(); manifest=json.load(open(a.manifest)); model=create(a.seed_model if a.model=='umetrack' else a.model); direct=None
    if a.model=='umetrack':
        from handpose.models.umetrack_adapter import UmeTrackAdapter
        direct=UmeTrackAdapter()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    all_results=[]
    for entry in manifest:
        if entry['split']!=a.split: continue
        preds=[]; truths=[]; world=[]; visibilities=[]; latencies=[]; writer=None
        destination=out/Path(entry['path']).stem; destination.mkdir(exist_ok=True)
        samples=iter_scene(entry['path'],entry['annotations'],entry['max_frames']) if 'annotations' in entry else iter_clip(entry['path'],entry['max_frames'])
        for sample in samples:
            start=time.perf_counter()
            observations=[model.predict(sample[k]) for k in ('left','right')]
            pred=reconstruct(*sample['cameras'],*observations,sample['timestamp_ns']) if sample['cameras'] else reconstruct(None,None,[],[],sample['timestamp_ns'])
            if direct and sample['cameras']: pred=direct.predict_stereo(sample['left'],sample['right'],*sample['cameras'],pred)
            latencies.append(time.perf_counter()-start); preds.append(pred); truths.append(sample['ground_truth']); world.append(sample['T_world_from_left']); visibilities.append(sample['visibility'])
            image=sample['left'].copy()
            gtuv=sample['cameras'][0].project(sample['ground_truth']) if sample['cameras'] else np.full((21,2),np.nan)
            for points,color in ((gtuv,(0,0,255)),(pred.joints_2d_left,(0,255,0))):
                for j,k in EDGES:
                    if np.isfinite(points[[j,k]]).all(): cv2.line(image,tuple(np.rint(points[j]).astype(int)),tuple(np.rint(points[k]).astype(int)),color,1)
            if writer is None: writer=cv2.VideoWriter(str(destination/'visualization.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),60 if 'annotations' in entry else 30,(image.shape[1],image.shape[0]))
            writer.write(image)
        if writer: writer.release()
        save_predictions(preds,destination,dict(model=a.model,observation_code=4 if direct else 1,split=a.split,dataset=entry,ground_truth_convention='HOT3D UmeTrack 19 comparable joints; wrist and thumb CMC excluded',seconds_per_frame=float(np.mean(latencies))))
        gt=np.stack(truths); prediction=np.stack([x.joints_3d_camera for x in preds]); valid=np.stack([x.joint_validity for x in preds]); timestamps=np.array([x.timestamp_ns for x in preds])
        np.savez_compressed(destination/'ground_truth.npz',joints_3d=gt,validity=np.isfinite(gt).all(-1),timestamps=timestamps,frame_ids=np.arange(len(gt)),T_world_from_left=world)
        metrics=evaluate(prediction,gt,valid,timestamps_ns=timestamps)
        metrics.update(seconds_per_frame=float(np.mean(latencies)),frames=len(gt),model=a.model,split=a.split,sequence=entry['sequence'])
        (destination/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)); (destination/'visibility.json').write_text(json.dumps(visibilities))
        all_results.append(metrics); print(json.dumps(metrics),flush=True)
    model.close(); (out/'results.json').write_text(json.dumps(all_results,indent=2,allow_nan=False))
if __name__=='__main__': main()
