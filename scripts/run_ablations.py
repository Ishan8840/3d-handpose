import argparse,json,time
from pathlib import Path
import numpy as np
from handpose.evaluation.metrics import evaluate
from handpose.temporal.smoothing import smooth
from handpose.fitting.mano import fit_sequence
from handpose.data.hot3d import iter_clip
from handpose.data.show3d import iter_scene

p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--manifest',default='data/hot3d/manifest.json');p.add_argument('--output',required=True);p.add_argument('--mano',action='store_true');a=p.parse_args()
source=Path(a.predictions);pred=np.load(source/'predictions.npz');gt=np.load(source/'ground_truth.npz');out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
results={};start=time.perf_counter()
for window in (.033,.067,.1):
    xyz,provenance=smooth(pred['joints_3d'],pred['confidence'],pred['timestamps'],window,max_gap_seconds=.1)
    name=f'temporal_{window}'
    results[name]=evaluate(xyz,gt['joints_3d'],timestamps_ns=pred['timestamps'])
    np.savez_compressed(out/f'{name}.npz',joints_3d=xyz,validity=np.isfinite(xyz).all(-1),timestamps=pred['timestamps'],frame_ids=pred['frame_ids'],observation_type=provenance,original_observation_type=pred['observation_type'])
if a.mano:
    entry=next(x for x in json.load(open(a.manifest)) if Path(x['path']).stem==source.name)
    samples=list(iter_scene(entry['path'],entry['annotations'],entry['max_frames']) if 'annotations' in entry else iter_clip(entry['path'],entry['max_frames']))
    Ks=[np.stack([s['cameras'][v].K for s in samples]) for v in (0,1)];T=np.stack([s['cameras'][1].T_camera_from_left for s in samples])
    xyz,params=fit_sequence(pred['joints_3d'],pred['validity'],pred['joints_2d_left'],pred['joints_2d_right'],*Ks,T,'checkpoints/mano_converted')
    results['mano']=evaluate(xyz,gt['joints_3d'],timestamps_ns=pred['timestamps'])
    (out/'mano_parameters.json').write_text(json.dumps(params))
    np.savez_compressed(out/'mano.npz',joints_3d=xyz,validity=np.isfinite(xyz).all(-1),timestamps=pred['timestamps'],frame_ids=pred['frame_ids'],observation_type=np.where(np.isfinite(xyz).all(-1),5,0))
results['elapsed_seconds']=time.perf_counter()-start
(out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False));print(json.dumps(results),flush=True)
