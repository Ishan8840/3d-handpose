"""No-GT robust spatial optimization; evaluation happens only after fitting."""
import argparse,json,time
from pathlib import Path
import numpy as np
from handpose.data.hot3d import iter_clip
from handpose.models.base import EDGES
from handpose.fitting.stereo import refine_frame
from handpose.evaluation.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--clip',required=True);p.add_argument('--output',required=True);p.add_argument('--prior');a=p.parse_args()
source=Path(a.source);pred=np.load(source/'predictions.npz');out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
prior=np.load(a.prior) if a.prior else None
if prior is not None and (not np.array_equal(pred['timestamps'],prior['timestamps']) or len(prior['joints_3d'])!=len(pred['joints_3d'])):raise ValueError('Prior and observations must share exact timestamps')
samples=list(iter_clip(a.clip,len(pred['timestamps'])))
bones={edge:float(np.nanmedian(np.linalg.norm(pred['joints_3d'][:,edge[0]]-pred['joints_3d'][:,edge[1]],axis=-1))) for edge in EDGES}
results={}
for aw,pw in ((0.,0.),(.1,0.),(1.,0.),(.1,.1)):
    if pw and prior is None:continue
    start=time.perf_counter();xyz=[]
    for i,s in enumerate(samples):
        xyz.append(refine_frame(pred['joints_3d'][i],pred['joints_2d_left'][i],pred['joints_2d_right'][i],pred['confidence'][i],*s['cameras'],prior=prior['joints_3d'][i] if prior is not None else None,prior_confidence=prior['confidence'][i] if prior is not None else None,anatomy_weight=aw,prior_weight=pw,bone_lengths=bones))
    name=f'anatomy{aw}_prior{pw}';xyz=np.array(xyz)
    np.savez_compressed(out/f'{name}.npz',joints_3d=xyz,validity=np.isfinite(xyz).all(-1),timestamps=pred['timestamps'],frame_ids=pred['frame_ids'],original_observation_type=pred['observation_type'])
    # Ground truth is only read after optimization completes.
    gt=np.load(source/'ground_truth.npz');m=evaluate(xyz,gt['joints_3d'],timestamps_ns=pred['timestamps']);m['seconds']=time.perf_counter()-start;results[name]=m
    print(name,m['absolute_mpjpe_mm'],m['capped_error_with_missing_penalty_mm'],flush=True)
(out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
