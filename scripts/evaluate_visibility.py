"""Stratify completed predictions by dataset-provided hand visibility only."""
import argparse,json
from pathlib import Path
import numpy as np
from handpose.data.hot3d import iter_clip
from handpose.evaluation.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--clip',required=True);a=p.parse_args()
folder=Path(a.predictions);pred=np.load(folder/'predictions.npz');gt=np.load(folder/'ground_truth.npz');visibility=[];fov=[]
for sample in iter_clip(a.clip,len(pred['timestamps'])):
    v=sample['visibility'];visibility.append(min(v.values()) if v else np.nan)
    masks=[]
    for cam in sample['cameras']:
        uv=cam.project(sample['ground_truth']);masks.append(np.isfinite(uv).all(-1)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<640)&(uv[:,1]<640))
    fov.append(masks[0]&masks[1])
v=np.array(visibility);results={}
for name,mask in [('visibility_low',v<.3),('visibility_partial',(v>=.3)&(v<.8)),('visibility_high',v>=.8)]:
    masked=np.where(mask[:,None,None],gt['joints_3d'],np.nan)
    results[name]=evaluate(pred['joints_3d'],masked,pred['validity'])
results['inside_both_image_bounds']=evaluate(pred['joints_3d'],np.where(np.array(fov)[...,None],gt['joints_3d'],np.nan),pred['validity'])
(folder/'visibility_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False))
