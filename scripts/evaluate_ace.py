import argparse,json
from pathlib import Path
import numpy as np
from handpose.models.ace_adapter import ACEExport
from handpose.data.hot3d import iter_clip
from handpose.inference.pipeline import reconstruct
from handpose.inference.cli import save_predictions
from handpose.evaluation.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--clip',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=30);p.add_argument('--upright',action='store_true');a=p.parse_args()
l=ACEExport(a.left,(640,640),a.upright);r=ACEExport(a.right,(640,640),a.upright);preds=[];gt=[];mono=[]
for i,s in enumerate(iter_clip(a.clip,a.frames)):
    preds.append(reconstruct(*s['cameras'],l.observations(i),r.observations(i),s['timestamp_ns']))
    gt.append(s['ground_truth']);mono.append(l.direct(i))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);save_predictions(preds,out,{'model':'ace_stereo','checkpoint':'6fb3decade1b0331048fa9d02df605e95a0575c2','training_contamination':'HOT3D listed in ACE training datasets'})
gt=np.stack(gt);t=np.array([p.timestamp_ns for p in preds]);xyz=np.stack([p.joints_3d_camera for p in preds])
np.savez_compressed(out/'ground_truth.npz',joints_3d=gt,validity=np.isfinite(gt).all(-1),timestamps=t,frame_ids=np.arange(len(gt)))
np.savez_compressed(out/'monocular.npz',joints_3d=mono,validity=np.isfinite(mono).all(-1),timestamps=t,frame_ids=np.arange(len(gt)))
result={'A1_monocular':evaluate(mono,gt,timestamps_ns=t),'A3_stereo_epipolar':evaluate(xyz,gt,timestamps_ns=t)}
# A2 ungated DLT, retain cheirality and degeneracy protection.
raw=[]
for i,s in enumerate(iter_clip(a.clip,a.frames)):
    raw.append(reconstruct(*s['cameras'],l.observations(i),r.observations(i),s['timestamp_ns'],max_epipolar_px=float('inf'),max_reprojection_px=float('inf')).joints_3d_camera)
result['A2_stereo_ungated']=evaluate(raw,gt,timestamps_ns=t)
(out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)
