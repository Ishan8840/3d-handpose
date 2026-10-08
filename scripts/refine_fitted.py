"""Bounded temporal refinement of fitted poses, with explicit observation provenance."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from handpose.temporal.smoothing import smooth
from handpose.inference.cli import save_predictions
from handpose.models.base import Prediction
from handpose.evaluation.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--window',type=float,required=True);a=p.parse_args();ps=[];gs=[]
for source in sorted(Path(a.source).iterdir()):
 if not (source/'predictions.npz').exists():continue
 q=np.load(source/'predictions.npz');xyz,types=smooth(q['joints_3d'],q['validity'].astype(float),q['timestamps'],a.window,.1)
 out=Path(a.output)/source.name
 preds=[Prediction(int(t),x,l,r,np.where(np.isfinite(x).all(-1),c,0),np.isfinite(x).all(-1),np.full((3,3),np.nan)) for t,x,l,r,c in zip(q['timestamps'],xyz,q['joints_2d_left'],q['joints_2d_right'],q['confidence'])]
 save_predictions(preds,out,{'model':'WiLoR MANO Adam temporal','window_seconds':a.window,'max_gap_seconds':.1,'weighting':'uniform among finite fitted joints; scores not calibrated'})
 arrays=dict(np.load(out/'predictions.npz'));arrays.update(pre_temporal_observation_type=q['observation_type'],observation_type=types,original_observation_type=q['original_observation_type'] if 'original_observation_type' in q else q['observation_type']);np.savez_compressed(out/'predictions.npz',**arrays)
 for name in ('ground_truth.npz','ground_truth_mano21.npz'):
  if (source/name).exists():shutil.copy2(source/name,out/name)
 gt=np.load(out/'ground_truth.npz')['joints_3d'];ps.append(xyz);gs.append(gt)
 (out/'metrics.json').write_text(json.dumps(evaluate(xyz,gt,timestamps_ns=q['timestamps']),indent=2,allow_nan=False))
m=evaluate(np.concatenate(ps),np.concatenate(gs));(Path(a.output)/'summary.json').write_text(json.dumps(m,indent=2,allow_nan=False));print(a.window,m['absolute_mpjpe_mm'],m['joint_coverage'],m['capped_error_with_missing_penalty_mm'])
