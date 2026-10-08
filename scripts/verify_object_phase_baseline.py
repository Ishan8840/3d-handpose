"""Recompute the frozen prior results from immutable saved predictions."""
import hashlib,json
from pathlib import Path
import numpy as np
from handpose.evaluation.metrics import evaluate
checks=[('HOT3D MANO-21','reports/expanded_held_mano21/comparison.json','configs/experiments/expanded_held_runs.json','ground_truth_mano21.npz',['WiLoR stereo','HaMeR stereo','ACE hybrid','WiLoR MANO Adam temporal']),('SHOW3D common19','reports/expanded_show3d_comparison/comparison.json','configs/experiments/expanded_show3d_runs.json','ground_truth.npz',['WiLoR MANO Adam temporal'])]
records=[]
for track,report,mapping,gtname,names in checks:
 refs=json.load(open(report))['models'];runs=json.load(open(mapping))
 for name in names:
  ps=[];gs=[];files=[]
  for folder in sorted(Path(runs[name]).iterdir()):
   if not (folder/'predictions.npz').exists():continue
   p=np.load(folder/'predictions.npz');g=np.load(folder/gtname)
   assert np.array_equal(p['timestamps'],g['timestamps']);assert np.array_equal(p['frame_ids'],g['frame_ids'])
   ps.append(np.where(p['validity'][...,None],p['joints_3d'],np.nan));gs.append(g['joints_3d'])
   for f in (folder/'predictions.npz',folder/gtname):files.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
  pred=np.concatenate(ps);gt=np.concatenate(gs);m=evaluate(pred,gt)
  independent=float(np.nanmean(np.linalg.norm(pred-gt,axis=-1))*1000)
  np.testing.assert_allclose(independent,m['absolute_mpjpe_mm'],atol=1e-10)
  keys=['absolute_mpjpe_mm','joint_coverage','wrist_mm','fingertips_mm','p90_mm','p95_mm','capped_error_with_missing_penalty_mm']
  for key in keys:
   if refs[name][key] is not None:np.testing.assert_allclose(m[key],refs[name][key],rtol=0,atol=1e-8)
  records.append({'track':track,'model':name,'sequences':len(ps),'frames':len(pred),'metrics':{k:m[k] for k in keys},'files':files,'status':'reproduced from archived predictions; not yet fresh model inference'})
  print(track,name,m['absolute_mpjpe_mm'],m['joint_coverage'])
Path('reports/object_aware/baseline_verification.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
