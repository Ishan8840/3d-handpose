"""Apply frozen spatial/temporal settings without annotation-dependent optimization."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from handpose.data.hot3d import iter_clip
from handpose.fitting.stereo import refine_sequence
from handpose.temporal.smoothing import smooth
from handpose.inference.cli import save_predictions
from handpose.models.base import Prediction
from handpose.evaluation.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--anatomy',type=float,default=1.);p.add_argument('--temporal',type=float,default=.1);p.add_argument('--reuse-development-spatial',action='store_true');a=p.parse_args()
if a.reuse_development_spatial and a.split!='development':raise ValueError('Development reuse cannot apply to held-out data')
for e in json.load(open('data/hot3d/manifest.json')):
    if e['split']!=a.split:continue
    name=Path(e['path']).stem;source=Path(a.source)/name;out=Path(a.output)/name;out.mkdir(parents=True,exist_ok=True);pred=np.load(source/'predictions.npz')
    if a.reuse_development_spatial and name in ('clip-000175','clip-000838'):
        root='outputs/spatial-ace-dev' if name=='clip-000175' else 'outputs/spatial-ace-dev-838'
        xyz=np.load(Path(root)/f'anatomy{a.anatomy}_prior0.0.npz')['joints_3d']
    else:
        cameras=[s['cameras'] for s in iter_clip(e['path'],len(pred['timestamps']))]
        xyz=refine_sequence(pred['joints_3d'],pred['joints_2d_left'],pred['joints_2d_right'],pred['confidence'],cameras,a.anatomy)
    xyz,provenance=smooth(xyz,pred['confidence'],pred['timestamps'],a.temporal,max_gap_seconds=.1)
    predictions=[Prediction(int(t),x,l,r,c,np.isfinite(x).all(-1),np.full((3,3),np.nan)) for t,x,l,r,c in zip(pred['timestamps'],xyz,pred['joints_2d_left'],pred['joints_2d_right'],pred['confidence'])]
    save_predictions(predictions,out,{'model':'ace_spatial_temporal','anatomy_weight':a.anatomy,'temporal_seconds':a.temporal,'split':a.split,'source':str(source)})
    with np.load(out/'predictions.npz') as f:arrays=dict(f)
    arrays.update(observation_type=provenance,original_observation_type=pred['observation_type']);np.savez_compressed(out/'predictions.npz',**arrays)
    shutil.copy2(source/'ground_truth.npz',out/'ground_truth.npz');gt=np.load(out/'ground_truth.npz');m=evaluate(xyz,gt['joints_3d'],timestamps_ns=pred['timestamps']);(out/'metrics.json').write_text(json.dumps(m,indent=2,allow_nan=False));print(name,m['absolute_mpjpe_mm'],m['capped_error_with_missing_penalty_mm'],flush=True)
